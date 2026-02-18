import os
import re
import sys
import inspect
import datetime
import importlib
import importlib.util
from pydantic import Field, BaseModel
from typing import Any, Union, Annotated, Optional, get_args, get_origin
from psi.config.config import LaunchConfig as OriginalLaunchConfig

# Extract Union members from DataTransform fields dynamically using inspect
def extract_union_configs(union_type):
    """Extract {name: Annotated[Type, cmd(name)]} from a Union type."""
    configs = {}
    for arg in get_args(union_type):
        if hasattr(arg, "__metadata__"):  # This is an Annotated type
            # Get the command name from metadata
            for meta in arg.__metadata__:
                if hasattr(meta, "name"):  # tyro subcommand
                    configs[meta.name] = arg
                    break
    return configs

"""Dynamically create a LaunchConfig with only the needed Union members."""
def create_dynamic_config_class(model_type: str, DataConfigClass: BaseModel):
    model_union = OriginalLaunchConfig.model_fields["model"].annotation
    model_configs = extract_union_configs(model_union)
    
    from psi.config.config import (
        LoggingConfig, WandbConfig, TrainConfig
        # VQVaeActionTokenizerConfig
    )

    # Create dynamic LaunchConfig
    class DynamicLaunchConfig(BaseModel):
        exp: str = Field(..., description="Name of the experiment", frozen=True)
        seed: int | None = None
        auto_tag_run: bool = False
        eval: bool = False
        debug: bool = False
        timestamp: str | None = None

        log: LoggingConfig
        wandb: WandbConfig
        train: TrainConfig
        data: DataConfigClass
        model: model_configs[model_type] #.get(model_type, DummyModelConfig) #type:ignore
        # tokenizer: VQVaeActionTokenizerConfig

        def model_post_init(self, __context: Any) -> None:
            is_multinode = (
                "SLURM_NODELIST" in os.environ and
                len(os.environ["SLURM_NODELIST"].split(",")) > 1
            )
            if is_multinode:
                assert self.timestamp is not None, "Timestamp must be provided for multi-node training, eg., --timestamp=$(date +\"%y%m%d%H%M\")"

            def extract_timestamp(folder_name):
                parts = folder_name.split('.')
                return parts[-1] if len(parts) > 1 else ''

            if self.train.resume_from_checkpoint == "latest":
                """ auto resume by looking up timestamp or latest run folder """
                auto_resume_success = False
                trainer_dir = os.path.join(self.train.output_dir, self.train.name)
                if os.path.exists(trainer_dir):
                    # Sort folders by timestamp (assumed to be last part after a dot)
                    runs = dict(sorted({
                        extract_timestamp(f): os.path.join(trainer_dir, f) for f in os.listdir(trainer_dir)
                    }.items(), reverse=True))

                    if self.timestamp is not None and self.timestamp in runs:
                        print(f"Will resume latest run with specified timestamp: {self.timestamp}")
                        self.train.resume_from_checkpoint = runs[self.timestamp]
                        auto_resume_success = True

                    """ elif len(runs) > 0:
                        latest_timestamp = next(iter(runs))
                        print(f"Will auto-resume from latest run with timestamp: {latest_timestamp}")
                        if self.timestamp is not None and self.timestamp != latest_timestamp:
                            print(f"Overriding timestamp {self.timestamp} with latest timestamp {latest_timestamp}")
                        self.timestamp = latest_timestamp
                        self.train.resume_from_checkpoint = runs[latest_timestamp]
                        auto_resume_success = True """
               
                if not auto_resume_success:
                    self.train.resume_from_checkpoint = None

            if self.timestamp is None:
                self.timestamp = datetime.datetime.now().strftime("%y%m%d%H%M")

    return DynamicLaunchConfig

def parse_dynamic_config_types(argv):
    parsed = {}
    remaining_argv = [argv[0]] if argv else []

    for arg in argv[1:]:
        if ":" in arg and not arg.startswith("-") and not arg.startswith("model."):
            key, val = arg.split(":", 1)
            parsed[key] = val
        elif arg.startswith("--deepspeed"):
            continue # skip deepspeed arg
        else:
            remaining_argv.append(arg)

    # Sort by key depth (number of dots + 1), descending order (deepest first)
    parsed = dict(sorted(parsed.items(), key=lambda x: x[0].count("."), reverse=False))
    return parsed, remaining_argv

def parse_data_config_args(argv):
    data_config_args = []
    remaining_argv = [argv[0]] if argv else []

    i = 1  # skip script name
    while i < len(argv):
        arg = argv[i]
        if arg.startswith("--data."):
            # --data.* argument; include its value if present
            data_config_args.append(arg) # .replace("data.", "")
            i += 1
            # Check for multiple consecutive args that are values (not flags)
            while i < len(argv) and not argv[i].startswith("-"):
                data_config_args.append(argv[i]) # .replace("data.", "") remove data. prefix for tyro
                i += 1
        else:
            remaining_argv.append(arg)
            i += 1
    return data_config_args, remaining_argv


def dynamic_tyro_args(force_rewrite_config_file: bool = False):
    dynamic_types, argv = parse_dynamic_config_types(sys.argv)

    config_class_hierarchy = {}
    for k, v in dynamic_types.items():
        if not k.startswith("data"): continue
        # print(f"parsing dynamic arg: --{k}")
        # Get the union definition for LaunchConfig.{k} if it exists
        if k in OriginalLaunchConfig.model_fields:
            union_type = OriginalLaunchConfig.model_fields[k].annotation
            configs = extract_union_configs(union_type)
            if v not in configs:
                raise ValueError(f"Unknown config type '{v}' for '{k}'. Available: {list(configs.keys())}")
            clazz = configs[v]
            # Extract the actual class from Annotated type
            actual_class = get_args(clazz)[0]  # e.g., psi.config.data.EgoDexDataConfig
            config_class_hierarchy[k] = actual_class
        else:
            config_paths = k.split(".")
            for i in range(len(config_paths)):
                path = ".".join(config_paths[:i+1])
                if path in config_class_hierarchy:
                    continue

                parent_class = config_class_hierarchy[".".join(config_paths[:i])]
                field_type = parent_class.model_fields[config_paths[i].replace("-", "_")].annotation
                if get_origin(field_type) is Union:
                    configs = extract_union_configs(field_type)
                    if v not in configs:
                        raise ValueError(f"Unknown config type '{v}' for '{path}'. Available: {list(configs.keys())}")
                    config_class_hierarchy[path] = get_args(configs[v])[0]
                else:
                    config_class_hierarchy[path] = field_type

    if "data" not in config_class_hierarchy:
        raise ValueError("maybe you forget to pass args like --data.xxx=...")
    # Get the source code of the class
    source_code = inspect.getsource(config_class_hierarchy["data"])

    # Generate a dynamic config module
    def generate_dynamic_config_module(config_hierarchy: dict, output_path: str):
        """Generate a .py file with the dynamically resolved config classes.
        
        Args:
            config_hierarchy: dict mapping paths like 'data', 'data.transform.repack' to Annotated types
            output_path: path to write the generated .py file
        """
        imports = set()
        
        # Collect all classes we need to import
        for path, annotated_type in config_hierarchy.items():
            # Extract the actual class from Annotated type
            if hasattr(annotated_type, "__origin__") and annotated_type.__origin__ is Annotated:
                actual_class = get_args(annotated_type)[0]
            else:
                actual_class = annotated_type
            
            module = actual_class.__module__
            class_name = actual_class.__name__
            if module == "psi.config.data":
                # Skip importing from the same module we're generating
                continue
            imports.add(f"from {module} import {class_name}")
        
        # Build the imports section
        import_lines = sorted(imports)
        
        # Build the dynamic config class
        # We need to build nested Pydantic models
        leaf_paths = {}  # path -> class_name
        for path, annotated_type in config_hierarchy.items():
            if hasattr(annotated_type, "__origin__") and annotated_type.__origin__ is Annotated:
                actual_class = get_args(annotated_type)[0]
            else:
                actual_class = annotated_type
            leaf_paths[path] = actual_class.__name__
        
        # Generate the module content
        content = '''"""Auto-generated dynamic config module."""
from __future__ import annotations
import os, sys
import torch
import numpy as np
from pydantic import BaseModel, Field
from typing import TYPE_CHECKING, Any, List, Optional, Union
from psi.data.dataset import Dataset as MapStyleDataset, IterableDataset

{imports}

# Resolved config class hierarchy:
# {hierarchy_comment}

class DynamicDataTransform(BaseModel):
    """Dynamically resolved DataTransform with specific implementations."""
    repack: {repack_class}
    action_state: {action_state_class}
    model: {model_class}
    
    def __call__(self, data, **kwargs):
        data = self.repack(data, **kwargs)
        data = self.action_state(data, **kwargs)
        data = self.model(data, **kwargs)
        return data

class DataConfig(BaseModel):
    transform: DynamicDataTransform

{data_config_source}

'''.format(
            imports="\n".join(import_lines),
            hierarchy_comment=str({k: v.__name__ if hasattr(v, '__name__') else str(v) for k, v in leaf_paths.items()}),
            repack_class=leaf_paths.get("data.transform.repack", "IdentityTransform"),
            action_state_class=leaf_paths.get("data.transform.action-state", "IdentityTransform"),
            model_class=leaf_paths.get("data.transform.model", "IdentityTransform"),
            # data_class=leaf_paths.get("data", "BaseModel"),
            data_config_source=source_code
        )
        
        # Write to file
        with open(output_path, "w") as f:
            f.write(content)
        return output_path
    
    # Generate the module
    # import tempfile
    output_dir = os.path.join(os.path.dirname(__file__), "train")
    os.makedirs(output_dir, exist_ok=True)
    data_class_name = config_class_hierarchy["data"].__name__
    dynamic_config_module_name = f"dynamic_config_{_get_trainer_name(argv)}_{data_class_name}"
    output_path = os.path.join(output_dir, f"{dynamic_config_module_name}.py")
    if force_rewrite_config_file or not os.path.exists(output_path):
        generate_dynamic_config_module(config_class_hierarchy, output_path)
        print(f"Generated dynamic config module at: {output_path}")
        
    # Add the output directory to sys.path so worker processes can import dynamic_config
    if output_dir not in sys.path:
        sys.path.insert(0, output_dir)
    
    # Import the generated module dynamically
    spec = importlib.util.spec_from_file_location(dynamic_config_module_name, output_path)
    assert spec is not None
    dynamic_config_module = importlib.util.module_from_spec(spec)
    # Register in sys.modules so inspect.getsource() can find it
    sys.modules[dynamic_config_module_name] = dynamic_config_module
    assert spec.loader is not None
    spec.loader.exec_module(dynamic_config_module)
    
    # Access the generated classes using the resolved class name from hierarchy
    DataConfigClass = getattr(dynamic_config_module, data_class_name)

    # Fix __module__ attribute so tyro/inspect can find the source
    DataConfigClass.__module__ = dynamic_config_module_name
    return dynamic_types, DataConfigClass, argv

def _get_trainer_name(argv):
    import re
    # If argv contains a --train.name=xxx entry, extract and convert to CamelCase
    train_name = None
    for a in argv:
        if a.startswith("--train.name="):
            train_name = a.split("=", 1)[1]
            break

    if train_name:
        # Convert train_name to CamelCase (e.g., "my-name" -> "MyName")
        parts = re.split(r"[^0-9a-zA-Z]+", train_name)
        camel_name = "_".join(p.lower() for p in parts if p)
        return camel_name
    
    return "default"
