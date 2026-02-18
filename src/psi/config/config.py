from __future__ import annotations
import os
from typing_extensions import Self, Annotated
from pydantic import BaseModel, Field, model_validator
import os
import tyro
from typing import Union, Any, TYPE_CHECKING
from tyro.conf import subcommand as cmd
from pathlib import Path
import datetime
from psi.config import data as wd
from psi.config import model as wm

if TYPE_CHECKING:
    from .tokenizer import VQVaeActionTokenizerConfig

class LoggingConfig(BaseModel):
    logging_dir: str = "logs"
    report_to: str | None = None
    log_freq: int = 100


class WandbConfig(BaseModel):
    project: str = "hfm"
    entity: str | None = None
    group: str | None = None
    id: str | None = None
    name: str | None = None
    resume: str = "allow"  # allow, must, never

    def model_post_init(self, __context: Any) -> None:
        if self.entity is None:
            self.entity = os.getenv("WANDB_ENTITY", None)


class TrainConfig(BaseModel):
    num_workers: int = 8
    overfit_single_batch: bool = False
    name: str = "human3d"  # "vqvae"
    resume_from_checkpoint: str | None = None
    skip_resumed_steps: bool = False

    # HF Hub Credentials (for any gated models)
    hf_token: str | Path = Path(".hf_token")  # Environment variable or Path to HF Token

    lora: bool = False
    output_dir: str = ".runs"
    gradient_accumulation_steps: int = 1
    mixed_precision: str = "bf16"  # "no" for fp32, "fp16", "bf16"
    max_grad_norm: float | None = None  # 1.0

    train_batch_size: int = 16
    val_batch_size: int = 16

    val_num_batches: int = 20

    checkpointing_steps: int = 5000
    max_checkpoints_to_keep: int | None = None
    validation_steps: int = 50

    learning_rate: float = 1e-5
    # linear, cosine, cosine_with_restarts, polynomial, constant, constant_with_warmup
    lr_scheduler_type: str = "cosine_with_min_lr"
    lr_scheduler_kwargs: dict[str, float | tuple[float, ...]] = Field(
        default_factory=lambda: {
            # "num_warmup_steps": 0,
            "betas": (0.9, 0.95),
            "weight_decay": 1e-8,
            "eps": 1e-8,
        }
    )
    scheduler_specific_kwargs: dict[str, float | tuple[float, ...]] = Field(
        default_factory=lambda: {"min_lr": 5.0e-07}
    )

    ## FSDP or DDP
    data_parallel: str = "ddp"  # "deepspeed", "ddp" or "fsdp"
    sharding_strategy: str = "full-shard"
    deepspeed_config: str = "src/InternVLA/config/deepseeds/zero3.json"
    enable_gradient_checkpointing: bool = True
    enable_mixed_precision_training: bool = True
    reduce_in_full_precision: bool = True

    max_training_steps: int | None = 100_000
    num_train_epochs: int | None = None
    warmup_steps: int | None = 0  # if > 0, overrides warmup_ratio
    warmup_ratio: float | None = 0.05  # used if warmup_steps is not set or 0

    @model_validator(mode="after")
    def check_warmup(self) -> Self:
        steps, ratio = self.warmup_steps, self.warmup_ratio
        if (steps is not None and steps > 0) and (ratio is not None and ratio > 0):
            raise ValueError("Only one of warmup_steps or warmup_ratio can be set")

        steps, ratio = self.max_training_steps, self.num_train_epochs
        if (steps is None or steps == 0) and (ratio is None or ratio == 0):
            raise ValueError(
                "At least one of max_training_steps or num_train_epochs must be set"
            )
        if steps is not None and ratio is not None and steps > 0 and ratio > 0:
            raise ValueError(
                "Only one of max_training_steps or num_train_epochs can be set"
            )
        return self

    def model_post_init(self, __context: Any) -> None:
        if self.lr_scheduler_type != "cosine_with_min_lr":
            self.scheduler_specific_kwargs = {}

        if not os.path.isabs(self.deepspeed_config):
            from psi.utils import resolve_path
            self.deepspeed_config = os.path.abspath(
                resolve_path(self.deepspeed_config)
            )

class ServerConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 21074
    device: str = "cuda:0"
    policy: str | None = None
    action_exec_horizon: int | None = None
    rtc: bool = False
    run_dir: str 
    ckpt_step: int 

    @model_validator(mode="after")
    def set_policy(self):
        if self.policy is None:
            run_dir_path = Path(self.run_dir)
            self.policy = run_dir_path.parts[1]
        return self
        

class LaunchConfig(BaseModel):
    # NOTE: This class is used for type hinting only!
    # The real implmentation is DynamicLaunchConfig .

    exp: str = Field(..., description="Name of the experiment", frozen=True)
    seed: int | None = None
    auto_tag_run: bool = False
    eval: bool = False
    debug: bool = False
    timestamp: str | None = None

    log: LoggingConfig
    wandb: WandbConfig
    train: TrainConfig
    # fmt: off
    data: Union[
        Annotated[wd.DummyDataConfig, cmd("dummy")],
        Annotated[wd.EgoDexDataConfig, cmd("egodex")],
        Annotated[wd.RLDSDataConfig, cmd("rlds")],
        Annotated[wd.LerobotDataConfig, cmd("lerobot")],
        Annotated[wd.PushTDataConfig, cmd("pusht")],
        Annotated[wd.HumanoidDataConfig, cmd("humanoid")],
        Annotated[wd.HEDataConfig, cmd("humanoideveryday")],
        Annotated[wd.HERawDataConfig, cmd("humanoideveryday-raw")],
        Annotated[wd.HRDTDataConfig, cmd("hrdt")],
        Annotated[wd.VLMDataConfig, cmd("toy-vlm")],
        Annotated[wd.GR00TDataConfig, cmd("gr00t")],
        Annotated[wd.MixedDataConfig, cmd("mixed")],
        Annotated[wd.SimpleDataConfig, cmd("simple")]
    ]
    model: Union[
        Annotated[wm.DummyModelConfig, cmd("dummy")],
        Annotated[wm.InternVLA_M1_ModelConfig, cmd("internvla")],
        Annotated[wm.Qwen25VL_ModelConfig, cmd("qwen25vl")],
        Annotated[wm.OpenVLA_Qwenvl_ModelConfig, cmd("openvla_qwenvl")],
        Annotated[wm.OpenVLA_ModelConfig, cmd("openvla")],
        # Annotated[wm.VQVaeActionTokenizerConfig, cmd("vqvae")],
        Annotated[wm.OpenVLA_Flow_ModelConfig, cmd("openvla-flow")],
        Annotated[wm.DitPolicy_ModelConfig, cmd("ditp")],
        Annotated[wm.Vlt_ModelConfig, cmd("vlt")],
        Annotated[wm.DiffusionPolicy_ModelConfig, cmd("dp")],
        Annotated[wm.Hfm_Qwen3VL_ModelConfig, cmd("hfm-qwen3vl")],
        Annotated[wm.Qwen3vl_ModelConfig, cmd("qwen3vl")],
        Annotated[wm.Qwen3vl_7d_ModelConfig, cmd("qwen3vl-7d")],
        Annotated[wm.Hfm_Action_ModelConfig, cmd("hfm-action")],
        Annotated[wm.Together_ModelConfig, cmd("hfm-together")]
    ]
    # fmt: on
