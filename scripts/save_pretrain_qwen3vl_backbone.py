import dotenv
dotenv.load_dotenv()

import torch
import numpy as np
from pathlib import Path
from psi.utils import parse_args_to_tyro_config, seed_everything #, move_to_device
from psi.config.config import LaunchConfig
from psi.config.tokenizer import FastActionTokenizerConfig

ckpt_step = 200000
run_dir = Path(".runs/hfm/pre.fast.egodex.delta.c1.const.lr1.0e-04.b1024.gpus64.2512241941")
config:LaunchConfig = parse_args_to_tyro_config(run_dir / "argv.txt") # type: ignore
conf = (run_dir / "run_config.json").open("r").read()
launch_config = config.model_validate_json(conf)
seed_everything(config.seed or 42)

from psi.config.data_lerobot import LerobotDataConfig
data_cfg: LerobotDataConfig = config.data # type: ignore

from psi.config.model_qwen3vl import Qwen3VLModelConfig
model_cfg: Qwen3VLModelConfig = config.model # type: ignore

from transformers import Qwen3VLForConditionalGeneration, AutoProcessor #, Qwen2TokenizerFast, Qwen3VLProcessor
# from psi.tokenizer import FastActionTokenizer
# from qwen_vl_utils import process_vision_info

vlm_processor = AutoProcessor.from_pretrained(model_cfg.model_name_or_path)
tokenizer = vlm_processor.tokenizer

DEVICE = "cuda:0"
print(f"Using device: {DEVICE}")
print(f"GPU name: {torch.cuda.get_device_name(0)}")

QWEN3VL_VARIANT = "Qwen/Qwen3-VL-2B-Instruct"
model = Qwen3VLForConditionalGeneration.from_pretrained(
    QWEN3VL_VARIANT,
    attn_implementation="flash_attention_2",
    dtype=torch.bfloat16,
    device_map={"": DEVICE},  # Load entire model to GPU 7
)

if isinstance(model_cfg.action_tokenizer, FastActionTokenizerConfig):
    model.resize_token_embeddings(
        len(tokenizer) + model_cfg.action_tokenizer.bins, 
        pad_to_multiple_of = 192,
        mean_resizing = True
    )
    print(f"Resized model token embeddings to {model.lm_head.weight.shape[0]}")

from safetensors.torch import load_file
state_dict = load_file(run_dir / "checkpoints" / f"ckpt_{ckpt_step}" / "model.safetensors")
state_dict["lm_head.weight"] = state_dict["model.language_model.embed_tokens.weight"] # HACK for '_tied_weights_keys = ["lm_head.weight"]'
# state_dict["model.language_model.embed_tokens.weight"] = state_dict["lm_head.weight"]
model.load_state_dict(state_dict)
print("loaded model checkpoint successfully.")

vlm_processor = AutoProcessor.from_pretrained(model_cfg.model_name_or_path)

# save both model and processor
dst_dir = run_dir / "pretrained" / f"ckpt_{ckpt_step}"
model.save_pretrained(dst_dir)
vlm_processor.save_pretrained(dst_dir)

print(f"saved model checkpoint successfully to {dst_dir}.")