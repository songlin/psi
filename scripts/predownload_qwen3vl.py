import os
dotenv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
assert os.path.exists(dotenv_path), f".env file not found at {dotenv_path}"
from dotenv import load_dotenv
load_dotenv(dotenv_path)
print("HF_HOME:", os.getenv("HF_HOME"))
print("TORCH_HOME:", os.getenv("TORCH_HOME"))

import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

model = Qwen3VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen3-VL-2B-Instruct",
    attn_implementation="flash_attention_2",
    dtype=torch.bfloat16,
    device_map={"": "cuda:0"},
    # local_files_only=True
)
print("Model loaded.")

vlm_processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-2B-Instruct")
print("Processor loaded.")
