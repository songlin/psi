import dotenv
dotenv.load_dotenv()

from pathlib import Path
# from psi.utils import inspect

ckpt_step = 8800
run_dir = Path(".runs/hfm-post-pre/postpre.abl.10per.mixed.flow1000.cosin.lr1.0e-04.b2048.gpus32.2602050006")

print("Loading checkpoint...")
from safetensors.torch import load_file
state_dict = load_file(run_dir / "checkpoints" / f"ckpt_{ckpt_step}" / "model.safetensors", device="cpu")
print("Checkpoint loaded.")

""" # test
print("Loading pretrained qwen3vl...") 
qwen3vl_state_dict = load_file(".../workspace/project131/.cache/checkpoints/hfm.pre.fast.egodex.2512241941.ckpt200k/model.safetensors")
print("Pretrained model loaded.")

for k, v in qwen3vl_state_dict.items():
    # if k.startswith("vlm_model."):
        # new_k = k.replace("model.vision_encoder.", "vision_encoder.")
        # state_dict[new_k] = v
        print(k, inspect(v))
        break """

action_header_only = {}
for k, v in state_dict.items():
    if k.startswith("vlm_model."):
        # new_k = k.replace("model.vision_encoder.", "vision_encoder.")
        # state_dict[new_k] = v
        continue
        # print(k, inspect(v))
        # break
    else:
        action_header_only[k.replace("action_header.", "")] = v

# Save the filtered weights as model.safetensors
from safetensors.torch import save_file
save_dir = run_dir / "pretrained" / f"ckpt_{ckpt_step}"
save_dir.mkdir(parents=True, exist_ok=True)
save_path = save_dir / "action_header.safetensors"
save_file(action_header_only, str(save_path))
print(f"Saved action header weights to {save_path}")
