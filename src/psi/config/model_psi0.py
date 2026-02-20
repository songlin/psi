from tyro.conf import subcommand as cmd
from typing import Union, Annotated, Optional, List
from psi.config.config import ModelConfig
from pydantic import Field

class Psi0ModelConfig(ModelConfig):
    ######################### hfm_action ############################
    # pretrained_model_name_or_path: str = "stabilityai/stable-diffusion-3-medium-diffusers"
    resnet_store_path: str | None = None # = "cache/checkpoints/resnet18/IN_1M_resnet18.pth"
    pretrained_action_header_path: str | None = None
    # precomputed_text_encodings: str = "cache/precomputed_text_encodings.pkl"
    # ckpt_step: Optional[int] = None

    rtc: bool = False
    max_delay: int = 8

    action_dim: int = 7 #36 # Da
    action_chunk_size: int = 6 #30 # Tp
    action_exec_horizon: int = 6 #30 # Ta
    observation_horizon: int = 1 # To past frames

    img_chunk: int = 1 # ?
    n_cams: int = 1 
    use_obs: str = "add_token"
    dropout: float = 0.1
    noise_scheduler: str = "flow"
    train_diffusion_steps: int = 1000
    eval_diffusion_steps: int = 10
    share_cam_features: bool = False
    early_fusion: bool = False

    # observations
    odim: int = 15 # 32

    # conditions
    n_conditions: Optional[int] = 1
    token_fusion: Optional[str] = "concat"  # "concat", "cross", "perceiver"

    # tune_vision_backbone: bool = False
    # vision_backbone_lr: float = 1e-5

    """
        training loss weight for xyz, rpy, and gripper
        should all sumed to 1 in total, eg 0.1x3+0.2x3+0.1x1=1
    """
    loss_w: List[float] = Field(default_factory=lambda: [0.1, 0.2, 0.1])

    # noise nets
    time_dim: int = 256
    hidden_dim: int = 1536
    num_blocks: int = 6
    dim_feedforward: int = 2048
    nhead: int = 24
    activation: str = "gelu"

    view_feature_dim: int = 1920 # views feature dim for views and/ or vlm token dim
    use_film: bool = False
    combined_temb: bool = False

    use_dit: bool = False

    ################### qwen3vl ####################
    # Training Schedule
    weight_decay: float = 0.01 # L2 regularization strength

    # Core Arguments
    model_name_or_path: str = "Qwen/Qwen3-VL-2B-Instruct"  # First load, for initalization
    # vlm_run_dir: str | None = None
    vlm_ckpt_step: str | None = None
    tune_vlm: bool = False


    tune_mm_llm: bool = False                   # [TrainingArguments] Train LLM or not
    tune_mm_vision: bool = False                # [TrainingArguments] Train VIT or not
    tune_mm_mlp: bool = False                   # [TrainingArguments] Train MLP or not
    # dataset_use: str = "egodex"               # [DataArguments] Dataset specification
    # output_dir: str = "./outputs/hfm_qwen3vl" # Output directory for checkpoints
    # cache_dir: str = "./cache/models"         # [TrainingArguments] Model cache location
    gradient_checkpointing: bool = True  # [TrainingArguments] Enable gradient checkpointing

    # Learning Rate Configuration
    lang_backbone_lr: float = 1e-5 
    mm_projector_lr: float = 1e-5   # [TrainingArguments] Projector-specific LR
    vision_tower_lr: float = 1e-6   # [TrainingArguments] Vision encoder LR
    optim: str = "adamw_torch"      # [TrainingArguments] Optimizer selection

    # Sequence Configuration
    model_max_length: int = 4096 # [TrainingArguments] Max sequence length
    data_flatten: bool = True    # [DataArguments] Concatenate batch sequences
    data_packing: bool = True    # [DataArguments] Using packing data

    # Image Processing
    max_pixels: int = 576 * 28 * 28  # [DataArguments] Max image pixels (H*W) for image
    min_pixels: int = 16 * 28 * 28   # [DataArguments] Min image pixels for image


    # lora_r: int = 8                          # [TrainingArguments] LoRA r
    # lora_alpha: int= 16                      # [TrainingArguments] LoRA alpha 
    # lora_dropout: float = 0.0                 # [TrainingArguments] LoRA dropout
