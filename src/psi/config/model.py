from __future__ import annotations
from pydantic import BaseModel, Field, model_validator
from typing import Optional, Union, Annotated, TYPE_CHECKING
import tyro

from psi.utils.utils import get_cache_dir
# from .tokenizer import 
# BinActionTokenizerConfig, 
# VQVaeActionTokenizerConfig, 
# FastActionTokenizerConfig, 
# VQActionTokenizerConfig, 
# TextActionTokenizerConfig
import os
from typing import List, Any
from pathlib import Path

from .tokenizer import (
    FastActionTokenizerConfig, 
    VQActionTokenizerConfig, 
    TextActionTokenizerConfig
)

if TYPE_CHECKING:
    from .tokenizer import BinActionTokenizerConfig
    from .tokenizer import VQVaeActionTokenizerConfig
    # from .tokenizer import FastActionTokenizerConfig
    from .tokenizer import VQActionTokenizerConfig
    from .tokenizer import TextActionTokenizerConfig

class ModelConfig(BaseModel): ...


class DummyModelConfig(ModelConfig):
    # boilerplate model config
    # put model related configs here, eg., hidden dim
    input_dim: int = 7
    hidden_dim: int = 100
    output_dim: int = 7


class InternVLA_M1_ModelConfig(ModelConfig):
    pretrain_path: str = (
        "/hfm/cache/checkpoints/InternVLA-M1-Pretrain-RT-1-Bridge/checkpoints/steps_50000_pytorch_model.pt"  # '/hfm/boqian/cache/checkpoints/InternVLA-M1-Pretrain-RT-1-Bridge/checkpoints/steps_50000_pytorch_model.pt'#
    )
    vlm_backbone: str = "Qwen/Qwen2.5-VL-3B-Instruct"
    action_tokenizer: Union[
        Annotated[BinActionTokenizerConfig, tyro.conf.subcommand("bin")],
        Annotated[VQVaeActionTokenizerConfig, tyro.conf.subcommand("vqvae")],
    ]


class Qwen25VL_ModelConfig(ModelConfig):
    test2: str = "abc"


class OpenVLA_ModelConfig(ModelConfig):
    pretrained_checkpoint: str | None

    vla_id: str = "prism-dinosiglip-224px+mx-grasp"
    base_vlm: str = "prism-dinosiglip-224px+7b"

    freeze_vision_backbone: bool = True
    freeze_llm_backbone: bool = False
    unfreeze_last_llm_layer: bool = False

    is_resume: bool = False  # Whether we are continuing a prior training run
    #   (only applicable given pretrained checkpoint)
    resume_step: Optional[int] = None  # Global Step to Resume (should match checkpoint)
    resume_epoch: Optional[int] = None  # Epoch to Resume (should match checkpoint)

    trackers: list[str] = ["jsonl", "wandb"]

    run_id_note: Optional[str] = None

    action_tokenizer: BinActionTokenizerConfig  
    
class OpenVLA_Qwenvl_ModelConfig(ModelConfig):
    pretrained_checkpoint: str = "Qwen/Qwen2.5-VL-3B-Instruct"      # Prefer local path for faster loading
    vla_id: str = "qwen25vl3b-224px+mx-grasp"
    base_vlm: str = "qwen25vl3b-224px"

    freeze_vision_backbone: bool = True
    freeze_llm_backbone: bool = False
    unfreeze_last_llm_layer: bool = False

    is_resume: bool = False                                         # Whether we are continuing a prior training run
                                                                    #   (only applicable given pretrained checkpoint)
    resume_step: Optional[int] = None                               # Global Step to Resume (should match checkpoint)
    resume_epoch: Optional[int] = None                              # Epoch to Resume (should match checkpoint)

    trackers: list[str] = ["jsonl", "wandb"]

    run_id_note: Optional[str] = None
    
    action_tokenizer: BinActionTokenizerConfig                        # Extra note for logging, Weights & Biases   action_tokenizer: BinActionTokenizerConfig                        # Extra note for logging, Weights & Biases

    action_tokenizer: (
        BinActionTokenizerConfig  # Extra note for logging, Weights & Biases
    )


class OpenVLA_Flow_ModelConfig(ModelConfig):
    pretrained_checkpoint: str | None

    ##### Flow Action Head #####
    obs_context_horizon: int = 1  # To
    action_pred_horizon: int = 6  # Tp
    action_dim: int = 7  # Da
    noise_beta_alpha: float = 1.5
    noise_beta_beta: float = 1.0
    noise_s: float = 0.999
    num_timestep_buckets: int = 1000
    num_inference_timesteps: int = 10
    num_transformer_layers: int = 6

    action_lr_scheduler_config: dict = Field(
        default_factory=lambda: {
            # "lr_scheduler": "coside",
            "learning_rate": 1e-4,
            # "betas": (0.95, 0.999),
            # "weight_decay": 1e-6,
            # "num_warmup_steps": 1000,
        }
    )

    hidden_dim: int = 1536
    # input_embedding_dim: int = 1536
    max_state_dim: int = 64
    max_seq_len: int = 1024

    # vlm feature config
    vlm_config: dict = Field(
        default_factory=lambda: {
            "attention_head_dim": 64,
            "dropout": 0.2,
            "final_dropout": True,
            "num_attention_heads": 32,
            "num_layers": 4,
            "positional_embeddings": None,
        }
    )

    # diffsuion head config
    attention_head_dim: int = 48
    backbone_embedding_dim: int = 4096
    cross_attention_dim: int = 4096
    dropout: float = 0.2
    final_dropout: bool = True
    interleaved_self_attention: bool = True
    norm_type: str = "ada_norm"
    num_attention_heads: int = 32
    num_layers: int = 16
    output_dim: int = 1024  # TODO ?
    positional_embeddings: str | None = None

    """
        training loss weight for xyz, rpy, and gripper
        should all sumed to 1 in total, eg 0.1x3+0.2x3+0.1x1=1
    """
    loss_w: List[float] = [0.1, 0.2, 0.1]  # [xyz, rpy, gripper]

    ##### duplicate from OpenVLA_ModelConfig #####
    vla_id: str = "prism-dinosiglip-224px+mx-grasp"
    base_vlm: str = "prism-dinosiglip-224px+7b"
    freeze_vision_backbone: bool = True
    freeze_llm_backbone: bool = False
    unfreeze_last_llm_layer: bool = False
    is_resume: bool = False  # Whether we are continuing a prior training run
    #   (only applicable given pretrained checkpoint)
    resume_step: Optional[int] = None  # Global Step to Resume (should match checkpoint)
    resume_epoch: Optional[int] = None  # Epoch to Resume (should match checkpoint)
    trackers: list[str] = ["jsonl", "wandb"]
    # Extra note for logging, Weights & Biases
    run_id_note: Optional[str] = None

    # action_tokenizer: (
    #     BinActionTokenizerConfig
    # )

class DitPolicy_ModelConfig(ModelConfig):
    # vision backbone config
    pretrained_path: str = "cache/checkpoints/resnet18/IN_1M_resnet18.pth"

    img_chunk: int = 1 # ?

    odim: int = 15
    camera_indices: list[int] = Field(default_factory=lambda: [0])
    conditions: list[str] = Field(default_factory=lambda: [])

    num_past_frames: int = 0 # To
    n_cams: int = 1 
    # n_conditions: int = 1
    use_obs: str = "add_token"
    dropout: float = 0.1
    train_diffusion_steps: int = 100
    eval_diffusion_steps: int = 8
    ac_dim: int = 7
    ac_chunk: int = 12
    ac_exec_horizon: int = 12
    # imgs_per_cam: 
    share_cam_features: bool = False
    early_fusion: bool = False

    # noise scheduler
    noise_scheduler: str = "ddpm" # "flow"
    """
        training loss weight for xyz, rpy, and gripper
        should all sumed to 1 in total, eg 0.1x3+0.2x3+0.1x1=1
    """
    loss_w: List[float] = Field(default_factory=lambda: [0.1, 0.2, 0.1])

    # noise nets
    time_dim: int = 256
    hidden_dim: int = 512
    num_blocks: int = 6
    dim_feedforward: int = 2048
    nhead: int = 8
    activation: str = "gelu"

    def model_post_init(self, __context: Any) -> None:
        if not os.path.isabs(self.pretrained_path):
            self.pretrained_path = str(get_cache_dir() / self.pretrained_path)

class Vlt_ModelConfig(ModelConfig):
    pretrained_model_name_or_path: str = "stabilityai/stable-diffusion-3-medium-diffusers"
    resnet_store_path: str = "cache/checkpoints/resnet18/IN_1M_resnet18.pth"
    precomputed_text_encodings: str = "cache/precomputed_text_encodings.pkl"
    ckpt_step: Optional[int] = None

    action_dim: int = 7 # Da
    action_chunk_size: int = 6 # Tp
    action_exec_horizon: int = 6 # Ta
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
    odim: int = 15

    # conditions
    n_conditions: Optional[int] = 1
    token_fusion: Optional[str] = "concat"  # "concat", "cross", "perceiver"

    tune_vision_backbone: bool = False
    vision_backbone_lr: float = 1e-5

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


class Hfm_Action_ModelConfig(ModelConfig):
    pretrained_model_name_or_path: str = "stabilityai/stable-diffusion-3-medium-diffusers"
    resnet_store_path: str = "cache/checkpoints/resnet18/IN_1M_resnet18.pth"
    precomputed_text_encodings: str = "cache/precomputed_text_encodings.pkl"
    ckpt_step: Optional[int] = None

    action_dim: int = 36 # 7 # Da
    action_chunk_size: int = 30 # 6 # Tp
    action_exec_horizon: int = 30 #6 # Ta
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
    odim: int = 32 #15

    # conditions
    n_conditions: Optional[int] = 1
    token_fusion: Optional[str] = "concat"  # "concat", "cross", "perceiver"

    tune_vision_backbone: bool = False
    vision_backbone_lr: float = 1e-5

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
    use_film: bool = True
    combined_temb: bool = True


class Together_ModelConfig(ModelConfig):
    ######################### hfm_action ############################
    # pretrained_model_name_or_path: str = "stabilityai/stable-diffusion-3-medium-diffusers"
    resnet_store_path: str = "cache/checkpoints/resnet18/IN_1M_resnet18.pth"
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
    vlm_run_dir: str | None = None
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
    
    # action_tokenizer: BinActionTokenizerConfig


class DiffusionPolicy_ModelConfig(ModelConfig):
    num_diffusion_iters: int = 100
    pred_horizon: int = 16

class GR00T_ModelConfig(ModelConfig):
    pretrained_model_name_or_path: str = "nvidia/GR00T-N1.5-3B"
    tune_visual: bool = False
    tune_llm: bool = False
    tune_projector: bool = True
    tune_diffusion_model: bool = True
    pretrained_checkpoint: Optional[str] = None
    is_resume: bool = False
    resume_step: Optional[int] = None
    resume_epoch: Optional[int] = None
    run_id_note: Optional[str] = None

class Qwen3vl_ModelConfig(ModelConfig):
    # Training Schedule
    weight_decay: float = 0.01 # L2 regularization strength

    # Core Arguments
    model_name_or_path: str = "Qwen/Qwen3-VL-2B-Instruct"  # [ModelArguments] Model identifier
    tune_mm_llm: bool = True                    # [TrainingArguments] Train LLM or not
    tune_mm_vision: bool = False                # [TrainingArguments] Train VIT or not
    tune_mm_mlp: bool = True                   # [TrainingArguments] Train MLP or not
    # dataset_use: str = "egodex"               # [DataArguments] Dataset specification
    # output_dir: str = "./outputs/hfm_qwen3vl" # Output directory for checkpoints
    # cache_dir: str = "./cache/models"         # [TrainingArguments] Model cache location
    gradient_checkpointing: bool = True  # [TrainingArguments] Enable gradient checkpointing

    # Learning Rate Configuration
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


    lora_r: int = 8                          # [TrainingArguments] LoRA r
    lora_alpha: int= 16                      # [TrainingArguments] LoRA alpha 
    lora_dropout: float = 0.0                 # [TrainingArguments] LoRA dropout
    
    action_tokenizer: FastActionTokenizerConfig
    deepspeed: str = "scripts/deepspeed/zero3.json"  # DeepSpeed configuration name

class Qwen3vl_7d_ModelConfig(ModelConfig):
    # Training Schedule
    weight_decay: float = 0.01 # L2 regularization strength

    # Core Arguments
    model_name_or_path: str = "Qwen/Qwen3-VL-2B-Instruct"  # [ModelArguments] Model identifier
    tune_mm_llm: bool = True                    # [TrainingArguments] Train LLM or not
    tune_mm_vision: bool = False                # [TrainingArguments] Train VIT or not
    tune_mm_mlp: bool = True                   # [TrainingArguments] Train MLP or not
    # dataset_use: str = "egodex"               # [DataArguments] Dataset specification
    # output_dir: str = "./outputs/hfm_qwen3vl" # Output directory for checkpoints
    # cache_dir: str = "./cache/models"         # [TrainingArguments] Model cache location
    gradient_checkpointing: bool = True  # [TrainingArguments] Enable gradient checkpointing

    # Learning Rate Configuration
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


    lora_r: int = 8                          # [TrainingArguments] LoRA r
    lora_alpha: int= 16                      # [TrainingArguments] LoRA alpha 
    lora_dropout: float = 0.0                 # [TrainingArguments] LoRA dropout
    
    action_tokenizer: BinActionTokenizerConfig


class Hfm_Qwen3VL_ModelConfig(ModelConfig):
    # Training Schedule
    weight_decay: float = 0.01 # L2 regularization strength

    # Core Arguments
    model_name_or_path: str = "Qwen/Qwen3-VL-2B-Instruct"  # [ModelArguments] Model identifier
    tune_mm_llm: bool = True                    # [TrainingArguments] Train LLM or not
    tune_mm_vision: bool = False                # [TrainingArguments] Train VIT or not
    tune_mm_mlp: bool = True                   # [TrainingArguments] Train MLP or not
    # dataset_use: str = "egodex"               # [DataArguments] Dataset specification
    # output_dir: str = "./outputs/hfm_qwen3vl" # Output directory for checkpoints
    # cache_dir: str = "./cache/models"         # [TrainingArguments] Model cache location
    gradient_checkpointing: bool = True  # [TrainingArguments] Enable gradient checkpointing

    # Learning Rate Configuration
    mm_projector_lr: float = 1e-5   # [TrainingArguments] Projector-specific LR
    vision_tower_lr: float = 1e-6   # [TrainingArguments] Vision encoder LR
    optim: str = "adamw_torch"      # [TrainingArguments] Optimizer selection

    # Sequence Configuration
    model_max_length: int = 8192 # [TrainingArguments] Max sequence length

    lora_r: int = 8                          # [TrainingArguments] LoRA r
    lora_alpha: int= 16                      # [TrainingArguments] LoRA alpha 
    lora_dropout: float = 0.0                 # [TrainingArguments] LoRA dropout
    
    action_tokenizer: Union[
        Annotated[FastActionTokenizerConfig, tyro.conf.subcommand("fast")],
        Annotated[VQActionTokenizerConfig, tyro.conf.subcommand("vq")],
        Annotated[TextActionTokenizerConfig, tyro.conf.subcommand("text")],
    ]
    # action_tokenizer: FastActionTokenizerConfig
    deepspeed: str = "scripts/deepspeed/zero3.json"  # DeepSpeed configuration name
