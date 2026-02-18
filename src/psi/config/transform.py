from __future__ import annotations
import re
import base64
import copy
import json
import math
import time
from io import BytesIO
from typing import (TYPE_CHECKING, Annotated, Any, Dict, List, Optional, Tuple,
                    Union)

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from pydantic import BaseModel, Field, model_validator
from torchvision.transforms import v2
from psi.config.augmentation import ResizeImage, ColorJitter, CenterCrop, Normalize
from psi.utils import get_asset_dir, pt_to_pil, resolve_path

IGNORE_INDEX = -100

def pad_to_len(x, target_len, dim=1, pad_value=0.0):
    """Pads a tensor to the target length along the specified dimension.
    Args:
        x: np.ndarray to pad
        target_len: int, target length to pad to
        dim: int, dimension along which to pad
        pad_value: value to use for padding
    Returns:
        padded: np.ndarray, padded array
        mask: np.ndarray of bool, True for original data, False for padded region
    """
    current_len = x.shape[dim]
    if current_len >= target_len:
        mask = np.ones(x.shape, dtype=bool)
        return x, mask
    pad_width = [(0, 0)] * x.ndim
    pad_width[dim] = (0, target_len - current_len)
    # np.pad pads as (before, after) for each axis
    padded = np.pad(x, pad_width, mode='constant', constant_values=pad_value)
    mask_shape = list(x.shape)
    mask_shape[dim] = target_len
    mask = np.ones(mask_shape, dtype=bool)
    # Mark padded region as False (0)
    idx = [slice(None)] * x.ndim
    idx[dim] = slice(current_len, target_len)
    mask[tuple(idx)] = False
    return padded, mask

class RepackTransform(BaseModel):
    def __call__(self, data: dict[str, Any], **kwargs) -> dict[str, Any]:
        return data

class ModelTransform(BaseModel):
    def __call__(self, data: dict[str, Any], no_aug: bool = False, **kwargs) -> dict[str, Any]:
        return data

class EgodexRepackTransform(BaseModel):
    dataset_name: str = "egodex"
    stage:str = "pretrain"  # "pretrain" or "postpre"
    pad_action_dim: int | None = None
    pad_state_dim: int | None = None

    def __call__(self, data: dict[str, Any], **kwargs) -> dict[str, Any]:
        states, _ = pad_to_len(data["states"], self.pad_state_dim) if self.pad_state_dim is not None else (data["states"], None)
        if self.pad_action_dim is not None:
            actions, mask = pad_to_len(data["actions"], self.pad_action_dim) 
        else:
            actions = data["actions"]
            mask = np.ones_like(actions, dtype=bool)

        return dict(
            observations = [Image.fromarray(img[0]) for img in data["current_images"]],  # list of PIL Image
            states=states,  # (To, Da)
            actions=actions,  # (Tp, Da)
            instruction=data["instruction"].lower(),
            dataset=data.get("dataset_name", self.dataset_name),
            actions_mask=mask, #(Tp, Da)
        )


class SimpleRepackTransform(RepackTransform):
    dataset_name: str = "simple"

    num_past_frames: int = 0 # single current frame
    action_chunk_size: int = 16

    pad_action_dim: int | None = None
    pad_state_dim: int | None = None

    def delta_timestamps(self, fps: int):
        return {
            "action": [t/fps for t in range(self.action_chunk_size)],
            "observation.amo_policy_command": [t/fps for t in range(-(1+self.num_past_frames), self.action_chunk_size)],
            "observation.proprio_joint_positions": [-t/fps for t in range(self.num_past_frames, -1, -1)]
        }
    
    @staticmethod
    def to_psi0_state_format(
        proprio_joint_positions: torch.Tensor,
        amo_policy_command: torch.Tensor,
    ):
        return torch.cat([
            proprio_joint_positions[:, 29:32], # left hand: thumb 012
            proprio_joint_positions[:, 34:36], # left hand: middle 01
            proprio_joint_positions[:, 32:34], # left hand: index 01
            proprio_joint_positions[:, 36:43], # right hand
            proprio_joint_positions[:, 15:22], # left arm
            proprio_joint_positions[:, 22:29], # right arm
            proprio_joint_positions[:, 13:15], # torso rpy -->  waist rpy
            proprio_joint_positions[:, 12:13],
            amo_policy_command[:, 6:7] # last commanded base height
        ], dim=1).to(torch.float32)

    @staticmethod
    def to_psi0_action_format(data: dict):
        proprio = data["observation.proprio_joint_positions"]
        history_cmd = data["observation.amo_policy_command"]
        action = data["action"]
        To = proprio.shape[0]
        Ta = action.shape[0]
        return torch.cat([
            action[:, 29:32], # left hand: thumb 012 [0:3]
            action[:, 34:36], # left hand: middle 01 [3:5]
            action[:, 32:34], # left hand: index 01  [5:7]
            action[:, 36:43], # right hand           [7:14]   
            action[:, 15:22], # left arm             [14:21]
            action[:, 22:29], # right arm            [21:28]
            action[:, 13:15], # waist rpy            [28:30]
            action[:, 12:13], #                      [30:31]
            history_cmd[To:, 6:7], # base height     [31:32]
            history_cmd[To:, 0:2], # vx,vy,          [32:34]
            torch.zeros((Ta, 1)), # vyaw, not used,  [34:35]
            history_cmd[To:, 3:4], # target yaw      [35:36]
        ], dim=1).to(torch.float32)

    def __call__(self, data: dict[str, Any], **kwargs) -> dict[str, Any]:
        image_key = "observation.rgb_head_stereo_left"
        states = self.to_psi0_state_format(
            data["observation.proprio_joint_positions"], # TODO TBD 
            data["observation.amo_policy_command"][:1+self.num_past_frames]
        ).numpy() # (To, Ds)
        actions = self.to_psi0_action_format(data).numpy() # (To, Ds)
        action_is_pad = data["action_is_pad"].numpy() # (Ta,) 
        pad_mask = np.ones_like(actions) * (1. - action_is_pad[...,None].astype(np.float32))

        states, _ = pad_to_len(states, self.pad_state_dim) if self.pad_state_dim is not None else (states, None)
        if self.pad_action_dim is not None:
            actions, _ = pad_to_len(actions, self.pad_action_dim) 
            mask,_ = pad_to_len(pad_mask, self.pad_action_dim)
        else:
            mask = np.ones_like(actions, dtype=np.float32)

        return {
            "observations": [pt_to_pil(data[image_key],normalized=False)], # single view
            "states": states.astype(np.float32), # (To, Do)
            "actions": actions.astype(np.float32), # (Tp, Da)
            # "action_is_pad": np.array(data["action_is_pad"], dtype=bool),
            "instruction": data["task"].lower(),
            "actions_mask": mask
        }


class IdentityTransform(BaseModel):
    # boilerplate transform config
    # define primitive data types only! eg., int, float, str
    # container types are allowed eg., list[int], dict[str, float]
    # default no-op transform
    # passing external dependency through **kwargs, eg., tokenizer, processor
    def __call__(self, data: dict[str, Any], **kwargs) -> dict[str, Any]:
        return data

    def denormalize_L1_action_err(self, L1_err):
        """return denormalized L1 err loss"""
        return L1_err
    
    def denormalize(self, normalized: dict) -> dict:
        """ Denormalize the action. """
        return normalized

class ActionMaxMinTransform(BaseModel):
    
    stat_path: str
    action_norm_type: str = "bounds"  # "bounds_q99"
    stat_action_key: str = "action"
    stat_state_key: str = "states"
    use_norm_mask: bool = False
    action_norm_masks: List[bool] = [
        True,
        True,
        True,
        True,
        True,
        True,
        False,
    ]  # delta-eef + gripper
    action_min: Optional[List[float]] = None
    action_max: Optional[List[float]] = None
    state_min: Optional[List[float]] = None
    state_max: Optional[List[float]] = None
    normalize_state: bool = False  # whether to normalize states

    pad_action_dim: int | None = None
    pad_state_dim: int | None = None

    def model_post_init(self, __context: Any) -> None:
        with open(resolve_path(self.stat_path), "r") as f:
            stat = json.load(f)
            
        if self.action_norm_type == "bounds_q99":
            self.action_min = stat[self.stat_action_key]["q01"]
            self.action_max = stat[self.stat_action_key]["q99"]
            if self.normalize_state:
                self.state_min = stat[self.stat_state_key]["q01"]
                self.state_max = stat[self.stat_state_key]["q99"]
        elif self.action_norm_type == "bounds":
            self.action_min = stat[self.stat_action_key]["min"]
            self.action_max = stat[self.stat_action_key]["max"]
            if self.normalize_state:
                self.state_min = stat[self.stat_state_key]["min"]
                self.state_max = stat[self.stat_state_key]["max"]
        else:
            raise ValueError(f"Unsupported action normalization type: {self.action_norm_type}")

        if self.pad_action_dim is not None:
            self.action_min= pad_to_len(np.array(self.action_min, dtype=np.float32), self.pad_action_dim, dim=0)[0].tolist()
            self.action_max = pad_to_len(np.array(self.action_max, dtype=np.float32), self.pad_action_dim, dim=0)[0].tolist()

        if self.pad_state_dim is not None and self.normalize_state:
            self.state_min = pad_to_len(np.array(self.state_min, dtype=np.float32), self.pad_state_dim, dim=0)[0].tolist()
            self.state_max = pad_to_len(np.array(self.state_max, dtype=np.float32), self.pad_state_dim, dim=0)[0].tolist()

    def __call__(self, data: dict[str, Any], **kwargs) -> dict[str, Any]:
        assert self.action_min is not None and self.action_max is not None
        action_min = np.array(self.action_min, dtype=np.float32)
        action_max = np.array(self.action_max, dtype=np.float32)
        if self.normalize_state:
            data["states"] = self.normalize_state_func(data["states"])

        ill_mask = (action_max - action_min) == 0
        action_max[ill_mask] = 1.0  # prevent division by zero
        actions_normalized = np.where(
            ill_mask, data["actions"], (data["actions"] - action_min) / (action_max - action_min) * 2 - 1
        )

        if self.use_norm_mask:
            actions = np.where(self.action_norm_masks, actions_normalized, data["actions"])
        else: 
            actions = actions_normalized

        actions = np.clip(actions, -1, 1).astype(np.float32)
        # print(data["dataset"], actions.max(), actions.min(), actions.mean(), actions.std())
        data["raw_actions"] = data["actions"]
        data["actions"] = actions

        return data

    def normalize(self, action, **kwargs):
        data = {"actions": action}
        return self.__call__(data)["actions"]

    def normalize_state_func(self, states, **kwargs):
        state_min = np.array(self.state_min, dtype=np.float32)
        state_max = np.array(self.state_max, dtype=np.float32)
        # Normalize states
        ill_mask = (state_max - state_min) == 0
        state_max[ill_mask] = 1.0  # prevent division by zero
        current_state = np.where(
            ill_mask, 0, 
            (states - state_min) / (state_max - state_min) * 2 - 1
        )
        current_state = np.clip(current_state, -1, 1).astype(np.float32)
        if np.isnan(current_state).any():
            current_state = np.nan_to_num(current_state, nan=0.0)
        return current_state

    def reverse_call(self, array: Any, **kwargs) -> Any:
        assert self.action_min is not None and self.action_max is not None
        action_min = np.array(self.action_min)
        action_max = np.array(self.action_max)
        reversed_array = (array + 1) / 2 * (action_max - action_min) + action_min
        return reversed_array

    def denormalize_L1_action_err(self, L1_err):
        """return denormalized L1 err loss"""
        
        array_class = torch.tensor if torch.is_tensor(L1_err) else np.array
        where = torch.where if torch.is_tensor(L1_err) else np.where
        data_type = L1_err.dtype

        if self.action_norm_type == "bounds" or \
            self.action_norm_type == "bounds_q99":
            low = array_class(self.action_min, dtype=data_type)  # type: ignore
            high = array_class(self.action_max, dtype=data_type)  # type: ignore
            
            if self.use_norm_mask:
                if torch.is_tensor(L1_err):
                    low = low.to(L1_err.device)
                    high = high.to(L1_err.device)
                    action_norm_masks = torch.tensor(
                        self.action_norm_masks, device=L1_err.device
                    )
                else:
                    action_norm_masks = self.action_norm_masks
                result = where(action_norm_masks, 0.5 * L1_err * (high - low), L1_err) # type:ignore
            else:
                result = 0.5 * L1_err * (high - low)
            return result
        else:
            raise NotImplementedError
    
    
    def denormalize(self, normalized: np.ndarray|torch.Tensor) -> np.ndarray|torch.Tensor: 
        """ Denormalize the action. """
        array_class = torch.tensor if torch.is_tensor(normalized) else np.array
        where = torch.where if torch.is_tensor(normalized) else np.where
        data_type = normalized.dtype

        if self.action_norm_type == "bounds" or \
                self.action_norm_type == "bounds_q99":
            if self.action_norm_type == "bounds":
                low = array_class(self.action_min, dtype=data_type)  # type: ignore
                high = array_class(self.action_max, dtype=data_type)  # type: ignore
            elif self.action_norm_type == "bounds_q99":
                low = array_class(self.action_min, dtype=data_type)  # type: ignore
                high = array_class(self.action_max, dtype=data_type)  # type: ignore

            if not self.use_norm_mask:
                assert self.action_min is not None
                action_norm_masks = [True] * len(self.action_min)
            else:
                action_norm_masks = self.action_norm_masks

            if torch.is_tensor(normalized):
                low = low.to(normalized.device)  # type: ignore
                high = high.to(normalized.device)  # type: ignore
                action_norm_masks = torch.tensor(action_norm_masks, device=normalized.device)
            else:
                action_norm_masks = np.array(action_norm_masks)

            action = where(action_norm_masks, 0.5 * (normalized + 1) * (high - low) + low, normalized)  # type: ignore
        return action # type: ignore


class Hfm_ModelTransform(ModelTransform):
    resize: ResizeImage = Field(default_factory=lambda: ResizeImage(size=(270, 480)))
    color_jitter: ColorJitter
    img_aug: bool = False
    # anchor_size: tuple[int, int] = (256, 256) # H, W

    # for mixed dataset with different image sizes
    adaptive_resize: bool = False
    img_sizes: dict[str, Any] = Field(default_factory=lambda: {
        "egodex": [270, 480],
        "we"    : [240, 320],
    })

    def __call__(
        self, 
        data: dict[str, Any], 
        no_aug: bool = False,
        vlm_processor = None, 
        action_tokenizer = None, 
        **kwargs
    ) -> dict[str, Any]:
        if vlm_processor is None or action_tokenizer is None:
            return {"raw_actions": data["actions"], **data}
        """
        data: dict with keys:
            - instruction: str
            - observations: List[Image] (To x PIL)
            - states: (To, Da) torch.Tensor
            - action: (Tp, Da) torch.Tensor
        """
        do_img_aug = False if no_aug else self.img_aug
        if self.adaptive_resize:
            assert data["dataset"] is not None
            match data["dataset"]:
                case "egodex":
                    target_size = self.img_sizes["egodex"]
                case "humanoid-everyday":
                    target_size = self.img_sizes["we"]
                case _:
                    target_size = (256, 256)
            resizer = ResizeImage(size=tuple(target_size))() # type: ignore
        else:
            resizer = self.resize()
        t1 = v2.Compose([
            resizer,
            self.color_jitter() if do_img_aug else v2.Identity(),
        ])
        
        images = [t1(img) for img in data["observations"]]
        instruction = data["instruction"]
        state = data["states"]
        action = data["actions"]
        inputs, num_answer_tokens_list = self.build_qwenvl_inputs(
            vlm_processor, action_tokenizer, [images], [instruction], [state], [action]
        )
        labels = copy.deepcopy(inputs["input_ids"])
        # keep loss on the answer + EOS + formatting tokens
        labels[:, : -(num_answer_tokens_list[0] + 2)] = IGNORE_INDEX 
        inputs["labels"] = labels
        inputs["dataset_name"] = data.get("dataset", "unknown")
        inputs["raw_actions"] = action
        # inputs["raw_instruction"] = instruction
        inputs["raw_images"] = np.stack([np.array(img) for img in images])
        return inputs

    def build_qwenvl_inputs(
        self,
        vlm_processor,
        action_tokenizer,
        images,
        instructions,
        states,
        actions,
        **kwargs,
    ):
        """adapted from Qwen_VL_Interface.build_qwenvl_inputs"""
        # Create messages: one message per sample
        messages = []
        num_answer_tokens_list = []
        assert len(images) == len(
            instructions
        ), "Images and instructions must have the same length"

        for imgs, instruction, action in zip(images, instructions, actions):
            tokenized_action = action_tokenizer(action)
            raw_action_tokens = vlm_processor.tokenizer(tokenized_action)["input_ids"]
            num_answer_tokens = len(raw_action_tokens)
            num_answer_tokens_list.append(num_answer_tokens)

            # print(action[0].shape)
            content = [{"type": "image", "image": img} for img in imgs]
            content.append({"type": "text", "text": instruction})
            user_msg = {"role": "user", "content": content}
            assistant_msg = {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": tokenized_action}
                ],  # squeeze batch dim
            }
            messages.append([user_msg, assistant_msg])

        # Prepare text prompts using processor
        # default process is json --> message --> texts --> input_ids
        texts = [
            vlm_processor.apply_chat_template(
                m, tokenize=False, add_generation_prompt=False
            )
            for m in messages
        ]

        try:
            from qwen_vl_utils import process_vision_info
        except:
            raise ImportError("qwen_vl_utils not found, make sure to install it if using Qwen-VL model!")
        image_inputs, video_inputs = process_vision_info(messages, image_patch_size=16)  # type: ignore
        inputs = vlm_processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        return inputs, num_answer_tokens_list


class DataTransform(BaseModel):
    def __call__(self, data: dict[str, Any], **kwargs) -> dict[str, Any]:
        return data
