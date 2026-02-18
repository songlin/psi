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
# Lazy imports - only import heavy dependencies when actually needed
# if TYPE_CHECKING:
from PIL import Image
# import requests
# import tyro
from pydantic import BaseModel, Field, model_validator
from torchvision.transforms import v2
from typing_extensions import Self
from tyro.conf import subcommand as cmd

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

class Egodex_Openvla_RepackTransform(BaseModel):
    dataset_name: str = "egodex"
    resize_resolution: Tuple[int, int] = (224, 224)
    
    def resize_image(self, image, size: Tuple[int, int]):
        """Resizes an image using Lanczos interpolation. Expects & returns uint8."""
        assert image.dtype == np.uint8, f"Expected uint8, got {image.dtype}"
        # Convert numpy array to PIL Image
        pil_image = Image.fromarray(image)
        # Resize using Lanczos (equivalent to Lanczos3)
        # PIL.Image.LANCZOS is deprecated in newer versions, use Image.Resampling.LANCZOS
        try:
            resized_pil = pil_image.resize((size[1], size[0]), Image.Resampling.LANCZOS)
        except AttributeError:
            # Fallback for older Pillow versions
            resized_pil = pil_image.resize((size[1], size[0]), Image.LANCZOS)  # type: ignore
        return resized_pil # type: ignore

    def __call__(self, data: dict[str, Any], resize_resolution: Tuple[int, int], **kwargs) -> dict[str, Any]:
        import numpy as np
        return dict(
            # observations=[
            #     Image.fromarray(data["observation"][key][0])
            #     for key in data["observation"].keys()
            #     if "image" in key
            # ],  # list of PIL Image
            observations=[
                self.resize_image(image[0], resize_resolution)
                for image in data["current_images"]
            ],  # list of PIL Image
            states=data["states"],  # (To, Da)
            actions=data["actions"],  # (Tp, Da)
            # action_mask=torch.tensor(data["absolute_action_mask"]),  # (Tp, Da)
            instruction=data["instruction"].lower(),
            # task="tabletop_grasp",
            dataset=self.dataset_name
            # frame_index=data["step_index"]
        )

class RLDSRepackTransform(RepackTransform):
    dataset_name: str = "rlds"
    image_key: str = "observation.image_camera_0"
    state_key: str = "observation.proprio"
    action_key: str = "action"
    instruction_key: str = "task.language_instruction"

    def __call__(self, data: dict[str, Any], **kwargs) -> dict[str, Any]:
        return dict(
            observations=[
                Image.fromarray(data["observation"][key][0])
                for key in data["observation"].keys()
                if "image" in key
            ],  # list of PIL Image
            states=np.array(data["observation"]["proprio"], dtype=np.float32),  # (To, Da)
            actions=np.array(data["action"], dtype=np.float32),  # (Tp, Da)
            # action_mask=torch.tensor(data["absolute_action_mask"]),  # (Tp, Da)
            instruction=data["task"]["language_instruction"].decode().lower(),
            task="tabletop_grasp",
            dataset=self.dataset_name
            # frame_index=data["step_index"]
        )

class HumanoidRepackTransform(RepackTransform):
    dataset_name: str = "humanoid"
    humanoid: bool = True

    def __call__(self, data: dict[str, Any], **kwargs) -> dict[str, Any]:
        return dict(
            observations=[
                Image.fromarray(data["observation"][key][0])
                for key in data["observation"].keys()
                if "image" in key
            ],  # list of PIL Image
            # states=np.array(data["observation"]["proprio"], dtype=np.float32),  # (To, Da)
            actions=data["action"],  # (Tp, Da)
            instruction=data['language_instruction'].lower(),
            dataset=self.dataset_name
        )

class HERepackTransform(RepackTransform):
    num_past_frames: int = 0
    action_chunk_size: int = 1

    def __call__(self, data: dict[str, Any], metadata=None, action_scale=None, action_shift=None, **kwargs) -> dict[str, Any]:
        assert (metadata is not None
            and action_scale is not None
            and action_shift is not None), "metadata must be provided for HERepackTransform"
        hand_states = np.asarray(data["observation.hand_joints"],dtype=np.float32 ) # [14]
        arm_states = np.asarray(data["observation.arm_joints"], dtype=np.float32) # [14]
        states = np.concatenate((hand_states, arm_states))

        raw_action = np.asarray(data["action"], dtype=np.float32)

        action_norm = raw_action * action_scale + action_shift
        # action_norm = 2 * (raw_action - action_min)/(action_max - action_min) - 1 # FIXME: TOO slow. coould be loading from stats or this per batch normalization calculation

        result = dict(
            observations=[
                pt_to_pil(data["observation.images.egocentric"])
            ],  # list of PIL Image
            states=states,
            actions=action_norm,  # (Tp, Da)
            instruction=metadata.episodes[data["episode_index"].item()]["instruction"], # TODO: verify for different tasks
            task="pick_dumpling_toy_and_turn_and_walk_and_squat_to_put_on_chair",
        )
        return result

    def delta_timestamps(self, fps: int):
        delta = {}
        delta["observation.images.egocentric"] = [
            -t / fps for t in range(self.num_past_frames, -1, -1)
        ]

        delta["observation.hand_joints"] = [
            -t / fps for t in range(self.num_past_frames, -1, -1)
        ]
        delta["observation.arm_joints"] = [
            -t / fps for t in range(self.num_past_frames, -1, -1)
        ]

        delta["action"] = [t / fps for t in range(self.action_chunk_size)]
        return delta

class HEPostPreRepackTransform(RepackTransform):
    dataset_name: str = "humanoid-everyday"
    num_past_frames: int = 0
    action_chunk_size: int = 16
    use_delta_actions: bool = True
    pad_action_dim: int | None = None
    pad_state_dim: int | None = None

    def __call__(
        self,
        data: dict[str, Any],
        **kwargs,
    ) -> dict[str, Any]:
        # HACK: different action format determines robot type
        if data["action.joint_angles"].shape[1] == 26:
            # H1 robot
            raw = data["action.joint_angles"]
            actions = np.concatenate([
                raw[:, 6:12][::-1], # from left thumb to little
                np.zeros((raw.shape[0], 1), dtype=np.float32), # pad finger to 7 dof
                raw[:, 0:6][::-1], # from right thumb to little
                np.zeros((raw.shape[0], 1), dtype=np.float32), # pad finger to 7 dof
                raw[:, 12:], # left arm + right arm (each from should to wrist)
            ], axis=1).astype(np.float32)
            hand = data["observation.hand_joints"]
            states = np.concatenate([
                hand[:, 6:12],
                np.zeros((hand.shape[0], 1), dtype=np.float32), # pad finger to 7 dof
                hand[:, 0:6],
                np.zeros((hand.shape[0], 1), dtype=np.float32), # pad finger to 7 dof 
                data["observation.arm_joints"]
            ], axis=1).astype(np.float32)
        else:
           actions = data["action.joint_angles"].astype(np.float32)
           states = np.concatenate((data["observation.hand_joints"], data["observation.arm_joints"]), axis=1).astype(np.float32)


        if self.use_delta_actions:
            actions = actions[1:] - actions[:-1]
            action_mask = data["action_mask"][1:]
        else:
            action_mask = data["action_mask"]

        # Expand action_mask to (T, Da) by repeating along last dimension
        if action_mask.ndim == 1:
            action_mask = np.repeat(action_mask[:, None], actions.shape[1], axis=1)

        states, _ = pad_to_len(states, self.pad_state_dim) if self.pad_state_dim is not None else (states, None)
        if self.pad_action_dim is not None:
            actions, _ = pad_to_len(actions, self.pad_action_dim)
            mask, _  = pad_to_len(action_mask, self.pad_action_dim)
        else:
            mask = np.ones_like(actions, dtype=bool)
            # mask = action_mask
        return {
            "observations": [pt_to_pil(img, normalized=False) for img in data["observation.images.egocentric"]],
            "states": states,
            "actions": actions,
            "instruction": str(data["task"]).lower(),
            "dataset": data.get("dataset_name", self.dataset_name),
            "actions_mask": mask,# torch.from_numpy(mask),
            "obs_mask": data["obs_mask"], #torch.from_numpy(data["obs_mask"]),
        }

class HEPretrainRepackTransform(RepackTransform):
    dataset_name: str = "humanoid-everyday"
    num_past_frames: int = 0
    action_chunk_size: int = 1
    use_delta_actions: bool = True
    robot_type: str = "g1"
    # pad_action_dim: int | None = None
    # pad_state_dim: int | None = None

    def _to_numpy(self, value, dtype=np.float32):
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        return np.asarray(value, dtype=dtype)

    def _ensure_2d(self, value, dim: int, name: str):
        arr = self._to_numpy(value)
        if arr.ndim == 1:
            if arr.shape[0] != dim:
                raise ValueError(f"{name} expected shape ({dim},), got {arr.shape}")
            return arr.reshape(1, dim)
        if arr.ndim == 2:
            if arr.shape[1] != dim:
                raise ValueError(f"{name} expected shape (T,{dim}), got {arr.shape}")
            return arr
        raise ValueError(f"{name} expected 1D or 2D, got {arr.shape}")

    def _rpy_to_rot6d(self, rpy):
        from scipy.spatial.transform import Rotation as R
        rot = R.from_euler("xyz", rpy).as_matrix()
        return rot[:, :2].reshape(6)

    def _delta_rpy(self, rpy_seq):
        from scipy.spatial.transform import Rotation as R
        rot = R.from_euler("xyz", rpy_seq).as_matrix()
        rel = rot[1:] @ np.transpose(rot[:-1], (0, 2, 1))
        return R.from_matrix(rel).as_euler("xyz", degrees=False).astype(np.float32)

    def _build_actions(self, data: dict[str, Any]) -> np.ndarray:
        left_wrist_xyz = self._ensure_2d(
            data["action.wrists.left.xyz"], 3, "action.wrists.left.xyz"
        )
        left_wrist_rpy = self._ensure_2d(
            data["action.wrists.left.rpy"], 3, "action.wrists.left.rpy"
        )
        right_wrist_xyz = self._ensure_2d(
            data["action.wrists.right.xyz"], 3, "action.wrists.right.xyz"
        )
        right_wrist_rpy = self._ensure_2d(
            data["action.wrists.right.rpy"], 3, "action.wrists.right.rpy"
        )

        def tip_xyz(keys):
            for key in keys:
                if key in data:
                    return self._ensure_2d(data[key], 3, key)
            return np.zeros_like(left_wrist_xyz)

        left_thumb = tip_xyz(["action.hands.left_thumb.xyz"])
        left_index = tip_xyz(["action.hands.left_index.xyz"])
        left_middle = tip_xyz(
            ["action.hands.left_middle.xyz", "action.hands.left_middle_finger.xyz"]
        )
        left_ring = tip_xyz(["action.hands.left_ring_finger.xyz"])
        left_little = tip_xyz(["action.hands.left_little_finger.xyz"])

        right_thumb = tip_xyz(["action.hands.right_thumb.xyz"])
        right_index = tip_xyz(["action.hands.right_index.xyz"])
        right_middle = tip_xyz(
            ["action.hands.right_middle.xyz", "action.hands.right_middle_finger.xyz"]
        )
        right_ring = tip_xyz(["action.hands.right_ring_finger.xyz"])
        right_little = tip_xyz(["action.hands.right_little_finger.xyz"])

        if self.use_delta_actions:
            if left_wrist_xyz.shape[0] < 2:
                raise ValueError(
                    "use_delta_actions requires at least 2 action frames; "
                    "increase action_chunk_size or disable delta."
                )
            left_wrist_xyz = left_wrist_xyz[1:] - left_wrist_xyz[:-1]
            right_wrist_xyz = right_wrist_xyz[1:] - right_wrist_xyz[:-1]
            left_wrist_rpy = self._delta_rpy(left_wrist_rpy)
            right_wrist_rpy = self._delta_rpy(right_wrist_rpy)

            left_thumb = left_thumb[1:] - left_thumb[:-1]
            left_index = left_index[1:] - left_index[:-1]
            left_middle = left_middle[1:] - left_middle[:-1]
            left_ring = left_ring[1:] - left_ring[:-1]
            left_little = left_little[1:] - left_little[:-1]

            right_thumb = right_thumb[1:] - right_thumb[:-1]
            right_index = right_index[1:] - right_index[:-1]
            right_middle = right_middle[1:] - right_middle[:-1]
            right_ring = right_ring[1:] - right_ring[:-1]
            right_little = right_little[1:] - right_little[:-1]

            left_rot_pad = np.zeros_like(left_wrist_rpy)
            right_rot_pad = np.zeros_like(right_wrist_rpy)
        else:
            left_rot6d = np.stack(
                [self._rpy_to_rot6d(rpy) for rpy in left_wrist_rpy], axis=0
            )
            right_rot6d = np.stack(
                [self._rpy_to_rot6d(rpy) for rpy in right_wrist_rpy], axis=0
            )
            left_wrist_rpy = left_rot6d
            right_wrist_rpy = right_rot6d
            left_rot_pad = None
            right_rot_pad = None

        left_wrist = (
            np.concatenate([left_wrist_xyz, left_wrist_rpy, left_rot_pad], axis=1)
            if self.use_delta_actions
            else np.concatenate([left_wrist_xyz, left_wrist_rpy], axis=1)
        )
        right_wrist = (
            np.concatenate([right_wrist_xyz, right_wrist_rpy, right_rot_pad], axis=1)
            if self.use_delta_actions
            else np.concatenate([right_wrist_xyz, right_wrist_rpy], axis=1)
        )

        actions = np.concatenate(
            [
                left_wrist,
                left_thumb,
                left_index,
                left_middle,
                left_ring,
                left_little,
                right_wrist,
                right_thumb,
                right_index,
                right_middle,
                right_ring,
                right_little,
            ],
            axis=1,
        ).astype(np.float32)
        return actions

    def __call__(
        self,
        data: dict[str, Any],
        metadata=None,
        **kwargs,
    ) -> dict[str, Any]:
        hand_states = self._to_numpy(data["observation.hand_joints"])
        arm_states = self._to_numpy(data["observation.arm_joints"])
        if hand_states.ndim == 2:
            hand_states = hand_states[-1]
        if arm_states.ndim == 2:
            arm_states = arm_states[-1]
        states = np.concatenate((hand_states, arm_states), axis=0).astype(np.float32)

        try:
            actions = self._build_actions(data)
        except Exception as e:
            print("Error building actions:")
            print(data)

            import traceback
            traceback.print_exc()
            raise e

        instruction = ""
        if metadata is not None and "episode_index" in data:
            episode_idx = int(self._to_numpy(data["episode_index"]).reshape(-1)[0])
            if (
                hasattr(metadata, "episodes")
                and metadata.episodes is not None
                and episode_idx < len(metadata.episodes)
            ):
                instruction = metadata.episodes[episode_idx].get("instruction", "")
        if not instruction and "task" in data:
            instruction = data["task"]
        if isinstance(instruction, bytes):
            instruction = instruction.decode("utf-8")
            instruction = self.nice_instruction(instruction.split("/")[1])

        img = data["observation.images.egocentric"]
        if hasattr(img, "dim") and img.dim() == 4:
            img = img[0]
        elif isinstance(img, np.ndarray) and img.ndim == 4:
            img = img[0]

        return dict(
            observations=[pt_to_pil(img, normalized=False)],  # list of PIL Image
            states=states,
            actions=actions,
            instruction=str(instruction).lower(),
            dataset=data.get("dataset_name", self.dataset_name),
        )

    def nice_instruction(self, instruction: str) -> str:
        instruction = instruction.replace("_", " ").replace("-", " ")
        instruction = re.sub(r"\s+", " ", instruction)
        return instruction.strip()

    def delta_timestamps(self, fps: int):
        delta = {}
        delta["observation.hand_joints"] = [
            -t / fps for t in range(self.num_past_frames, -1, -1)
        ]
        delta["observation.arm_joints"] = [
            -t / fps for t in range(self.num_past_frames, -1, -1)
        ]

        action_keys = [
            "action.joint_angles",
            "action.wrists.left.xyz",
            "action.wrists.left.rpy",
            "action.wrists.right.xyz",
            "action.wrists.right.rpy",
            "action.hands.left_thumb.xyz",
            "action.hands.left_index.xyz",
            "action.hands.right_thumb.xyz",
            "action.hands.right_index.xyz",
        ]
        if self.robot_type == "h1":
            action_keys += [
                "action.hands.left_middle_finger.xyz",
                "action.hands.left_ring_finger.xyz",
                "action.hands.left_little_finger.xyz",
                "action.hands.right_middle_finger.xyz",
                "action.hands.right_ring_finger.xyz",
                "action.hands.right_little_finger.xyz",
            ]
        else:
            action_keys += [
                "action.hands.left_middle.xyz",
                "action.hands.right_middle.xyz",
            ]
        action_len = (
            self.action_chunk_size + 1 if self.use_delta_actions else self.action_chunk_size
        )
        for key in action_keys:
            delta[key] = [t / fps for t in range(action_len)]
        return delta


class MixedRepackTransform(RepackTransform):
    dataset_name: str = "mixed"

    num_past_frames: int = 0
    action_chunk_size: int = 1
    use_delta_actions: bool = True
    robot_type: str = "mixed"
    stage: str = "pretrain"  # "pretrain" or "postpre"
    pad_action_dim: int | None = None
    pad_state_dim: int | None = None

    def model_post_init(self, __context: Any) -> None:
        if self.stage == "pretrain":
            self._he_repacker = HEPretrainRepackTransform.model_validate(self.model_dump())
        else:
            self._he_repacker = HEPostPreRepackTransform.model_validate(self.model_dump())
        self._egodex_repacker = EgodexRepackTransform.model_validate(self.model_dump())

    def __call__(self, data: dict[str, Any], **kwargs) -> dict[str, Any]:
        dataset_name = data.get("dataset_name", "default")
        match(dataset_name):
            case "egodex":
                return self._egodex_repacker(data, **kwargs)
            case "humanoid-everyday":
                return self._he_repacker(data, **kwargs)
        return data


class HRDTRepackTransform(HEPretrainRepackTransform):
    dataset_name: str = "humanoid-everyday"
    action_chunk_size: int = 16
    use_delta_actions: bool = True
    robot_type: str = "g1"


class LerobotRepackTransform(RepackTransform):
    image_keys: List[str] = Field(default_factory=lambda: ["front_stereo_left"])
    # traj_keys: list[str] = field(default_factory=lambda: ["left_future_traj_2d"])
    state_key: str = "state"
    action_key: str = "action"
    num_past_frames: int = 0
    action_chunk_size: int = 12 #
    is_delta_actions: bool = False

    conditions: list[str] = Field(default_factory=lambda: [])

    def __call__(self, data: dict[str, Any], **kwargs) -> dict[str, Any]:
        result = {# TODO here ...
            "observations": [
                pt_to_pil(data[f"observation.images.{key}"], normalized=False)
                for key in self.image_keys
            ], # list of PIL Image
            "states": np.array(data[f"observation.{self.state_key}"], dtype=np.float32), # (To, Da)
            "actions": np.array(data[self.action_key], dtype=np.float32),  # (Tp, Da)
            # action_mask=torch.tensor(data["absolute_action_mask"]),  # (Tp, Da)
            "action_is_pad": np.array(data["action_is_pad"], dtype=bool),
            "instruction": data["task"].lower(),
            # "task": "tabletop_grasp",
            "actions_mask": data["action_is_pad"]
            # dataset=self.dataset_name
        } 
        for key in self.conditions:
            result[key] = [pt_to_pil(data[f"observation.images.{key}"], normalized=False)]

        # result["conditions"] =  [
        #     pt_to_pil(data[f"observation.images.{key}"], normalized=False)
        #     for key in self.conditions
        # ]
            # (H, W) 2D traj heatmap
        # (To, n_conditions, H, W)
        if len(self.conditions) > 0:
            result["traj2ds"] = torch.stack([data[f"observation.images.{key}"] for key in self.conditions]) # dim=0
            # # HACK 
            # result["traj2ds"]  = torch.zeros_like(result["traj2ds"]) # zero out the traj2d conds for now

        if self.is_delta_actions and data["action_is_pad"].sum() > 0:
            """ delta action must be set to zero rather than repeating the last action """
            zero_action = np.zeros((1, result["actions"].shape[-1]), dtype=np.float32)
            result["actions"][result["actions_mask"]] = zero_action
        
        # cond_tensors = [data[f"observation.images.{key}"] for key in self.conditions if f"observation.images.{key}" in data]
        # if len(cond_tensors) > 0:
        #     result["traj2ds"] = torch.cat(cond_tensors, dim=0)


        return result
    
    def delta_timestamps(self, fps):
        
        delta = {}
        for key in self.image_keys:
            delta["observation.images." + key] = [-t/fps for t in range(self.num_past_frames, -1, -1)]

        for key in self.conditions:
            delta["observation.images." + key] = [-t/fps for t in range(self.num_past_frames, -1, -1)]
        
        delta["observation."+self.state_key] = [-t/fps for t in range(self.num_past_frames,-1,-1)]
        delta[self.action_key] = [t/fps for t in range(self.action_chunk_size)]
        
        return delta

class WeGR00TRepackTransform(RepackTransform):
    dataset_name: str = "we-g1-teleop"
    conditions: list[str] = Field(default_factory=lambda: [])
    image_keys: List[str] = Field(default_factory=lambda: ["video"])
    state_key: str = "state"
    action_key: str = "action"
    instruction_key: str = "annotation.human.task_description"
    # input data:
    # annotation.human.task_description ['g1-fullbody/pick_dumpling_toy_and_turn_and_walk_and_squat_to_put_on_chair']
    # video (1, 1, 480, 640, 3) nint8 (0-255) np array
    # state (1, 32) np array    
    # action (30, 36) np array

    pad_action_dim: int | None = None
    pad_state_dim: int | None = None

    def __call__(self, data: dict[str, Any], **kwargs) -> dict[str, Any]:
        states, _ = pad_to_len(data[self.state_key], self.pad_state_dim) if self.pad_state_dim is not None else (data[self.state_key], None)

        if self.pad_action_dim is not None:
            actions, mask = pad_to_len(data[self.action_key], self.pad_action_dim) 
        else:
            actions = data[self.action_key]
            mask = np.ones_like(actions, dtype=bool)

        result = {
            "observations": [
                Image.fromarray(data[key][0, 0])
                for key in self.image_keys
            ], # list of PIL Image
            "states": states.astype(np.float32), # (To, Da)
            "actions": actions.astype(np.float32),  # (Tp, Da)
            "instruction": data[self.instruction_key][0].lower(), # "pick up toy rhinocero." # FIXME: hack here!!!
            "actions_mask": mask, #(Tp, Da)
        } 
        return result


class GR00TRepackTransform(RepackTransform):
    data_config: str = Field(default="fourier_gr1_arms_only", description="GR00T data config name")
    embodiment_tag: str = Field(default="new_embodiment", description="Embodiment tag")
    video_backend: str = Field(default="decord", description="Video backend")
    
    def __call__(self, data: dict[str, Any], **kwargs) -> dict[str, Any]:
        return data
    
    def delta_timestamps(self, fps):
        return {}


class PushTRepackTransform(RepackTransform):

    def __call__(self, data: dict[str, Any], **kwargs) -> dict[str, Any]:
        return {
            "observations": [
                Image.fromarray(np.transpose(i, (1,2,0))) for i in data["image"].astype(np.uint8)
            ],  # list of PIL Image
            "states": data["agent_pos"],  # (To, Da)
            "actions": data["action"],  # (Tp, Da)
            "dataset_name": "pusht",
            "task": "pusht",
            "instruction": "push the object to the target location"
        }
    
class RLDS_Simple_RepackTransform(RepackTransform):
    dataset_name: str = "rlds"
    image_keys: str = "observation.image_camera_0"
    state_key: str = "observation.proprio"
    action_key: str = "action"
    instruction_key: str = "task.language_instruction"

    def __call__(self, data: dict[str, Any], **kwargs) -> dict[str, Any]:
        return dict(
            # observations=[
            #     Image.fromarray(data["observation"][key][0])
            #     for key in data["observation"].keys()
            #     if "image" in key
            # ],  # list of PIL Image
            # states=torch.tensor(data["observation"]["proprio"]),  # (To, Da)
            actions=np.array(data["action"], dtype=np.float32),  # (Tp, Da)
            # instruction=data["task"]["language_instruction"].decode().lower(),
            # task="tabletop_grasp",
            dataset=self.dataset_name,
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

class ActionStandardizationTransform(BaseModel):
    stat_relative_path: str  # mean std path; dataset_statistics/MeanStd.json
    mean: Optional[List[float]] = None
    std: Optional[List[float]] = None
    action_norm_masks: List[bool] = [
        True,
        True,
        True,
        True,
        True,
        True,
        False,
    ]
    action_norm_type: str = 'normal'

    def model_post_init(self, __context: Any) -> None:
        with (get_asset_dir() / self.stat_relative_path).open("r") as f:
            meta = json.load(f)
        self.mean = meta['action']["mean"]
        self.std = meta['action']["std"]
        assert self.action_norm_type == 'normal'

    def __call__(self, data: dict[str, Any], **kwargs) -> dict[str, Any]:
        assert self.mean is not None and self.std is not None
        # data["states"] = (data["states"] - mean) / std
        # if "states" in data:
        #     states_normalized = (data["states"] - np.array(self.mean)) / np.array(self.std)
        #     data["states"] = np.where(self.action_norm_masks, states_normalized, data["states"])
        actions_normalized = (data["actions"] - np.array(self.mean)) / np.array(self.std)
        data["actions"] = np.where(self.action_norm_masks, actions_normalized, data["actions"])
        return data
    
    # def reverse_call(self, array: Any, **kwargs) -> Any:
    #     assert self.mean is not None and self.std is not None
    #     reversed_array = (array * np.array(self.std)) + np.array(self.mean)
    #     return reversed_array

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

class ActionStateTransform(BaseModel):
    def __call__(self, data: dict[str, Any], **kwargs) -> dict[str, Any]:
        return data


class ResizeImage(BaseModel):
    size: int | tuple[int, int] = (256, 480)  # H,W

    def __call__(self):
        try:
            from torchvision.transforms import v2
        except:
            from torchvision import transforms as v2
        return v2.Resize(self.size, interpolation=v2.InterpolationMode.NEAREST)

    @property
    def resolution(self) -> Tuple[int, int]:
        if isinstance(self.size, int):
            return (self.size, self.size)
        elif isinstance(self.size, list) and len(self.size) == 2:
            return (self.size[0], self.size[1])
        elif isinstance(self.size, tuple) and len(self.size) == 2:
            return self.size
        else:
            raise ValueError("size should be int or list of two ints")


class ColorJitter(BaseModel):
    brightness: float | tuple[float, float] = Field(default_factory=lambda: 0.2)
    contrast: float | tuple[float, float] = Field(default_factory=lambda: (0.8, 1.2))
    saturation: float | tuple[float, float] = Field(default_factory=lambda: (0.8, 1.2))
    hue: float | tuple[float, float] = Field(default_factory=lambda: 0.05)

    def __call__(self):
        try:
            from torchvision.transforms import v2
        except:
            from torchvision import transforms as v2
        return v2.ColorJitter(
            brightness=self.brightness,
            contrast=self.contrast,
            saturation=self.saturation,
            hue=self.hue,
        )


class CenterCrop(BaseModel):
    size: int | tuple[int, int] = (224,224) # H,W

    def __call__(self):
        try:
            from torchvision.transforms import v2
        except:
            from torchvision import transforms as v2
        return v2.CenterCrop(self.size)


class Normalize(BaseModel):
    mean: float | List[float] = Field(default_factory=lambda: [0.485, 0.456, 0.406])
    std: float | List[float] = Field(default_factory=lambda: [0.229, 0.224, 0.225])

    def __call__(self):
        try:
            from torchvision.transforms import v2
        except:
            from torchvision import transforms as v2
        mean = [self.mean] * 3 if isinstance(self.mean, float) else self.mean
        std = [self.std] * 3 if isinstance(self.std, float) else self.std
        return v2.Normalize(mean, std)  # type: ignore


class Qwen25VL_ModelTransform(BaseModel):
    resize: ResizeImage
    action_dim: int = 7  # action dimension

    def __call__(
        self, data: dict[str, Any], vlm_processor, action_tokenizer, **kwargs
    ) -> dict[str, Any]:
        """
        data: dict with keys:
            - instruction: str
            - observations: List[Image] (To x PIL)
            - states: (To, Da) torch.Tensor
            - action: (Tp, Da) torch.Tensor
        """
        resizer = self.resize()
        images = [resizer(img) for img in data["observations"]]
        instruction = data["instruction"]
        state = data["states"]
        action = data["actions"]
        inputs = self.build_qwenvl_inputs(
            vlm_processor, action_tokenizer, [images], [instruction], [state], [action]
        )
        labels = copy.deepcopy(inputs["input_ids"])
        # keep loss on the answer + EOS + formatting tokens
        labels[:, : -(len(action[0]) + 2)] = IGNORE_INDEX
        inputs["labels"] = labels
        inputs["dataset_name"] = data.get("dataset", "unknown")
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
        assert len(images) == len(
            instructions
        ), "Images and instructions must have the same length"

        for imgs, instruction, action in zip(images, instructions, actions):
            # print(action[0].shape)
            content = [{"type": "image", "image": img} for img in imgs]
            content.append({"type": "text", "text": instruction})
            user_msg = {"role": "user", "content": content}
            assistant_msg = {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": action_tokenizer(action[0])}
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
        image_inputs, video_inputs = process_vision_info(messages)  # type: ignore
        inputs = vlm_processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        assert (
            inputs["input_ids"] >= action_tokenizer.action_token_begin_idx
        ).sum() == self.action_dim
        return inputs

class Qwen3VL_ModelTransform(BaseModel):

    def __call__(
        self, data: dict[str, Any], vlm_processor, **kwargs
    ) -> dict[str, Any]:
        return data
    
class SaltPepperNoise:
    """
    Adds salt-and-pepper noise to a tensor image.
    - prob: total probability of altering a pixel
    - salt_vs_pepper: fraction of salt (white) vs pepper (black)
    """
    def __init__(self, prob=0.01, salt_vs_pepper=0.5):
        self.prob = prob
        self.salt_vs_pepper = salt_vs_pepper

    def __call__(self, img):
        # img is PIL Image
        arr = np.array(img)  # H x W x C

        # Random mask
        rnd = np.random.rand(arr.shape[0], arr.shape[1])

        pepper_mask = rnd < (self.prob * (1 - self.salt_vs_pepper))
        salt_mask   = rnd > (1 - self.prob * self.salt_vs_pepper)

        arr = arr.copy()
        # Pepper = 0
        arr[pepper_mask] = 0
        # Salt = 255
        arr[salt_mask] = 255

        return Image.fromarray(arr)

    def __repr__(self):
        return f"{self.__class__.__name__}(prob={self.prob}, salt_vs_pepper={self.salt_vs_pepper})"

class GaussianNoise(BaseModel):
    mean: float = 0
    std: float = 3
    prob_skip: float = 0.1


    def __call__(self, img: Image.Image):

        rnd = np.random.rand()
        if rnd < self.prob_skip:
            return img  # 10% chance to skip adding noise

        arr = np.array(img).astype(np.float32)

        # mean = int(np.random.uniform(-self.mean, self.mean)) # real mean
        noise = np.random.normal(self.mean, self.std, arr.shape)
        arr_noisy = arr + noise

        arr_noisy = np.clip(arr_noisy, 0, 255).astype(np.uint8)
        return Image.fromarray(arr_noisy)

class Qwen3VL_7d_ModelTransform(ModelTransform):
    resize: ResizeImage = Field(default_factory=lambda: ResizeImage(size=(224)))
    center_crop: CenterCrop = Field(default_factory=lambda: CenterCrop(size=224))
    color_jitter: ColorJitter = Field(default_factory=lambda: ColorJitter())
    gaussian_noise: GaussianNoise = Field(default_factory=lambda: GaussianNoise(mean=0, std=3, prob_skip=0.1))
    img_aug: bool = False

    def __call__(
        self, data: dict[str, Any], vlm_processor=None, action_tokenizer=None, no_aug=False, **kwargs
    ) -> dict[str, Any]:
        
        transforms = [self.resize(), self.center_crop()]
        if self.img_aug and not no_aug:
            transforms.append(self.gaussian_noise)
            transforms.append(self.color_jitter())
        t = v2.Compose(transforms)

        images = [t(img) for img in data["observations"]]
        instruction = data["instruction"]
        state = data["states"]
        action = data["actions"]
        inputs, num_action_tokens = self.build_qwenvl_inputs(
            vlm_processor, action_tokenizer, images, instruction, state, action
        )
        labels = copy.deepcopy(inputs["input_ids"])
        # keep loss on the answer + EOS + formatting tokens
        labels[:, : -(num_action_tokens + 2)] = IGNORE_INDEX 
        inputs["labels"] = labels
        inputs["dataset_name"] = data.get("dataset", "unknown")
        inputs["raw_actions"] = data["raw_actions"]
        inputs["actions_mask"] = data["actions_mask"]
        inputs["raw_images"] = images
        return inputs

    def build_qwenvl_inputs(
        self,
        vlm_processor,
        action_tokenizer,
        imgs,
        instruction,
        state,
        action,
        **kwargs,
    ) -> tuple[dict, int]:
        from qwen_vl_utils import process_vision_info

        """adapted from Qwen_VL_Interface.build_qwenvl_inputs"""
        messages = []
        # num_answer_tokens_list = []

        tokenized_action = action_tokenizer(action)
        num_action_tokens = len(tokenized_action)
        # raw_action_tokens = vlm_processor.tokenizer(tokenized_action)["input_ids"]
        # num_answer_tokens = len(raw_action_tokens)
        # num_answer_tokens_list.append(num_answer_tokens)

        content = [{"type": "image", "image": img} for img in imgs]
        content.append({"type": "text", "text": instruction})
        user_msg = {"role": "user", "content": content}

        assistant_msg = {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "!"} # HACK: use ! as action placeholder
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
        image_inputs, video_inputs = process_vision_info(messages, image_patch_size=16)  # type: ignore
        inputs = vlm_processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        # HACK replace "!" token with action tokens
        exclamation_token_id = vlm_processor.tokenizer.convert_tokens_to_ids("!")
        device = inputs["input_ids"].device
        assert torch.all(inputs["input_ids"][:, -3] == exclamation_token_id)
        inputs["input_ids"] = torch.concat([
            inputs["input_ids"][:, :-3], 
            torch.tensor([tokenized_action], device=device).repeat(inputs["input_ids"].shape[0], 1),
            inputs["input_ids"][:, -2:]
        ], dim=1)
        inputs["attention_mask"] = torch.concat([
            inputs["attention_mask"][:, :-3], 
            torch.ones((inputs["attention_mask"].shape[0], len(tokenized_action)), device=device),
            inputs["attention_mask"][:, -2:]
        ], dim=1)
        return inputs, num_action_tokens

class Hfm_Together_ModelTransform(ModelTransform):
    resize: ResizeImage = Field(default_factory=lambda: ResizeImage(size=(224)))
    center_crop: CenterCrop = Field(default_factory=lambda: CenterCrop(size=224))
    color_jitter: ColorJitter = Field(default_factory=lambda: ColorJitter())
    gaussian_noise: GaussianNoise = Field(default_factory=lambda: GaussianNoise(mean=0, std=3, prob_skip=0.1))
    img_aug: bool = False
    
    # for mixed dataset with different image sizes
    adaptive_resize: bool = False
    img_sizes: dict[str, Any] = Field(default_factory=lambda: {
        "egodex": [270, 480],
        "we"    : [240, 320],
    })

    def __call__(
        self, data: dict[str, Any], vlm_processor=None, no_aug=False, **kwargs
    ) -> dict[str, Any]:
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
            center_crop = CenterCrop(size=tuple(target_size))() # type: ignore
        else:
            resizer = self.resize()
            center_crop = self.center_crop()
            
        t1 = v2.Compose([
            resizer,
            center_crop,
            self.color_jitter() if do_img_aug else v2.Identity(),
        ])

        images = [t1(img) for img in data["observations"]]
        instruction = data["instruction"]

        inputs = self.build_qwenvl_inputs(
            vlm_processor, images, instruction,
        )
        # labels = copy.deepcopy(inputs["input_ids"])
        # keep loss on the answer + EOS + formatting tokens
        # labels[:, : -(num_action_tokens + 2)] = IGNORE_INDEX 
        # inputs["labels"] = labels
        inputs["dataset_name"] = data.get("dataset", "unknown")
        inputs["raw_actions"] = data["raw_actions"]
        if "actions_mask" in data:
            inputs["actions_mask"] = data["actions_mask"]

        inputs["raw_images"] = images
        inputs['actions'] = data["actions"]
        inputs['states'] = data["states"]
        inputs['instruction'] = data["instruction"]
        return inputs

    def build_qwenvl_inputs(
        self,
        vlm_processor,
        imgs,
        instruction,
        # state,
        # action,
        **kwargs,
    ) -> tuple[dict, int]:
        from qwen_vl_utils import process_vision_info

        """adapted from Qwen_VL_Interface.build_qwenvl_inputs"""
        messages = []
        # num_answer_tokens_list = []

        # raw_action_tokens = vlm_processor.tokenizer(tokenized_action)["input_ids"]
        # num_answer_tokens = len(raw_action_tokens)
        # num_answer_tokens_list.append(num_answer_tokens)

        content = [{"type": "image", "image": img} for img in imgs]
        content.append({"type": "text", "text": instruction})
        user_msg = {"role": "user", "content": content}

        # assistant_msg = {
        #     "role": "assistant",
        #     "content": [
        #         {"type": "text", "text": "!"} # HACK: use ! as action placeholder
        #     ],  # squeeze batch dim
        # }
        messages.append([user_msg])

        # Prepare text prompts using processor
        # default process is json --> message --> texts --> input_ids
        texts = [
            vlm_processor.apply_chat_template(
                m, tokenize=False, add_generation_prompt=True
            )
            for m in messages
        ]
        image_inputs, video_inputs = process_vision_info(messages, image_patch_size=16)  # type: ignore
        inputs = vlm_processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        # HACK replace "!" token with action tokens
        # exclamation_token_id = vlm_processor.tokenizer.convert_tokens_to_ids("!")
        # device = inputs["input_ids"].device
        # assert torch.all(inputs["input_ids"][:, -3] == exclamation_token_id)
        # inputs["input_ids"] = torch.concat([
        #     inputs["input_ids"][:, :-3], 
        #     torch.tensor([tokenized_action], device=device).repeat(inputs["input_ids"].shape[0], 1),
        #     inputs["input_ids"][:, -2:]
        # ], dim=1)
        # inputs["attention_mask"] = torch.concat([
        #     inputs["attention_mask"][:, :-3], 
        #     torch.ones((inputs["attention_mask"].shape[0], len(tokenized_action)), device=device),
        #     inputs["attention_mask"][:, -2:]
        # ], dim=1)
        return inputs


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


class VQVAEModelTransform(BaseModel):
    def __call__(self, data: dict[str, Any], **kwargs) -> dict[str, Any]:
        filtered_data = {}
        filtered_data["actions"] = torch.tensor(data["actions"])  # [T, 48]
        return filtered_data


class OpenvlaPrismatic_ModelTransform(BaseModel):
    predict_stop_token: bool = False
    resize_image: bool = True

    def __call__(
        self,
        data: dict[str, Any],
        action_tokenizer,
        prompt_builder_fn,
        image_transform,
        resize_image_shape,
        **kwargs,
    ) -> dict[str, Any]:
        # adapted from openvla RLDSBatchTransform.__call__
        """Converts a RLDS batch to the format expected by the OpenVLA collator/models."""
        # dataset_name = data["dataset"]
        action = data["actions"][0] if data["actions"].shape[0] == 1 else data["actions"][:] # (1, 7) or (T, 7)
        img = data["observations"][0]
        lang = data["instruction"].lower()
        # img.save(f"/home/boqian/liboqian_code/we_learn/tmp_dir/transform/{int(time.time() * 1000)}.png")

        action_token_ids, num_action_tokens = action_tokenizer(action)
        # action_token_ids is now List[int], not string
        
        # Construct Chat-based Prompt (without action tokens)
        prompt_builder = prompt_builder_fn("openvla")
        conversation = [
            {"from": "human", "value": f"What action should the robot take to {lang}?"},
            {"from": "gpt", "value": ""},  # empty string, action tokens are added separately
        ]
        for turn in conversation:
            prompt_builder.add_turn(turn["from"], turn["value"])

        # print(prompt_builder.get_prompt())
        # Tokenize prompt (without action tokens)
        prompt_input_ids = action_tokenizer.tokenizer(
            prompt_builder.get_prompt(), add_special_tokens=True
        ).input_ids

        # print("prompt_input_ids", prompt_input_ids)
        # print("len(prompt_input_ids)", len(prompt_input_ids))
        # print("action_token_ids", action_token_ids)
        # print("len(action_token_ids)", len(action_token_ids))
        # print("\n=== Token 详细信息 ===")
        # for idx, token_id in enumerate(prompt_input_ids):
        #     token_text = action_tokenizer.tokenizer.decode([token_id])
        #     print(f"位置 {idx}: token_id={token_id}, token='{token_text}'")
        
        # insert action token ids at the end of the prompt
        input_ids = prompt_input_ids[:-1] + action_token_ids + [prompt_input_ids[-1]]
        labels = list(input_ids)

        # assert img.size[0] != 224
        if self.resize_image:
            img = img.resize((resize_image_shape[1], resize_image_shape[0]), Image.LANCZOS) # type: ignore
        # assert img.size[0] == 224

 
        # Tensorize =>> Run Image Transform to get `pixel_values` =>> Return
        #   =>> IMPORTANT :: IF WE'RE USING HF LLM.forward(..., labels=labels), SHIFTING HAPPENS _INSIDE_ MODEL!
        # import torch
        input_ids, labels = torch.tensor(input_ids), torch.tensor(labels)
        pixel_values = image_transform(img)
        # print("pixel_values", pixel_values)

        # [CRITICAL] We do not want to take the loss for anything but the predicted action tokens!
        labels[: -(num_action_tokens + 1)] = IGNORE_INDEX
        if not self.predict_stop_token:
            labels[-1] = IGNORE_INDEX

        return dict(
            pixel_values=pixel_values, input_ids=input_ids, labels=labels
        )  # FIXME , dataset_name=dataset_name


class OpenvlaFlow_ModelTransform(BaseModel):

    def __call__(
        self,
        data: dict[str, Any],
        prompt_builder_fn,
        tokenizer,
        image_transform,
        **kwargs,
    ) -> dict[str, Any]:
        """Converts a RLDS batch to the format expected by the OpenVLA collator/models."""
        dataset_name, action = data["dataset"], data["actions"][0]
        img = data["observations"][0]
        lang = data["instruction"].lower()

        # Construct Chat-based Prompt =>> Input is default query + language instruction, output are the action tokens
        prompt_builder = prompt_builder_fn("openvla")
        conversation = [
            {"from": "human", "value": f"What action should the robot take to {lang}?"},
            {"from": "gpt", "value": ""},
        ]
        for turn in conversation:
            prompt_builder.add_turn(turn["from"], turn["value"])

        # Tokenize (w/ `base_tokenizer`)
        input_ids = tokenizer(
            prompt_builder.get_prompt(), add_special_tokens=True
        ).input_ids
        labels = list(input_ids)

        # Tensorize =>> Run Image Transform to get `pixel_values` =>> Return
        #   =>> IMPORTANT :: IF WE'RE USING HF LLM.forward(..., labels=labels), SHIFTING HAPPENS _INSIDE_ MODEL!
        # import torch
        # import numpy as np
        # try:
        from torchvision.transforms import v2

        # except:
        # from torchvision import transforms as v2
        input_ids, labels = torch.tensor(input_ids), torch.tensor(labels)
        pixel_values = image_transform(img)

        # # [CRITICAL] We do not want to take the loss for anything but the predicted action tokens!
        # labels[: -(len(action) + 1)] = IGNORE_INDEX

        data.update(
            dict(pixel_values=pixel_values, input_ids=input_ids, labels=labels)
        )  # FIXME , dataset_name=dataset_name
        # data["action_mask"] = torch.ones_like(data["actions"]) * (1. - data["action_mask"].unsqueeze(-1).to(torch.float32))
        # data["observations"] = torch.stack([image_transform(img) for img in data["observations"]])

        t = v2.Compose(
            [
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
            ]
        )
        data["observations"] = torch.stack([t(img) for img in data["observations"]]) # (To, 3, H, W)
        for k, v in list(data.items()):
            if isinstance(v, np.ndarray):
                data[k] = torch.from_numpy(v)
        return data

    
class DitPolicy_ModelTransform(ModelTransform):
    resize: ResizeImage
    color_jitter: ColorJitter
    center_crop: CenterCrop
    normalize: Normalize
    img_aug: bool = True

    def __call__(
        self,
        data: dict[str, Any],
        no_aug: bool = False,
        **kwargs,
    ) -> dict[str, Any]:
        # import torch
        # try:
        # from torchvision.transforms import v2
        # except:
        #     from torchvision import transforms as v2
        do_img_aug = False if no_aug else self.img_aug
        t = v2.Compose(
            [
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                self.resize(),
                self.center_crop(),
                self.color_jitter() if do_img_aug else v2.Identity(),
                self.normalize(),
            ]
        )
        data["observations"] = torch.stack([t(img) for img in data["observations"]]) # (To, 3, H, W)
        
        actions = data["actions"]
        actions_t = torch.as_tensor(actions)
        action_is_pad_t = torch.as_tensor(data["action_is_pad"])
        mask = torch.ones_like(actions_t) * (1. - action_is_pad_t.unsqueeze(-1).to(torch.float32))
        return dict(
            imgs={"cam0":data["observations"]}, # (1,3,H,W),
            obs=data["states"], # (1,M)
            # conditions=data["traj2ds"], # (3,H,W)
            conditions=data.get("conditions", {}),
            # actions=data["actions"], #(Tp,Da)
            # mask=data.get("mask", np.ones_like(data["actions"])), #(Tp,Da)
            actions=actions,
            mask=mask,
            text_instructions=data["instruction"], # str
            frame_idx=data.get("frame_index", -1) # int
        )

class Vlt_ModelTransform(ModelTransform):
    features: dict[str, str] = Field(default_factory=lambda:{})
    resize: ResizeImage = Field(default_factory=lambda: ResizeImage(size=224))
    center_crop: CenterCrop = Field(default_factory=lambda: CenterCrop(size=224))
    color_jitter: ColorJitter
    normalize: Normalize
    img_aug: bool = False

    def __call__(
        self,
        data: dict[str, Any],
        no_aug=False,
        **kwargs,
    ) -> dict[str, Any]:
        do_img_aug = False if no_aug else self.img_aug
        # from torchvision.transforms import v2
        t1 = v2.Compose([
            self.color_jitter()
        ]) if do_img_aug else v2.Identity()
        data["raw_images"] = [t1(img) for img in data["observations"]]
        t2 = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
        ])
        data["observations"] = torch.stack([t2(img) for img in data["raw_images"]]) # (To, 3, H, W)
        t3 = v2.Compose([
            self.normalize()
        ])
        for feature in self.features.keys():
            processs = kwargs.get(feature, None)
            if processs is not None:
                assert isinstance(data["raw_images"][0], Image.Image) # PIL Image
                data[feature] = processs(data["raw_images"], return_tensors="pt").pixel_values
            else:
                raise ValueError(f"Warning: no process found for feature {feature}!")
        if "left_future_traj_2d" in data:
            data["traj2ds"] = torch.stack([t3(t2(img)) for img in data["left_future_traj_2d"]])
        return data
    
    def get_image_processors(self):
        image_processors = {}
        kwargs = {
            "resume_download": None,
            "do_rescale": True,
            # "local_files_only": True,  # Avoid downloading from the internet, please manully disable for the first run.
        }
        for vision_backbone, model_path in self.features.items():
            if vision_backbone == "siglip":
                from transformers import SiglipImageProcessor

                image_processors[vision_backbone] = (
                    SiglipImageProcessor.from_pretrained(model_path, **kwargs)
                )
            elif vision_backbone == "dinov2":
                from transformers import BitImageProcessor

                image_processors[vision_backbone] = BitImageProcessor.from_pretrained(
                    model_path, **kwargs
                )
            else:
                raise ValueError(f"Unsupported vision backbone: {vision_backbone}")
        return image_processors

class Hfm_Action_ModelTransform(ModelTransform):
    features: dict[str, str] = Field(default_factory=lambda:{})
    resize: ResizeImage = Field(default_factory=lambda: ResizeImage(size=224))
    center_crop: CenterCrop = Field(default_factory=lambda: CenterCrop(size=224))
    color_jitter: ColorJitter
    normalize: Normalize
    img_aug: bool = False

    def __call__(
        self,
        data: dict[str, Any],
        no_aug=False,
        **kwargs,
    ) -> dict[str, Any]:
        do_img_aug = False if no_aug else self.img_aug
        # from torchvision.transforms import v2
        t1 = v2.Compose([
            self.color_jitter()
        ]) if do_img_aug else v2.Identity()
        data["raw_images"] = [t1(img) for img in data["observations"]]
        t2 = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
        ])
        data["observations"] = torch.stack([t2(img) for img in data["raw_images"]]) # (To, 3, H, W)
        t3 = v2.Compose([
            self.normalize()
        ])
        for feature in self.features.keys():
            processs = kwargs.get(feature, None)
            if processs is not None:
                assert isinstance(data["raw_images"][0], Image.Image) # PIL Image
                data[feature] = processs(data["raw_images"], return_tensors="pt").pixel_values
            else:
                raise ValueError(f"Warning: no process found for feature {feature}!")
        if "left_future_traj_2d" in data:
            data["traj2ds"] = torch.stack([t3(t2(img)) for img in data["left_future_traj_2d"]])
        return data
    
    def get_image_processors(self):
        image_processors = {}
        kwargs = {
            "resume_download": None,
            "do_rescale": True,
            "local_files_only": True,  # Avoid downloading from the internet, please manully disable for the first run.
        }
        for vision_backbone, model_path in self.features.items():
            if vision_backbone == "siglip":
                from transformers import SiglipImageProcessor

                image_processors[vision_backbone] = (
                    SiglipImageProcessor.from_pretrained(model_path, **kwargs)
                )
            elif vision_backbone == "dinov2":
                from transformers import BitImageProcessor

                image_processors[vision_backbone] = BitImageProcessor.from_pretrained(
                    model_path, **kwargs
                )
            else:
                raise ValueError(f"Unsupported vision backbone: {vision_backbone}")
        return image_processors


class DiffusionPolicy_ModelTransform(ModelTransform):
    normalize: Normalize
    
    def __call__(
        self,
        data: dict[str, Any],
        no_aug: bool = False,
        **kwargs,
    ) -> dict[str, Any]:

        # import torch
        # try:
        # from torchvision.transforms import v2
        # except:
        #     from torchvision import transforms as v2
        t = v2.Compose(
            [
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                self.normalize(),
            ]
        )
        return dict(
            image=torch.stack([t(i) for i in data["observations"]]), # (To, 3, H, W)
            agent_pos=data["states"], # (To, Da)
            action=data["actions"], #(Tp, Da)
        )

class GR00TModelTransform(BaseModel):
    def __call__(
        self,
        data: dict[str, Any],
        **kwargs,
    ) -> dict[str, Any]:
        return data

class DataTransform(BaseModel):
    # fmt: off
    repack: Union[
        Annotated[IdentityTransform, cmd("identity")],
        Annotated[EgodexRepackTransform, cmd("egodex")],
        Annotated[Egodex_Openvla_RepackTransform, cmd("egodex-openvla")],
        Annotated[RLDSRepackTransform, cmd("rlds")],
        Annotated[HumanoidRepackTransform, cmd("humanoid")],
        Annotated[RLDS_Simple_RepackTransform, cmd("rlds-simple")],
        Annotated[LerobotRepackTransform, cmd("lerobot")],
        Annotated[GR00TRepackTransform, cmd("gr00t")],
        Annotated[PushTRepackTransform, cmd("pusht")],
        Annotated[HERepackTransform, cmd("humanoideveryday")],
        Annotated[HEPretrainRepackTransform, cmd("humanoideveryday-pretrain")],
        Annotated[HEPostPreRepackTransform, cmd("he-postpre")],
        Annotated[HRDTRepackTransform, cmd("hrdt")],
        Annotated[WeGR00TRepackTransform, cmd("gr00t-we")],
        Annotated[MixedRepackTransform, cmd("mixed")],
        Annotated[SimpleRepackTransform, cmd("simple")]
    ]
    action_state: Union[
        Annotated[IdentityTransform, cmd("identity")],
        Annotated[ActionMaxMinTransform, cmd("maxmin")],
        Annotated[ActionStandardizationTransform, cmd("standardization")],
    ]
    model: Union[
        Annotated[IdentityTransform, cmd("identity")],
        Annotated[Qwen25VL_ModelTransform, cmd("qwen25-vl")],
        Annotated[OpenvlaPrismatic_ModelTransform, cmd("openvla-prismatic")],
        Annotated[VQVAEModelTransform, cmd("vqvae")],
        Annotated[OpenvlaFlow_ModelTransform, cmd("openvla-flow")],
        Annotated[DitPolicy_ModelTransform, cmd("ditp")],
        Annotated[Vlt_ModelTransform, cmd("vlt")],
        Annotated[DiffusionPolicy_ModelTransform, cmd("dp")],
        Annotated[GR00TModelTransform, cmd("gr00t")],
        Annotated[Hfm_ModelTransform, cmd("hfm")],
        Annotated[Qwen3VL_ModelTransform, cmd("qwen3vl")],
        Annotated[Qwen3VL_7d_ModelTransform, cmd("qwen3vl-7d")],
        Annotated[Hfm_Action_ModelTransform, cmd("hfm-action")],
        Annotated[Hfm_Together_ModelTransform, cmd("hfm-together")]
    ]
    # fmt: on

    def __call__(self, data: dict[str, Any], **kwargs) -> dict[str, Any]:
        data = self.repack(data, **kwargs)
        data = self.action_state(data, **kwargs)
        data = self.model(data, **kwargs)
        return data
