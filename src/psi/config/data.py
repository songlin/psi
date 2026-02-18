from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, List, Optional, Union

import numpy as np
import torch
from pydantic import BaseModel, Field, model_validator
from torch.utils.data import Dataset as TorchDataset

from psi.config.transform import DataTransform
from psi.data.dataset import Dataset as MapStyleDataset
from psi.data.dataset import IterableDataset
from psi.utils import get_data_dir

if TYPE_CHECKING:
    from psi.data.sampler import DatasetSpec

class DataConfig(BaseModel):
    transform: DataTransform

    def __call__(self, split: str = "train", transform_kwargs={}, **kwargs) -> Any:
        raise NotImplementedError

class DummyDataConfig(DataConfig):
    # boilerplate data config
    # inherit transform from DataConfig
    # put dataset related configs here, eg., file paths
    def __call__(self, split: str = "train", transform_kwargs={}, **kwargs) -> Any:
        # from torch.utils.data import Dataset as TorchDataset

        class DummyDataset(TorchDataset):
            def __init__(self, num_samples=1000):
                self.num_samples = num_samples

            def __len__(self):
                return self.num_samples

            def __getitem__(self, idx):
                x = torch.randn(1, 7)  # random input tensor
                return x

        train_dataset = DummyDataset(num_samples=1000)
        return MapStyleDataset(self, train_dataset, transform_kwargs=transform_kwargs)


class EgoDexDataConfig(DataConfig):
    root_dir: str
    upsample_rate: int = 3
    chunk_size: int = 16
    train_repo_ids: List[str] = []
    val_repo_ids: List[str] = []
    # val_chunk_size: int = 16 # original window size
    use_delta_actions: bool = True
    load_retarget: bool = False

    def __call__(self, split: str = "train", transform_kwargs={}, **kwargs) -> Any:
        from psi.data.egodex.egodex_dataset import EgoDexDataset

        dataset = EgoDexDataset(
            data_root=self.root_dir,
            upsample_rate=self.upsample_rate,
            val=(split != "train"),
            chunk_size=self.chunk_size,
            img_history_size=1,
            use_delta_actions=self.use_delta_actions,
            load_retarget=self.load_retarget,
        )
        return MapStyleDataset(
            self, dataset, transform_kwargs=transform_kwargs, **kwargs
        )

    def mock(self, split: str = "train", transform_kwargs={}, **kwargs) -> Any:
        dataset = self.__call__(split, transform_kwargs=transform_kwargs, **kwargs)
        # return dataset[0]
        n_samples = min(10, len(dataset))
        import random

        indices = random.sample(range(len(dataset)), n_samples)
        return [dataset[i] for i in indices]


class HumanoidDataConfig(DataConfig):
    chunk_size: int = 1
    root_dir: str
    require_image: bool = True
    states_based: bool = True
    action_dim: int = 32
    absolute_mask: List[bool] = [True] * 32


class HERawDataConfig(DataConfig):
    root_dir: str
    robot_type: str = "both"
    episodes: Optional[List[int]] = None
    chunk_size: int = 1
    use_delta_actions: bool = True
    upsample_rate: int = 1

    def __call__(self, split: str = "train", transform_kwargs={}, **kwargs) -> Any:
        from psi.data.humanoid.he_raw_dataset import HERawDataset

        repack = self.transform.repack
        num_past_frames = getattr(repack, "num_past_frames", 0)
        action_chunk_size = getattr(repack, "action_chunk_size", 1)
        use_delta_actions = getattr(repack, "use_delta_actions", False)
        self.chunk_size = int(action_chunk_size)
        self.use_delta_actions = bool(use_delta_actions)
        # FIXME create a validation dataset
        dataset = HERawDataset(
            data_root=self.root_dir,
            num_past_frames=num_past_frames,
            action_chunk_size=action_chunk_size,
            upsample_rate=self.upsample_rate,
            use_delta_actions=use_delta_actions,
            robot_type=self.robot_type,
            episodes=self.episodes,
        )
        return MapStyleDataset(self, dataset, transform_kwargs=transform_kwargs, **kwargs)

class MixedDataConfig(DataConfig):
    use_delta_actions: bool = True
    chunk_size: int = 1
    upsample_rate: int = 3

    egodex: dict[str, Any] = Field(default_factory=lambda: {
        "root_dir": "/hfm/data/egodex",
        "ratio": 0.5,
        "load_retarget": False
    })
    he: dict[str, Any] = Field(default_factory=lambda: {
        "root_dir": "/hfm/data/HE_RAW",
        "robot_type": "both", # g1, h1, or both
        "episodes": None,
        "num_past_frames": 0,
        "ratio": 0.5
    })
    sampler: str = "batch_mixture"  # or "token_mixture"
    tokens_per_device: int = 2048 # for token_mixture sampler

    def __call__(self, split: str = "train", transform_kwargs={}, **kwargs) -> Any:
        from psi.data.egodex.egodex_dataset import EgoDexDataset
        from psi.data.humanoid.he_raw_dataset import HERawDataset
        from psi.data.dataset import MixtureDataset

        if split == "val":
            he = HERawDataset(
                data_root=self.he["root_dir"],
                num_past_frames=self.he["num_past_frames"],
                action_chunk_size=self.chunk_size,
                upsample_rate=self.upsample_rate,
                use_delta_actions=self.use_delta_actions,
                robot_type=self.he["robot_type"],
                episodes=self.he["episodes"]
            )
            he_dataset = MapStyleDataset(self, he, transform_kwargs=transform_kwargs, **kwargs)
            return he_dataset

        egodex = EgoDexDataset(
            data_root=self.egodex["root_dir"],
            upsample_rate=self.upsample_rate,
            val=(split != "train"),
            chunk_size=self.chunk_size,
            img_history_size=1,
            use_delta_actions=self.use_delta_actions,
            load_retarget=self.egodex.get("load_retarget", False)
        )
        egodex_dataset = MapStyleDataset(
            self, egodex, transform_kwargs=transform_kwargs, **kwargs
        )
        he = HERawDataset(
            data_root=self.he["root_dir"],
            num_past_frames=self.he["num_past_frames"],
            action_chunk_size=self.chunk_size,
            upsample_rate=self.upsample_rate,
            use_delta_actions=self.use_delta_actions,
            robot_type=self.he["robot_type"],
            episodes=self.he["episodes"]
        )
        he_dataset = MapStyleDataset(self, he, transform_kwargs=transform_kwargs, **kwargs)

        # from torch.utils.data import ConcatDataset
        # mixed_dataset = ConcatDataset([egodex_dataset, he_dataset])

        mixed_dataset = MixtureDataset({
            egodex_dataset: self.egodex["ratio"],
            he_dataset: self.he["ratio"],
        }, num_samples_per_epoch=2*len(he_dataset))

        if self.sampler == "token_mixture":
            from psi.data.sampler import DatasetSpec
            dataset_specs = [
                DatasetSpec(
                    dataset_length=egodex_dataset.dataset_length,
                    prob=self.egodex["ratio"],
                    image_size=(270, 480), # H,W
                    tokens_per_image=18*30,
                ),
                DatasetSpec(
                    dataset_length=he_dataset.dataset_length,
                    prob=self.he["ratio"],
                    image_size=(240, 320), # H,W
                    tokens_per_image=16*20,
                ),
            ]
            mixed_dataset.specs = dataset_specs

        return mixed_dataset

class RLDSDataConfig(DataConfig):
    root_dir: str
    root_dir_test: Optional[str] = None
    data_mix: str = "grasp"
    image_aug: bool = True
    shuffle_buffer_size: int = 50_000

    rlds_aug: bool = True 
    train_repo_ids: List[str] = Field(default_factory=list)
    val_repo_ids: List[str] = Field(default_factory=list)

    # for RLDSDataset, please set "action_state" to "identity" 
    # and forward this parameter to the RLDSDataset in def create_datasets
    action_normalization_type: str = "bounds_q99"
    action_norm_masks: List[bool] = [
        True,
        True,
        True,
        True,
        True,
        True,
        False,
    ]

    action_pred_horizon: int = 1

    def model_post_init(self, __context: Any) -> None:
        if not os.path.isabs(self.root_dir):
            self.root_dir = str(get_data_dir() / self.root_dir)

    def __call__(self, split: str = "train", transform_kwargs={}, **kwargs) -> Any:
        from psi.data.rlds.dataset import RLDSDataset
        from pathlib import Path
        resize_image_resolution = (
            transform_kwargs.get("resize_image_shape", None) or
            self.transform.model.resize.resolution
        )
        assert len(self.train_repo_ids) ==1 and len(self.val_repo_ids)== 1, "train_repo_ids and val_repo_ids must be set"
        train_rlds_dataset = RLDSDataset(
            Path(f"{self.root_dir}/{self.train_repo_ids[0]}" if split=="train" else f"{self.root_dir}/{self.val_repo_ids[0]}"),
            self.data_mix,
            resize_resolution=resize_image_resolution,
            action_pred_horizon=self.action_pred_horizon,
            shuffle_buffer_size=self.shuffle_buffer_size,
            image_aug=self.image_aug,
            rlds_aug=self.rlds_aug,
            action_normalization_type=self.action_normalization_type,
            train=True,
            **kwargs
        )

        train_dataset = IterableDataset(
            self,
            train_rlds_dataset,
            transform_kwargs=transform_kwargs,
        )
        return train_dataset


class HEDataConfig(DataConfig):
    repo_id: str
    root_dir: str
    use_delta_actions: bool = True
    chunk_size: int = 1
    action_dim: int = 48
    episodes: Optional[List[int]] = None
    upsample_rate: Optional[int] = 3

    def __call__(
        self, split: str = "train", transform_kwargs={}, **kwargs
    ) -> MapStyleDataset:

        try:
            from lerobot.common.datasets.lerobot_dataset import (
                LeRobotDataset, LeRobotDatasetMetadata)
        except ImportError:
            print(
                "Please run: uv sync --group lerobot if you want to use LeRobot datasets."
            )

        print(f"using {self.root_dir} as root dir for LeRobot dataset {self.repo_id}")
        dataset_meta = LeRobotDatasetMetadata(
            self.repo_id, root=self.root_dir if self.root_dir else None
        )

        train_dataset = LeRobotDataset(
            self.repo_id,
            root=self.root_dir if self.root_dir else None,
            episodes=self.episodes,
            delta_timestamps=self.transform.repack.delta_timestamps(30),
            tolerance_s=1e-2,
            # episodes=[0] # DEBUG
        )
        repack_chunk = getattr(self.transform.repack, "action_chunk_size", None)
        if repack_chunk is not None:
            self.chunk_size = int(repack_chunk)
        repack_delta = getattr(self.transform.repack, "use_delta_actions", None)
        if repack_delta is not None:
            self.use_delta_actions = bool(repack_delta)
        transform_kwargs["metadata"] = dataset_meta

        action_min = dataset_meta.stats["action"]["min"]
        action_max = dataset_meta.stats["action"]["max"]

        transform_kwargs["action_scale"] = 2.0 / (action_max - action_min)
        transform_kwargs["action_shift"] = (
            -1.0 - action_min * transform_kwargs["action_scale"]
        )

        return MapStyleDataset(self, train_dataset, transform_kwargs=transform_kwargs)


class HRDTDataConfig(DataConfig):
    repo_id: str
    root_dir: str
    use_delta_actions: bool = True
    chunk_size: int = 16
    action_dim: int = 48
    episodes: Optional[List[int]] = None
    fps: int = 30

    def __call__(
        self, split: str = "train", transform_kwargs={}, **kwargs
    ) -> MapStyleDataset:

        try:
            from lerobot.common.datasets.lerobot_dataset import (
                LeRobotDataset, LeRobotDatasetMetadata)
        except ImportError:
            print(
                "Please run: uv sync --group lerobot if you want to use LeRobot datasets."
            )

        dataset_meta = LeRobotDatasetMetadata(
            self.repo_id, root=self.root_dir if self.root_dir else None
        )

        repack_chunk = getattr(self.transform.repack, "action_chunk_size", None)
        if repack_chunk is not None:
            self.chunk_size = int(repack_chunk)
        repack_delta = getattr(self.transform.repack, "use_delta_actions", None)
        if repack_delta is not None:
            self.use_delta_actions = bool(repack_delta)

        dataset_kwargs = {}
        if hasattr(self.transform.repack, "delta_timestamps"):
            dataset_kwargs["delta_timestamps"] = self.transform.repack.delta_timestamps(
                self.fps
            )

        train_dataset = LeRobotDataset(
            self.repo_id,
            root=self.root_dir if self.root_dir else None,
            episodes=self.episodes,
            tolerance_s=1e-2,
            **dataset_kwargs,
        )
        transform_kwargs["metadata"] = dataset_meta

        return MapStyleDataset(self, train_dataset, transform_kwargs=transform_kwargs)


class VLMDataConfig(DataConfig):
    root_dir: str = "examples/toy-vlm"

    dataset_use: str = ""
    data_flatten: bool = False
    data_packing: bool = False
    base_interval: int = 2
    max_pixels: int = 28 * 28 * 576
    min_pixels: int = 28 * 28 * 16
    video_max_frames: Optional[int] = 8
    video_min_frames: Optional[int] = 4
    video_max_pixels: int = 1024 * 28 * 28
    video_min_pixels: int = 256 * 28 * 28
    video_fps: float = 2
    model_type: str = "qwen3vl"

    def model_post_init(self, __context: Any) -> None:
        from psi.utils import resolve_path

        if not os.path.isabs(self.root_dir):
            self.root_dir = str(resolve_path(self.root_dir))

    def __call__(self, split: str = "train", transform_kwargs={}, **kwargs) -> Any:
        from qwen3vl.data.data_processor import LazySupervisedDataset

        dataset = LazySupervisedDataset(
            transform_kwargs["vlm_processor"],
            data_args=self,
        )
        return MapStyleDataset(self, dataset, transform_kwargs=transform_kwargs)

    def mock(self, split: str = "train", transform_kwargs={}, **kwargs) -> Any:
        dataset = self.__call__(split, transform_kwargs=transform_kwargs, **kwargs)
        return dataset[0]


class LerobotDataConfig(DataConfig):
    root_dir: str
    train_repo_ids: List[str] = Field(default_factory=list)
    val_repo_ids: List[str] = Field(default_factory=list)

    def __call__(self, split: str = "train", transform_kwargs={}, **kwargs) -> Any:
        try:
            from psi.data.lerobot import LeRobotDatasetWrapper
        except ImportError as e:
            print("Please run: uv sync --group lerobot if you want to use LeRobot datasets.")
            raise e

        train_dataset = LeRobotDatasetWrapper(self, split=split)
        return MapStyleDataset(self, train_dataset, transform_kwargs=transform_kwargs)

    def mock(self, split: str = "train", transform_kwargs={}, **kwargs) -> Any:
        dataset = self.__call__(split, transform_kwargs=transform_kwargs, **kwargs)
        return dataset[0]


class PushTDataConfig(DataConfig):
    dataset_path: str = "/hfm/cache/pusht_cchi_v7_replay.zarr.zip"
    pred_horizon: int = 16
    obs_horizon: int = 2
    action_horizon: int = 8
    action_dim: int = 2
    # disable default data augmentation including normalization
    data_aug: bool = False 

    def __call__(self, split: str = "train", transform_kwargs={}, **kwargs) -> Any:
        # from psi.data.dataset import Dataset as MapStyleDataset
        try:
            from psi.data.pusht.dataset import PushTImageDataset
        except ImportError as e:
            print("Please refer to README to install pusht dependencies.")
            raise e
        
        dataset = PushTImageDataset(
            dataset_path=self.dataset_path,
            pred_horizon=self.pred_horizon,
            obs_horizon=self.obs_horizon,
            action_horizon=self.action_horizon,
            data_aug=self.data_aug
        )
        return MapStyleDataset(self, dataset, transform_kwargs=transform_kwargs)

    def mock(self, split: str = "train", transform_kwargs={}, **kwargs) -> Any:
        dataset =  self.__call__(split, transform_kwargs=transform_kwargs, **kwargs)
        return dataset[0]

class GR00TDataConfig(DataConfig):
    """
    # boqian: please note that this dataset is also used for our model and InternVLA model
    """
    action_chunk_size: int = 16

    root_dir: str
    dataset_paths: List[str] = Field(
        default_factory=list, description="List of dataset paths (relative to root_dir)"
    )
    data_config: str = "fourier_gr1_arms_only"
    embodiment_tag: str = "new_embodiment"
    video_backend: str = "decord"
    balance_dataset_weights: bool = True
    balance_trajectory_weights: bool = True
    preload_all: bool = False

    def __call__(self, split: str = "train", transform_kwargs={}, **kwargs) -> Any:
        try:
            from psi.data.gr00t.lerobot_wrapper import GR00TLeRobotWrapper
        except ImportError as e:
            print("Please install GR00T dependencies.")
            exit(1)

        dataset = GR00TLeRobotWrapper(
            data_cfg=self,
            split=split,
            root_dir=self.root_dir,
            dataset_paths=self.dataset_paths,
            data_config=self.data_config,
            embodiment_tag=self.embodiment_tag,
            video_backend=self.video_backend,
            balance_dataset_weights=self.balance_dataset_weights,
            balance_trajectory_weights=self.balance_trajectory_weights,
            preload_all=self.preload_all,
        )

        return MapStyleDataset(self, dataset, transform_kwargs=transform_kwargs)

class SimpleDataConfig(DataConfig):
    """ it uses lerobot dataset under the hood """

    root_dir: str

    train_repo_ids: List[str] = Field(default_factory=list)
    val_repo_ids: List[str] = Field(default_factory=list)

    action_chunk_size: int = 16
    video_backend: str = "decord"

    def model_post_init(self, __context: Any) -> None:
        if len(self.val_repo_ids) == 0:
            self.val_repo_ids = [self.train_repo_ids[0]]

        assert self.action_chunk_size == self.transform.repack.action_chunk_size

    def __call__(self, split: str = "train", transform_kwargs={}, **kwargs) -> Any:
        try:
            from psi.data.lerobot import LeRobotDatasetWrapper
        except ImportError as e:
            print("Please run: uv sync --group lerobot if you want to use LeRobot datasets.")
            raise e

        train_dataset = LeRobotDatasetWrapper(self, split=split)
        return MapStyleDataset(self, train_dataset, transform_kwargs=transform_kwargs)
    
