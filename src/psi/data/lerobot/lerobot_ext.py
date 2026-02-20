from __future__ import annotations
from typing import Any, Dict, TYPE_CHECKING
if TYPE_CHECKING:
    from psi.config.data_lerobot import LerobotDataConfig
    from psi.config.data_simple import SimpleDataConfig

import torch
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, MultiLeRobotDataset
from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata

# from lerobot.datasets.lerobot_dataset import LeRobotDataset, MultiLeRobotDataset
# from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
# from lerobot.datasets.lerobot_dataset import LeRobotDataset, MultiLeRobotDataset, LeRobotDatasetMetadata

# from config import DataConfig
# from we.config import DataConfig
# from data.in_memory import InMemoryWrapperDataset
# from utils.misc import overlay
from functools import partial

class LeRobotDatasetWrapper(torch.utils.data.Dataset):
    """ A wrapper around LeRobotDataset to support multiple datasets.
    """

    def __init__(
            self, 
        data_cfg: LerobotDataConfig|SimpleDataConfig, 
        split: str = "train"
    ):
        repo_ids = data_cfg.train_repo_ids if split == "train" else data_cfg.val_repo_ids
        # = ["vlt-fr3-frontstereo+wrist+side.rest.randall-37312-256-traj2d-obj63" + ("" if split == "train" else "-test")]
        first_repo = repo_ids[0] if isinstance(repo_ids, list) else repo_ids
        dataset_meta = LeRobotDatasetMetadata(first_repo, f"{data_cfg.root_dir}/{first_repo}")
        delta_timestamps = data_cfg.transform.repack.delta_timestamps(dataset_meta.fps)# type: ignore

        if len(repo_ids) > 1:
            root_dir = data_cfg.root_dir
            lerobot_dataset_class = MultiLeRobotDataset
        else:
            repo_ids = first_repo
            root_dir = f"{data_cfg.root_dir}/{first_repo}"
            lerobot_dataset_class = LeRobotDataset

        # eps = [515] if split == "train" else None      
        self.base_dataset = LeRobotDataset( # lerobot_dataset_class
            repo_ids,# type: ignore
            root=root_dir,
            delta_timestamps=delta_timestamps,
            image_transforms=None,
            # episodes=eps
        )
        self._cache = {}

    def __getitem__(self, idx) -> dict:
        # idx = 0 # FIXME
        return self.base_dataset[idx]
    
    def __len__(self):
        return len(self.base_dataset)

    @property
    def episode_data_index(self):
        return self.base_dataset.episode_data_index # type: ignore

    @property
    def num_episodes(self):
        return self.base_dataset.num_episodes
    
    @property
    def num_frames(self):
        return self.base_dataset.num_frames
    
    @property
    def meta(self):
        return self.base_dataset.meta # type: ignore

    @property
    def stats(self):
        return self.base_dataset.stats if type(self.base_dataset) == MultiLeRobotDataset else self.base_dataset.meta.stats # type: ignore
