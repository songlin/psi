import dataclasses
import importlib.util
import pathlib
from typing import Any

import numpy as np
import tyro

from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
from gr00t.policy.server_client import PolicyClient


def load_modality_config(module_path: str, config_var: str | None) -> dict[str, Any]:
    module_path = pathlib.Path(module_path)
    spec = importlib.util.spec_from_file_location(module_path.stem, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load modality config module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if config_var:
        if not hasattr(module, config_var):
            raise AttributeError(f"{config_var} not found in {module_path}")
        return getattr(module, config_var)

    for name in dir(module):
        if name.endswith("_config") and isinstance(getattr(module, name), dict):
            return getattr(module, name)
    raise AttributeError(f"No *_config dict found in {module_path}")


def _to_state_array(value: Any) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    return arr[None, None, :]


@dataclasses.dataclass
class Args:
    host: str = "127.0.0.1"
    port: int = 5555
    dataset_path: str = "/hfm/data/real_teleop_g1/lerobot/Hold_lunch_bag_with_both_hands_and_squat_to_put_on_the_coffee_table"
    modality_config_path: str = "gr00t/configs/modality/new_embodiment_task_description.py"
    modality_config_var: str | None = None
    episode_index: int = 0
    target_index: int = 0


def main() -> None:
    args = tyro.cli(Args)

    modality_configs = load_modality_config(args.modality_config_path, args.modality_config_var)
    loader = LeRobotEpisodeLoader(
        dataset_path=args.dataset_path,
        modality_configs=modality_configs,
    )

    episode_meta = loader.episodes_metadata[args.episode_index]
    episode_id = episode_meta["episode_index"]
    episode_length = int(episode_meta["length"])

    action_horizon = len(modality_configs["action"].delta_indices)
    max_start = episode_length - action_horizon
    if max_start <= 0:
        raise ValueError("Episode too short for the configured action horizon.")

    target_index = max(0, min(args.target_index, max_start - 1))
    step_indices = [target_index]

    df = loader._load_parquet_data(episode_id)
    video_data = loader._load_video_data(episode_id, np.array(step_indices))
    video_key = modality_configs["video"].modality_keys[0]
    frames_by_index = {idx: frame for idx, frame in zip(step_indices, video_data[video_key])}

    policy = PolicyClient(host=args.host, port=args.port)

    action_keys = modality_configs["action"].modality_keys
    state_keys = modality_configs["state"].modality_keys
    language_key = modality_configs["language"].modality_keys[0]

    losses: dict[str, np.ndarray] = {}
    counts: dict[str, int] = {}

    for idx in step_indices:
        row = df.iloc[idx]
        text = row[f"language.{language_key}"]
        if not isinstance(text, str):
            text = str(text)

        obs = {
            "video": {
                video_key: frames_by_index[idx][None, None, :, :, :].astype(np.uint8),
            },
            "state": {key: _to_state_array(row[f"state.{key}"]) for key in state_keys},
            "language": {
                language_key: [[text]],
            },
        }

        pred_action, _ = policy.get_action(obs)

        state_shapes = " ".join(f"{k}:{obs['state'][k].shape}" for k in state_keys)
        print(f"step={idx} instr='{text}'")
        print(f"image={obs['video'][video_key].shape} state={state_shapes}")

        for key in action_keys:
            gt_seq = [
                np.asarray(df.iloc[idx + d][f"action.{key}"], dtype=np.float32)
                for d in modality_configs["action"].delta_indices
            ]
            gt_action = np.stack(gt_seq, axis=0)
            pred = pred_action[key][0].astype(np.float32)

            if pred.shape != gt_action.shape:
                raise ValueError(
                    f"Shape mismatch for {key}: pred {pred.shape} vs gt {gt_action.shape}"
                )

            gt_first = gt_action[0]
            pred_first = pred[0]
            gt_str = np.array2string(gt_first, precision=4, suppress_small=True)
            pred_str = np.array2string(pred_first, precision=4, suppress_small=True)
            print(f"action.{key} gt0={gt_str} pred0={pred_str}")

            abs_err = np.abs(pred_first - gt_first)
            if key not in losses:
                losses[key] = abs_err
                counts[key] = 1
            else:
                losses[key] += abs_err
                counts[key] += 1

    print("per-dim L1 (first horizon frame):")
    for key in action_keys:
        mean_err = losses[key] / max(counts[key], 1)
        dims = " ".join(f"{i}:{v:.6f}" for i, v in enumerate(mean_err.tolist()))
        print(f"  {key}: {dims}")


if __name__ == "__main__":
    main()
