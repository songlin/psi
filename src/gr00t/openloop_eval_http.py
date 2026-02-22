import dataclasses
import json
import pathlib
from typing import Any
import urllib.request

import numpy as np
import pyarrow.parquet as pq
import imageio.v3 as iio

from psi.deploy.helpers import RequestMessage, ResponseMessage


def _load_tasks(tasks_path: pathlib.Path) -> dict[int, str]:
    tasks = {}
    if not tasks_path.exists():
        return tasks
    with tasks_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            tasks[int(rec.get("task_index", 0))] = rec.get("task", "")
    return tasks


def _load_frame(video_path: pathlib.Path, frame_index: int) -> np.ndarray:
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    return iio.imread(video_path, index=frame_index)


@dataclasses.dataclass
class Args:
    host: str = "127.0.0.1"
    port: int = 5555
    dataset_path: str = "/hfm/data/simple/G1WholebodyBendPick-v0-psi0"
    episode_index: int = 0
    target_index: int = 0
    video_key: str = "observation.rgb_head_stereo_left"


def main() -> None:
    import tyro

    args = tyro.cli(Args)
    dataset_root = pathlib.Path(args.dataset_path).resolve()
    meta_dir = dataset_root / "meta"

    info = json.loads((meta_dir / "info.json").read_text())
    chunks_size = int(info.get("chunks_size", 1000))
    data_path_fmt = info.get("data_path", "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet")

    tasks = _load_tasks(meta_dir / "tasks.jsonl")

    ep_index = args.episode_index
    chunk_id = ep_index // chunks_size
    parquet_path = dataset_root / data_path_fmt.format(
        episode_chunk=chunk_id,
        episode_index=ep_index,
    )

    table = pq.read_table(parquet_path)
    df = table.to_pandas()
    if args.target_index >= len(df):
        raise ValueError(f"target_index {args.target_index} out of range (len={len(df)})")

    row = df.iloc[args.target_index]
    instruction = ""
    if "task" in row:
        instruction = str(row["task"])
    elif "task_index" in row:
        instruction = tasks.get(int(row["task_index"]), "")

    if "observation.proprio_joint_positions" not in row or "observation.amo_policy_command" not in row:
        raise KeyError(
            "Dataset is missing observation.proprio_joint_positions / observation.amo_policy_command. "
            "Use the original simple dataset or add these fields in conversion."
        )
    proprio = np.asarray(row["observation.proprio_joint_positions"], dtype=np.float32)
    command = np.asarray(row["observation.amo_policy_command"], dtype=np.float32)

    video_path = dataset_root / "videos" / f"chunk-{chunk_id:03d}" / args.video_key / f"episode_{ep_index:06d}.mp4"
    frame = _load_frame(video_path, args.target_index)

    request = RequestMessage(
        image={args.video_key: frame},
        instruction=instruction,
        history={},
        state={
            "proprio_joint_positions": proprio,
            "amo_policy_command": command,
        },
        condition={},
        gt_action=[],
        dataset_name="simple",
        timestamp=str(row.get("timestamp", "")),
    )

    payload = json.dumps(request.serialize()).encode("utf-8")
    url = f"http://{args.host}:{args.port}/act"
    req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as resp:
        resp_payload = json.loads(resp.read().decode("utf-8"))

    if "status" in resp_payload:
        raise RuntimeError(f"Server error: {resp_payload['status']}")

    response = ResponseMessage.deserialize(resp_payload)
    action = response.action
    print(f"got action shape={action.shape} err={response.err}")

    gt_action = np.asarray(row["action"], dtype=np.float32)
    print(f"gt action shape={gt_action.shape}")
    if gt_action.size > 0:
        print(
            f"gt action: {np.array2string(gt_action.reshape(-1)[:10], precision=4, suppress_small=True)}"
        )
    if action.size > 0:
        print(
            f"pred action: {np.array2string(action.reshape(-1)[:10], precision=4, suppress_small=True)}"
        )


if __name__ == "__main__":
    main()
