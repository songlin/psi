#!/usr/bin/env python3
import argparse
import json
import math
import shutil
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

STATE_SLICES = [
    ("left_hand_thumb", 29, 32),
    ("left_hand_middle", 34, 36),
    ("left_hand_index", 32, 34),
    ("right_hand", 36, 43),
    ("left_arm", 15, 22),
    ("right_arm", 22, 29),
    ("torso_rpy", 13, 15),
    ("torso_rpy_extra", 12, 13),
]

ACTION_SLICES = [
    ("left_hand_thumb", 29, 32),
    ("left_hand_middle", 34, 36),
    ("left_hand_index", 32, 34),
    ("right_hand", 36, 43),
    ("left_arm", 15, 22),
    ("right_arm", 22, 29),
    ("torso_rpy", 13, 15),
    ("torso_rpy_extra", 12, 13),
]


def load_jsonl(path: Path):
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            json.dump(row, f, separators=(",", ":"))
            f.write("\n")


def modality_dict():
    def _entry(start, end, original_key, absolute=True):
        return {
            "start": start,
            "end": end,
            "rotation_type": None,
            "absolute": absolute,
            "dtype": "float32",
            "original_key": original_key,
        }

    return {
        "state": {
            "left_hand": _entry(0, 7, "states"),
            "right_hand": _entry(7, 14, "states"),
            "left_arm": _entry(14, 21, "states"),
            "right_arm": _entry(21, 28, "states"),
            "rpy": _entry(28, 31, "states"),
            "height": _entry(31, 32, "states"),
        },
        "action": {
            "left_hand": _entry(0, 7, "action"),
            "right_hand": _entry(7, 14, "action"),
            "left_arm": _entry(14, 21, "action"),
            "right_arm": _entry(21, 28, "action"),
            "rpy": _entry(28, 31, "action"),
            "height": _entry(31, 32, "action"),
            "torso_vx": _entry(32, 33, "action", absolute=False),
            "torso_vy": _entry(33, 34, "action", absolute=False),
            "torso_vyaw": _entry(34, 35, "action", absolute=False),
            "target_yaw": _entry(35, 36, "action"),
        },
        "video": {
            "rs_view": {
                "original_key": "observation.images.egocentric"
            }
        },
        "annotation": {
            "human.task_description": {
                "original_key": "task_index"
            }
        },
    }


def build_vectors(proprio, cmd, action):
    to = proprio.shape[0]
    ta = action.shape[0]

    # states: match to_psi0_state_format ordering
    states = np.concatenate(
        [proprio[:, s:e] for _, s, e in STATE_SLICES] + [cmd[:to, 6:7]],
        axis=1,
    ).astype(np.float32)

    # actions: match to_psi0_action_format ordering
    if cmd.shape[0] < to:
        raise ValueError(
            f"observation.amo_policy_command too short: {cmd.shape[0]} < {to}"
        )
    cmd_future = cmd[to:to + ta]
    if cmd_future.shape[0] < ta:
        if cmd_future.shape[0] == 0:
            last = cmd[to - 1:to]
        else:
            last = cmd_future[-1:]
        pad = np.repeat(last, ta - cmd_future.shape[0], axis=0)
        cmd_future = np.concatenate([cmd_future, pad], axis=0)
    actions = np.concatenate(
        [action[:, s:e] for _, s, e in ACTION_SLICES]
        + [
            cmd_future[:, 6:7],  # base height
            cmd_future[:, 0:2],  # vx, vy
            np.zeros((ta, 1), dtype=np.float32),  # vyaw unused
            cmd_future[:, 3:4],  # target yaw
        ],
        axis=1,
    ).astype(np.float32)
    return states, actions


def build_proprio_obs(proprio, cmd):
    # hand joints: left thumb(29:32), left index(32:34), left middle(34:36),
    #              right thumb(36:39), right index(39:41), right middle(41:43)
    hand = np.concatenate(
        [
            proprio[:, 29:32],
            proprio[:, 32:34],
            proprio[:, 34:36],
            proprio[:, 36:39],
            proprio[:, 39:41],
            proprio[:, 41:43],
        ],
        axis=1,
    ).astype(np.float32)

    # arm joints: left arm(15:22), right arm(22:29)
    arm = np.concatenate([proprio[:, 15:22], proprio[:, 22:29]], axis=1).astype(
        np.float32
    )

    # leg joints: 12 leg joints + 3 waist joints (12:15)
    leg = np.concatenate([proprio[:, 0:12], proprio[:, 12:15]], axis=1).astype(
        np.float32
    )

    # base height from command
    prev_height = cmd[: proprio.shape[0], 6].astype(np.float32)

    return hand, arm, leg, prev_height


def stats_block(arr):
    arr = np.asarray(arr)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return {
        "mean": arr.mean(axis=0).astype(np.float32).tolist(),
        "std": arr.std(axis=0).astype(np.float32).tolist(),
        "min": arr.min(axis=0).astype(np.float32).tolist(),
        "max": arr.max(axis=0).astype(np.float32).tolist(),
        "q01": np.quantile(arr, 0.01, axis=0).astype(np.float32).tolist(),
        "q99": np.quantile(arr, 0.99, axis=0).astype(np.float32).tolist(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sim-root", required=True, help="/hfm/data/simple/G1WholebodyBendPick-v0-psi0")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--video-key", default="observation.rgb_head_stereo_left")
    parser.add_argument("--chunks-size", type=int, default=1000)
    args = parser.parse_args()

    sim_root = Path(args.sim_root).resolve()
    out_dir = Path(args.out_dir).resolve()

    (out_dir / "data").mkdir(parents=True, exist_ok=True)
    (out_dir / "videos").mkdir(parents=True, exist_ok=True)
    (out_dir / "meta").mkdir(parents=True, exist_ok=True)

    sim_info = json.loads((sim_root / "meta" / "info.json").read_text())
    sim_tasks = load_jsonl(sim_root / "meta" / "tasks.jsonl")

    task_by_index = {}
    for t in sim_tasks:
        ti = t.get("task_index", 0)
        task_by_index[int(ti)] = t.get("task", "")

    data_files = sorted((sim_root / "data").glob("chunk-*/episode_*.parquet"))
    total_frames = 0
    episodes = []
    episode_stats_rows = []

    all_states = []
    all_actions = []
    all_timestamp = []
    all_frame_index = []
    all_episode_index = []
    all_index = []
    all_task_index = []
    all_done = []

    for data_path in data_files:
        ep_index = int(data_path.stem.split("_")[-1])
        chunk_id = ep_index // args.chunks_size

        table = pq.read_table(data_path)
        proprio = np.asarray(table["observation.proprio_joint_positions"].to_pylist(), dtype=np.float32)
        cmd = np.asarray(table["observation.amo_policy_command"].to_pylist(), dtype=np.float32)
        action = np.asarray(table["action"].to_pylist(), dtype=np.float32)

        states, actions = build_vectors(proprio, cmd, action)
        hand_joints, arm_joints, leg_joints, prev_height = build_proprio_obs(
            proprio, cmd
        )

        n = states.shape[0]
        done = np.zeros((n,), dtype=bool)
        if n > 0:
            done[-1] = True

        timestamp = np.asarray(table["timestamp"].to_pylist(), dtype=np.float32)
        frame_index = np.asarray(table["frame_index"].to_pylist(), dtype=np.int64)
        episode_index = np.asarray(table["episode_index"].to_pylist(), dtype=np.int64)
        index = np.asarray(table["index"].to_pylist(), dtype=np.int64)
        task_index = np.asarray(table["task_index"].to_pylist(), dtype=np.int64)

        out_table = pa.table({
            "states": states.tolist(),
            "action": actions.tolist(),
            "observation.hand_joints": hand_joints.tolist(),
            "observation.arm_joints": arm_joints.tolist(),
            "observation.leg_joints": leg_joints.tolist(),
            "observation.prev_height": prev_height.tolist(),
            "timestamp": timestamp,
            "frame_index": frame_index,
            "episode_index": episode_index,
            "index": index,
            "task_index": task_index,
            "next.done": done,
        })

        out_data_dir = out_dir / "data" / f"chunk-{chunk_id:03d}"
        out_data_dir.mkdir(parents=True, exist_ok=True)
        pq.write_table(out_table, out_data_dir / f"episode_{ep_index:06d}.parquet")

        src_video = sim_root / "videos" / f"chunk-{chunk_id:03d}" / args.video_key / f"episode_{ep_index:06d}.mp4"
        dst_video_dir = out_dir / "videos" / f"chunk-{chunk_id:03d}" / "egocentric"
        dst_video_dir.mkdir(parents=True, exist_ok=True)
        dst_video = dst_video_dir / f"episode_{ep_index:06d}.mp4"
        if src_video.exists():
            shutil.copyfile(src_video, dst_video)

        total_frames += n
        ep_task = int(task_index[0]) if len(task_index) else 0
        episodes.append({
            "episode_index": ep_index,
            "tasks": [ep_task],
            "length": n,
            "dataset_from_index": total_frames - n,
            "dataset_to_index": total_frames - 1,
            "robot_type": "g1",
            "instruction": task_by_index.get(ep_task, ""),
        })

        ep_stats = {
            "episode_index": ep_index,
            "stats": {
                "action": {**stats_block(actions), "count": [int(n)]},
                "timestamp": {**stats_block(timestamp), "count": [int(n)]},
            },
        }
        episode_stats_rows.append(ep_stats)

        all_states.append(states)
        all_actions.append(actions)
        all_timestamp.append(timestamp)
        all_frame_index.append(frame_index)
        all_episode_index.append(episode_index)
        all_index.append(index)
        all_task_index.append(task_index)
        all_done.append(done.astype(np.float32))

    all_states = np.concatenate(all_states, axis=0) if all_states else np.zeros((0, 32), dtype=np.float32)
    all_actions = np.concatenate(all_actions, axis=0) if all_actions else np.zeros((0, 36), dtype=np.float32)
    all_timestamp = np.concatenate(all_timestamp, axis=0) if all_timestamp else np.zeros((0,), dtype=np.float32)
    all_frame_index = np.concatenate(all_frame_index, axis=0) if all_frame_index else np.zeros((0,), dtype=np.float32)
    all_episode_index = np.concatenate(all_episode_index, axis=0) if all_episode_index else np.zeros((0,), dtype=np.float32)
    all_index = np.concatenate(all_index, axis=0) if all_index else np.zeros((0,), dtype=np.float32)
    all_task_index = np.concatenate(all_task_index, axis=0) if all_task_index else np.zeros((0,), dtype=np.float32)
    all_done = np.concatenate(all_done, axis=0) if all_done else np.zeros((0,), dtype=np.float32)

    task_by_index = {}
    tasks_rows = []
    for t in sim_tasks:
        ti = t.get("task_index", 0)
        task_by_index[int(ti)] = t.get("task", "")
        tasks_rows.append({
            "task_index": int(ti),
            "task": t.get("task", ""),
            "category": "",
            "description": t.get("task", ""),
        })

    meta_dir = out_dir / "meta"
    write_jsonl(meta_dir / "tasks.jsonl", tasks_rows)
    write_jsonl(meta_dir / "episodes.jsonl", sorted(episodes, key=lambda r: r["episode_index"]))
    write_jsonl(meta_dir / "episodes_stats.jsonl", sorted(episode_stats_rows, key=lambda r: r["episode_index"]))

    video_feat = sim_info["features"].get(args.video_key, {
        "dtype": "video",
        "shape": [480, 640, 3],
        "names": ["height", "width", "channel"],
        "info": {"video.fps": 30.0, "video.codec": "h264", "video.pix_fmt": "yuv420p", "video.is_depth_map": False, "has_audio": False},
    })

    info = {
        "codebase_version": "v2.1",
        "robot_type": "g1",
        "total_episodes": len(episodes),
        "total_frames": int(total_frames),
        "total_tasks": len(tasks_rows),
        "total_videos": len(episodes),
        "total_chunks": math.ceil(len(episodes) / args.chunks_size) if args.chunks_size else 1,
        "chunks_size": args.chunks_size,
        "fps": 30,
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/egocentric/episode_{episode_index:06d}.mp4",
        "features": {
            "observation.images.egocentric": {
                "dtype": video_feat.get("dtype", "video"),
                "shape": video_feat.get("shape", [480, 640, 3]),
                "names": ["height", "width", "channel"],
                "video_info": video_feat.get("info", video_feat.get("video_info", {})),
            },
            "observation.hand_joints": {"dtype": "float32", "shape": [14], "names": ["hand_joints"]},
            "observation.arm_joints": {"dtype": "float32", "shape": [14], "names": ["arm_joints"]},
            "observation.leg_joints": {"dtype": "float32", "shape": [15], "names": ["leg_joints"]},
            "observation.prev_height": {"dtype": "float32", "shape": [1], "names": ["prev_height"]},
            "states": {"dtype": "float32", "shape": [-1]},
            "action": {"dtype": "float32", "shape": [-1]},
            "timestamp": {"dtype": "float32", "shape": [1]},
            "frame_index": {"dtype": "int64", "shape": [1]},
            "episode_index": {"dtype": "int64", "shape": [1]},
            "index": {"dtype": "int64", "shape": [1]},
            "next.done": {"dtype": "bool", "shape": [1]},
            "task_index": {"dtype": "int64", "shape": [1]},
        },
    }
    (meta_dir / "info.json").write_text(json.dumps(info, indent=4))

    stats = {
        "states": stats_block(all_states),
        "action": stats_block(all_actions),
        "timestamp": stats_block(all_timestamp),
        "frame_index": stats_block(all_frame_index),
        "episode_index": stats_block(all_episode_index),
        "index": stats_block(all_index),
        "task_index": stats_block(all_task_index),
        "next.done": stats_block(all_done),
    }
    (meta_dir / "stats.json").write_text(json.dumps(stats, indent=4))
    (meta_dir / "stats_psi0.json").write_text(json.dumps(stats, indent=4))
    (meta_dir / "relative_stats.json").write_text("{}")
    (meta_dir / "lang_map.json").write_text("{}")
    (meta_dir / "modality.json").write_text(json.dumps(modality_dict(), indent=2))


if __name__ == "__main__":
    main()
