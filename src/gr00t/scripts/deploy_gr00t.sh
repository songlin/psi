#!/usr/bin/env bash
set -euo pipefail

python gr00t/eval/run_gr00t_server.py \
  --embodiment-tag NEW_EMBODIMENT \
  --model-path ./checkpoints/Pick_toys_into_box_and_lift_and_turn_and_put_on_the_chair_new_target_yaw/checkpoint-10/ \
  --device cuda:0 \
  --host 0.0.0.0 \
  --port 5555 \
  --strict
