#!/usr/bin/env bash
set -euo pipefail

DATASET_PATH="/hfm/data/real_teleop_g1/lerobot/Pick_toys_into_box_and_lift_and_turn_and_put_on_the_chair_new_target_yaw"

CUDA_VISIBLE_DEVICES=5,6,7 torchrun --nproc_per_node=3 --master_port=29501 \
  gr00t/experiment/launch_finetune.py \
  --base-model-path nvidia/GR00T-N1.6-3B \
  --dataset-path "$DATASET_PATH" \
  --embodiment-tag NEW_EMBODIMENT \
  --modality-config-path gr00t/configs/modality/new_embodiment_task_description.py \
  --num-gpus 3 \
  --output-dir "./checkpoints/$(basename "$DATASET_PATH")" \
  --save-steps 10 \
  --save-total-limit 2 \
  --max-steps 20000 \
  --warmup-ratio 0.05 \
  --weight-decay 1e-5 \
  --learning-rate 1e-4 \
  --global-batch-size 24 \
  --color-jitter-params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08 \
  --dataloader-num-workers 4
