#!/usr/bin/env bash
set -euo pipefail

DATASET_PATH_DEFAULT="/hfm/data/real_teleop_g1/lerobot/Pick_toys_into_box_and_lift_and_turn_and_put_on_the_chair_new_target_yaw"
CUDA_VISIBLE_DEVICES_DEFAULT="5,6,7"
NPROC_PER_NODE_DEFAULT=3
MASTER_PORT_DEFAULT=29501
OUTPUT_DIR_DEFAULT=""

DATASET_PATH="$DATASET_PATH_DEFAULT"
CUDA_VISIBLE_DEVICES_VALUE="$CUDA_VISIBLE_DEVICES_DEFAULT"
NPROC_PER_NODE="$NPROC_PER_NODE_DEFAULT"
MASTER_PORT="$MASTER_PORT_DEFAULT"
OUTPUT_DIR="$OUTPUT_DIR_DEFAULT"

usage() {
  cat <<'USAGE'
Usage: train_gr00t.sh [options]
  --dataset-path PATH         Dataset path (default: built-in)
  --output-dir PATH           Output directory (default: ./checkpoints/<dataset-basename>)
  --cuda-visible-devices LIST CUDA_VISIBLE_DEVICES (default: 5,6,7)
  --nproc-per-node N          torchrun --nproc_per_node (default: 3)
  --master-port PORT          torchrun --master_port (default: 29501)
  -h, --help                  Show this help
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset-path)
      DATASET_PATH="$2"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --cuda-visible-devices)
      CUDA_VISIBLE_DEVICES_VALUE="$2"
      shift 2
      ;;
    --nproc-per-node)
      NPROC_PER_NODE="$2"
      shift 2
      ;;
    --master-port)
      MASTER_PORT="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ -z "$OUTPUT_DIR" ]]; then
  OUTPUT_DIR="./checkpoints/$(basename "$DATASET_PATH")"
fi

CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES_VALUE" torchrun --nproc_per_node="$NPROC_PER_NODE" --master_port="$MASTER_PORT" \
  gr00t/experiment/launch_finetune.py \
  --base-model-path nvidia/GR00T-N1.6-3B \
  --dataset-path "$DATASET_PATH" \
  --embodiment-tag NEW_EMBODIMENT \
  --modality-config-path gr00t/configs/modality/new_embodiment_task_description.py \
  --num-gpus 3 \
  --output-dir "$OUTPUT_DIR" \
  --save-steps 10 \
  --save-total-limit 2 \
  --max-steps 20000 \
  --warmup-ratio 0.05 \
  --weight-decay 1e-5 \
  --learning-rate 1e-4 \
  --global-batch-size 24 \
  --color-jitter-params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08 \
  --dataloader-num-workers 4
