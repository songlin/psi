#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="/hfm/zhenyu/psi/src:/hfm/zhenyu/psi/src/gr00t${PYTHONPATH:+:$PYTHONPATH}"
VENV_ACTIVATE="/hfm/zhenyu/psi/src/gr00t/.venv/bin/activate"
export TORCHINDUCTOR_DISABLE=1
export TORCH_COMPILE=0
MODEL_PATH="/hfm/zhenyu/psi/src/gr00t/checkpoints/G1WholebodyBendPick-v0-psi0-real/checkpoint-90"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model-path)
      MODEL_PATH="$2"
      shift 2
      ;;
    *)
      break
      ;;
  esac
done

if [[ ! -d "$MODEL_PATH" ]]; then
  echo "Error: model path not found: $MODEL_PATH" >&2
  exit 1
fi

PROCESSOR_DIR="/hfm/zhenyu/psi/src/gr00t/checkpoints/G1WholebodyBendPick-v0-psi0-real/processor"
if [[ -d "$PROCESSOR_DIR" ]]; then
  for f in processor_config.json statistics.json embodiment_id.json; do
    if [[ -f "$PROCESSOR_DIR/$f" ]] && [[ ! -f "$MODEL_PATH/$f" ]]; then
      cp "$PROCESSOR_DIR/$f" "$MODEL_PATH/$f"
    fi
  done
fi

cmd="source \"$VENV_ACTIVATE\" && python -m gr00t.deploy.gr00t_serve_simple \
  --embodiment-tag NEW_EMBODIMENT \
  --model-path \"$MODEL_PATH\" \
  --device cuda:0 \
  --host 0.0.0.0 \
  --port 5555 \
  --use-sim-policy-wrapper \
  --strict"

bash -c "$cmd"
