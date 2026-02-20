#!/bin/bash

set -e

source .venv-psi/bin/activate

export CUDA_VISIBLE_DEVICES=0
echo "Serving on GPU $CUDA_VISIBLE_DEVICES"

uv run --active --group psi --group serve serve_psi0 \
    --host 0.0.0.0 \
    --port 22085 \
    --run-dir=.runs/finetune/sim.bent-pick-50hz.rtc.simpl.flow1000.cosin.lr1.0e-04.b128.gpus8.2602181748 \
    --ckpt-step=41999 \
    --action-exec-horizon=24 \
    --rtc
