#!/bin/bash

set -e

source .venv-psi/bin/activate

export OMP_NUM_THREADS=8
export TF_CPP_MIN_LOG_LEVEL=3
export CUDA_VISIBLE_DEVICES="0,1"

NNODES=1
NPROC_PER_NODE=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)

args="data:egodex \
data.transform.repack:egodex \
data.transform.action-state:maxmin \
data.transform.model:hfm \
model:hfm-qwen3vl \
model.action-tokenizer:fast \
--debug \
--seed=7 \
--exp=pre \
--timestamp=$(date +"%y%m%d%H%M") \
--train.name=hfm \
--train.data_parallel=deepspeed \
--train.deepspeed_config=scripts/deepspeed/zero3.json \
--train.mixed_precision=bf16 \
--train.train_batch_size=16 \
--train.resume_from_checkpoint=latest \
--train.max_checkpoints_to_keep=5 \
--train.gradient_accumulation_steps=1 \
--train.learning_rate=1e-4 \
--train.max_training_steps=1000000 \
--train.warmup_ratio=None \
--train.warmup_steps=0 \
--train.checkpointing_steps=100 \
--train.validation_steps=0 \
--train.max_grad_norm=1.0 \
--train.lr_scheduler_type=constant \
--train.lr_scheduler_kwargs.betas 0.9 0.999 \
--train.lr_scheduler_kwargs.weight_decay=0.0 \
--train.lr_scheduler_kwargs.eps=1e-8 \
--log.report_to=wandb \
--data.chunk_size=1 \
--data.upsample_rate=3 \
--data.transform.action-state.action_norm_type=bounds_q99 \
--data.transform.action-state.stat-action-key=egodex \
--data.transform.action-state.stat-path=assets/stats/egodex_stat_all.json \
--data.transform.model.resize.size 270 480 \
--data.root-dir=/hfm/data/egodex \
--data.use-delta-actions \
--data.transform.model.no-img-aug \
--model.action_tokenizer.bins=2048 \
--model.action_tokenizer.pretrained_checkpoint=src/fast/egodex-rel-50w-1x48-v2048-s100 \
--model.tune-mm-llm \
--model.tune-mm-vision \
--model.tune-mm-mlp \
--model.mm_projector_lr=1e-5 \
--model.vision_tower_lr=1e-5
"

# Find an available TCP port starting at 29500 and increment until a free port is found.
find_free_port() {
    start_port=${1:-29500}
    port=${start_port}
    while true; do
        # Use Python socket bind test; binding to 0.0.0.0:port will fail if port is in use.
        CHECK_PORT=${port} python - <<'PY'
import os,sys,socket
port = int(os.environ.get('CHECK_PORT','0'))
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
try:
    sock.bind(('0.0.0.0', port))
    sock.close()
    sys.exit(0)
except OSError:
    sys.exit(1)
PY
        if [ $? -eq 0 ]; then
            echo ${port}
            return 0
        fi
        port=$((port+1))
        # avoid infinite loop in pathological cases
        if [ ${port} -gt $((start_port+1000)) ]; then
            echo "Failed to find free port after 1000 attempts" >&2
            return 1
        fi
    done
}

MAIN_PORT=$(find_free_port 29500)
if [ -z "${MAIN_PORT}" ]; then
    echo "Could not find free main process port, aborting." >&2
    exit 1
fi

torchrun --nproc_per_node=$NPROC_PER_NODE --master_port=${MAIN_PORT} scripts/train.py \
    ${args}