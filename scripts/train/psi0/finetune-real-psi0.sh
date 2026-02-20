#!/bin/bash

export OMP_NUM_THREADS=32
export CUDA_VISIBLE_DEVICES=0 #,1,2,3,4,5,6,7

source .venv-psi/bin/activate

NPROC_PER_NODE=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
ulimit -n 65535
echo "Training with $NPROC_PER_NODE GPUs"

args="
    finetune_real_psi0_config \
    --seed=292285 \
    --no-auto-tag-run \
    --exp=real \
    --train.name=finetune \
    --train.data_parallel=ddp \
    --train.mixed_precision=bf16 \
    --train.train_batch_size=32 \
    --train.max_checkpoints_to_keep=5 \
    --train.gradient_accumulation_steps=1 \
    --train.learning_rate=1e-4 \
    --train.max_training_steps=40000 \
    --train.warmup_ratio=None \
    --train.warmup_steps=1000 \
    --train.checkpointing_steps=5000 \
    --train.validation_steps=500 \
    --train.val_num_batches=20 \
    --train.max_grad_norm=1.0 \
    --train.lr_scheduler_type=cosine \
    --train.lr_scheduler_kwargs.weight_decay=1e-6 \
    --train.lr_scheduler_kwargs.betas 0.95 0.999 \
    --log.report_to=wandb \
    --data.root_dir=/hfm/data/real_teleop_g1/lerobot/ \
    --data.train_repo_ids=Pick_toys_into_box_and_lift_and_turn_and_put_on_the_chair_new_target_yaw \
    --data.transform.repack.pad-action-dim=36 \
    --data.transform.repack.pad-state-dim=36 \
    --data.transform.field.stat-path=/hfm/songlin/we_learn/src/we/assets/dataset_statistics/g1-stats-put-toys-box-lift-put-chair.json \
    --data.transform.field.stat-action-key=action \
    --data.transform.field.stat-state-key=states \
    --data.transform.field.action_norm_type=bounds \
    --data.transform.field.no-use-norm-mask \
    --data.transform.field.normalize-state \
    --data.transform.field.pad-action-dim=36 \
    --data.transform.field.pad-state-dim=36 \
    --data.transform.model.img-aug \
    --data.transform.model.resize.size 240 320 \
    --data.transform.model.center_crop.size 240 320 \
    --model.model_name_or_path=/hfm/cache/checkpoints/hfm.pre.fast.egodex.2512241941.ckpt200k \
    --model.pretrained-action-header-path=/hfm/cache/checkpoints/postpre.1by130k.pad36.mixed.2601131206.ckpt34k \
    --model.noise-scheduler=flow \
    --model.train-diffusion-steps=1000 \
    --model.n_conditions=0 \
    --model.action-chunk-size=30 \
    --model.action-dim=36 \
    --model.action-exec-horizon=30 \
    --model.observation-horizon=1 \
    --model.odim=36 \
    --model.view_feature_dim=2048 \
    --model.no-tune-vlm \
    --model.no-use_film \
    --model.no-combined_temb \
    --model.rtc \
    --model.max-delay=8
"

torchrun --nproc_per_node=$NPROC_PER_NODE --master_port=29500 scripts/train.py \
    ${args}

