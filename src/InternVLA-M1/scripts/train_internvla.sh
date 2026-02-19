export CUDA_VISIBLE_DEVICES=5
export TORCH_HOME=/hfm/boqian/torch_cache
export HF_HOME=/hfm/boqian/torch_cache/huggingface
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=1
# export DS_BUILD_OPS=0
# export DS_SKIP_CUDA_CHECK=1

source .venv/bin/activate

nprocs=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)

ulimit -n 65535
echo "Training with $nprocs GPUs"


accelerate launch \
    --config_file InternVLA/config/deepseeds/deepspeed_zero2.yaml  \
    --main_process_port 0 \
    --num_processes 1 InternVLA/training/train_internvla.py \
    --config_yaml InternVLA/config/training/internvla_cotrain_humanoid.yaml \
    --wandb_entity boqianli \
    --wandb_project hfm c\
    --datasets.vla_data.data_root_dir /hfm/data/real_teleop_g1/lerobot/Push_cart_grasp_and_place_grapes_on_plate \
    --run_id Push_cart_grasp_and_place_grapes_on_plate
