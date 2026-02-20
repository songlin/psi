# $\Psi_0$: An Open Foundation Model <br/> Towards Universal Humanoid Loco-Manipulation

## Installation

Clone the project:
```bash
git clone git@github.com:songlin/psi.git
cd psi
```

Install [uv](https://docs.astral.sh/uv/getting-started/installation/).
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Install Psi0

```
uv venv .venv-psi --python 3.10
source .venv-psi/bin/activate
GIT_LFS_SKIP_SMUDGE=1  uv sync --all-groups --index-strategy unsafe-best-match  --frozen --active
```

Test installation:
```bash
source .venv-psi/bin/activate
python -c "import psi;print(psi.__version__)"
```
a version number will be displayed.

## Fine-Tuning

override to fix a lerobot bug
```
cp src/lerobot_patch/common/datasets/lerobot_dataset.py \
  .venv-psi/lib/python3.10/site-packages/lerobot/common/datasets/lerobot_dataset.py
```
### Simulation

download data
```
hf download songlinwei/psi-data simple/G1WholebodyBendPick-v0-psi0.zip 
  --local-dir=/hfm/data/simple --repo-type=dataset
```

start training
```
bash scripts/train/psi0/finetune-simple-psi0-rtc.sh
```

serve
```
uv run --active --group psi --group serve serve_psi0 \
  --host 0.0.0.0 \
  --port 22085 \
  --run-dir=.runs/.runs/hfm-finetune/sim.bent-pick.simpl.flow1000.cosin.lr1.0e-04.b128.gpus4.2602060529 \
  --ckpt-step=41999
```

### Real-Robot

### Data Preprocessing
### Training
### Serve


## Pre-Training

predownload the `Qwen/Qwen3-VL-2B-Instruct` weights
```
scripts/predownload_qwen3vl.py
```

pretrain on egodex
```
bash scripts/train/psi0/pretrain-egodex-psi0-fast.sh 
```

pretrain on humanoid everyday
```
bash scripts/train/psi0/pretrain-he-psi0-fast.sh 
```

## Post-Training



## Setup environment variables

Refers to `.env.sample` 
```
cp .env.sample .env
```
set `PSI_HOME` to a folder where cache/data/checkpoints are located (by convention).

## visualize episode
```
# manually install pinocchio due to numpy version conflicts
uv pip install "pin>=3.8.0"

# revert numpy version
uv pip install numpy==1.26.4
```

## Install ffmpeg for torchdec

We should be relying on video decoding a lot because we are learning from video. See official docs [TorchDec](https://github.com/meta-pytorch/torchcodec?tab=readme-ov-file#installing-torchcodec)

1. install latest `ffmpeg`
```
sudo apt-get install ffmpeg
```

2. make sure it has `NVDEC` support
```
ffmpeg -decoders | grep -i nvidia
# This should show a line like this:
# V..... h264_cuvid           Nvidia CUVID H264 decoder (codec h264)
```
3. give it a test by decoding a sample mp4
```
ffmpeg -hwaccel cuda -hwaccel_output_format cuda -i /hfm/data/egodex/test/add_remove_lid/0.mp4 -f null -  
```


## Download pretrained models (Be selective)

Download psi-0 VLM (egodex)
```
python scripts/data/download.py \
  --repo-id=songlinwei/hfm-models \
  --remote-dir=pre.fast.egodex.2512241941/pretrained/ckpt_200000 \
  --local-dir=/hfm/cache/checkpoints/hfm.pre.fast.egodex.2512241941.ckpt200k \
  --repo-type=model
```

Download psi-0 VLM (mixed pre-training)
```
python scripts/data/download.py \
  --repo-id=songlinwei/hfm-models \
  --remote-dir=pre.fast.mixed.1by1.2601091803/pretrained/ckpt_30000 \
  --local-dir=hfm.pre.fast.mixed.1by1.2601091803.ckpt30k \
  --repo-type=model
```

Download psi-0 action-header
```
python scripts/data/download.py \
  --repo-id=songlinwei/hfm-models \
  --remote-dir=postpre.1by130k.pad36.mixed.2601131206/pretrained/ckpt_34000 \
  --local-dir=postpre.1by130k.pad36.mixed.2601131206.ckpt34k \
  --repo-type=model
```

## train
modify 
```
./scripts/train.sh
```


## Train Psi0
```
uv venv .venv-psi --python 3.10
source .venv-psi/bin/activate
cp envs/uv.lock.psi uv.lock
GIT_LFS_SKIP_SMUDGE=1 uv sync --group psi --group serve --group viz --active --frozen
cp src/lerobot_patch/common/datasets/lerobot_dataset.py \
  .venv-psi/lib/python3.10/site-packages/lerobot/common/datasets/lerobot_dataset.py
```

open-loop evaluation (offline)
```
python scripts/train/hfm/psi0_inference_simple.py
```

open-loop evaluation (online)
running serve 
```
uv run --active --group psi --group serve serve_psi0 \
    --host 0.0.0.0 \
    --port 22085 \
    --run-dir=.runs/.runs/hfm-finetune/sim.bent-pick.simpl.flow1000.cosin.lr1.0e-04.b128.gpus4.2602060529 \
    --ckpt-step=41999
```

check if server is running
```
curl -i http://localhost:22085/health
```

launch client
```
python scripts/client/psi0_simple.py
```

## Download SIMPLE data
```
hf download \
  SIMPLE-org/g1-tabletop-grasp \
  G1WholebodyBendPick-Processed.zip \
  --local-dir=/hfm/data/simple/learn \
  --repo-type=dataset
```

## Pretrain on egodex
submit a slurm job
```
scripts/train/hfm/nv_slurm.sh
```
save the pretrained checkpoints
```
python scripts/train/hfm/hfm_save_pretrain.py
```
upload to huggingface
```
hf upload songlinwei/hfm-models \
  .runs/hfm/pre.fast.egodex.delta.c1.const.lr1.0e-04.b1024.gpus64.2512241941/pretrained/ckpt_200000 \
  pre.fast.egodex.2512241941/pretrained/ckpt_200000 \
  --repo-type model
```
download pretrained hfm
```
python scripts/data/download.py \
  --repo-id=songlinwei/hfm-models \
  --remote-dir=pre.fast.egodex.2512241941/pretrained/ckpt_200000 \
  --local-dir=/hfm/cache/checkpoints/hfm.pre.fast.egodex.2512241941.ckpt200k \
  --repo-type=model
```

## Train Other Policies
### GR00T
Install the env 
```bash
cd src/gr00t; uv sync
```
1. training
```bash
cd src/gr00t
./scripts/train_gr00t.sh --dataset-path /your/lerobot/dataset
```
2. serving a checkpoint
```bash
cd src/gr00t
./scripts/deploy_gr00t.sh
```

3. openloop eval on trained checkpoint using gt
```bash
cd src/gr00t
./scripts/openloop_eval.sh
```


### InternVLA-M1
Install the env 
```bash
cd src/InternVLA-M1; uv sync --python 3.10
```
1. training
```bash
cd src/InternVLA-M1; 
bash scripts/train_internvla.sh
```


## Benchmarking of SIMPLE
1. download sim data
```
hf download \
  SIMPLE-org/g1-tabletop-grasp \
  G1WholebodyBendPick-Processed.zip \
  --local-dir=/hfm/data/simple/learn \
  --repo-type=dataset
```
and move the structure as follows:

```
├── G1WholebodyBendPick-Processed
│   └── level-0
│       ├── data
│       ├── images
│       ├── meta
│       └── videos
```

2. calculate stats
```
python scripts/data/calc_modality_stats.py \
  --task-dir=/hfm/data/simple/G1WholebodyBendPick-Processed/level-0
```
and convert the stats to psi0 format
```
python scripts/data/simple.py \
  --task-dir=/hfm/data/simple/G1WholebodyBendPick-Processed/level-0
```
3. start training
```
scripts/train/hfm/hfm_finetune_simple.sh
```

## Troubleshootings

1. Lerobot dataset issues: `stack(): argument 'tensors' (position 1) must be tuple of Tensors, not Column`

> change the path of `.env` if needed

```
cp src/lerobot_patch/common/datasets/lerobot_dataset.py \
  .venv/lib/python3.10/site-packages/lerobot/common/datasets/lerobot_dataset.py
```

2. Fail to install `evdev`, `src/evdev/input.c:10:10: fatal error: Python.h: No such file or directory`

```
sudo apt update
sudo apt install -y python3-dev python3-venv build-essential \
    linux-headers-$(uname -r)
```

3. RuntimeError: Could not load libtorchcodec. Likely causes ...
```
sudo apt-get install ffmpeg
```

4. ImportError: cannot import name 'Deprecated' from 'wandb.proto.wandb_telemetry_pb2' 

re-install `wandb`
```
source .venv-pusht/bin/activate
uv pip uninstall wandb
uv pip install wandb==0.18.0
```

5. support `sm_120` on newer GPUs like `5090` or `RTX 6000`, UserWarning: Ignoring invalid value for boolean flag CUDA_LAUNCH_BLOCKING: truevalid values are 0 or 1.

update `torch` and `flash-attn`
```
uv pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128
uv pip install flash-attn --no-build-isolation
```

6. Failed to download and build `lerobot ... `, Use `git lfs logs last` to view the log.

```
GIT_LFS_SKIP_SMUDGE=1 uv ...
```
