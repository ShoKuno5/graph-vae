#!/bin/bash
#PJM -L rscgrp=short-a
#PJM -L node=1
#PJM -L elapse=02:00:00
#PJM -g jh210022a
#PJM -L jobenv=singularity
#PJM -j

source /etc/profile.d/modules.sh
module load singularity/3.7.3
module load cuda/12.0

# -------- host-side paths --------
ROOT=/work/01/jh210022o/q25030
CODE=$ROOT/graph-vae
IMG=$CODE/images/gvae_cuda.sif

DATA=$ROOT/datasets
RUNS=$CODE/runs                     
WANDB=$CODE/wandb

# -------- experiment tag ---------
EXP=$(date +%Y%m%d_%H%M%S)          # 例: 20250601_214203
EXP_DIR=$RUNS/$EXP                  # /work/.../graph-vae/runs/20250601_...

mkdir -p "$DATA" "$EXP_DIR" "$WANDB"

# -------- env / NCCL --------
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=ib0,eth0
export GLOO_SOCKET_IFNAME=ib0,eth0
export OMP_NUM_THREADS=8

# -------- singularity + torchrun --------
singularity exec --nv \
  -B "$CODE":/workspace/graph-vae \
  -B "$DATA":/dataset \
  -B "$RUNS":/workspace/runs \
  -B "$WANDB":/workspace/wandb \
  "$IMG" bash -c '
    export PYTHONPATH=/workspace/graph-vae:$PYTHONPATH
    export TORCH_GEOMETRIC_HOME=/dataset
    export WANDB_MODE=offline

    torchrun --nproc_per_node 8 \
             --master_addr ${MASTER_ADDR} --master_port ${MASTER_PORT} \
             -m gvae.train.train_graphvae_ddp \
             --config /workspace/graph-vae/experiments/configs/qm9_ddp.yaml \
             --log_dir /workspace/runs/'"$EXP"'
  '
