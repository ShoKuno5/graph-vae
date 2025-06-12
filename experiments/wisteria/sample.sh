#!/bin/bash
#PJM -L rscgrp=regular-a
#PJM -L node=1
#PJM -L elapse=02:00:00
#PJM -g jh210022a
#PJM -L jobenv=singularity
#PJM -j

source /etc/profile.d/modules.sh
module load singularity/3.7.3
module load cuda/12.0

# -------- host-side paths --------
ROOT=/work/jh210022o/q25030
CODE=$ROOT/graph-vae
IMG=$CODE/images/gvae_cuda.sif
DATA=$CODE/datasets
RUNS=$CODE/runs


# -------- experiment tag ---------
#EXP=$(date +%Y%m%d_%H%M%S)          # ex.) 20250607_231045
#EXP_DIR=$RUNS/$EXP
#mkdir -p "$DATA" "$EXP_DIR"
#echo "Directory created: $EXP_DIR $DATA"

# -------- env / NCCL / PyTorch --------
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=ib0,eth0
export GLOO_SOCKET_IFNAME=ib0,eth0
export OMP_NUM_THREADS=8
export WANDB_MODE=online
export WANDB_API_KEY=fb39ca5f5835abaa4c40a8b61dde2a499b45fcba
export WANDB_PROJECT=graphvae
export WANDB_NAME="enzymes_${EXP}"
export TORCH_GEOMETRIC_HOME=/dataset  # TUDataset キャッシュ先

# -------- singularity + torchrun --------
singularity exec --nv \
  -B "$CODE":/w \
  -B "$RUNS":/workspace/runs \
  -B "$DATA":/dataset \
  "$IMG" \
  bash -c "
    export PYTHONPATH=/w:\$PYTHONPATH;
    export MASTER_ADDR=127.0.0.1;
    export MASTER_PORT=29500;
    export PYTHONUNBUFFERED=1;
    # Create experiment directory
    mkdir -p /workspace/runs/$EXP;
    
    # Run sampling (adjusted path from /workspace/graph-vae to /w)
    python /w/experiments/sample.py /workspace/runs/20250612_205911 --num 8 --th_edge 0.6 --th_node 0.5
  "
