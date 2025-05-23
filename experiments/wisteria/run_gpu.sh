#!/bin/bash
#PJM -L rscgrp=regular-a
#PJM -L node=1
#PJM -L elapse=00:30:00
#PJM -g jh210022a
#PJM -L jobenv=singularity
#PJM -j

source /etc/profile.d/modules.sh
module load singularity/3.7.3
module load cuda/12.0

ROOT=/work/01/jh210022o/q25030
CODE=$ROOT/graph-vae
IMG=$CODE/images/gvae_cuda.sif

DATA=$ROOT/datasets              # contains QM9
RUNS=$ROOT/runs
WANDB=$ROOT/wandb

mkdir -p "$DATA" "$RUNS" "$WANDB"

singularity exec --nv \
  -B "$CODE":/workspace/graph-vae \
  -B "$DATA":/dataset \
  -B "$RUNS":/workspace/runs \
  -B "$WANDB":/workspace/wandb \
  "$IMG" bash -c '
    export PYTHONPATH=/workspace/graph-vae:$PYTHONPATH
    export TORCH_GEOMETRIC_HOME=/dataset
    export WANDB_MODE=offline
    python -m gvae.train.cli_train \
           --config /workspace/graph-vae/experiments/configs/qm9_gpu1.yaml
  '
