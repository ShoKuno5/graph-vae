#!/bin/bash
#PJM -L rscgrp=regular-a
#PJM -L node=1
#PJM -L elapse=02:00:00
#PJM -L jobenv=singularity
#PJM -g jh210022a
#PJM -j

module load cuda/12.0 singularity/3.7.3 

ROOT=/work/01/jh210022o/q25030
IMG=$ROOT/graph-vae/images/gvae_cuda.sif
CODE=$ROOT/graph-vae
DATA=$ROOT/data                 # QM9 データを置く場所
RUNS=$ROOT/runs                 # TensorBoard ログ
WANDB=$ROOT/wandb               # wandb (offline の場合)

mkdir -p "$DATA" "$RUNS" "$WANDB"

singularity exec --nv --userns \
  -B "$CODE":/workspace/graph-vae \
  -B "$DATA":/dataset \
  -B "$RUNS":/workspace/runs \
  "$IMG" \
  bash -c 'export PYTHONPATH=/workspace/graph-vae:$PYTHONPATH
           python -m gvae.train.train_graphvae \
             --epochs 1 \
             --n_graph 32 \
             --data_root /dataset/QM9 \
             --device cuda'