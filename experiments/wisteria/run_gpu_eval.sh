#!/bin/bash
#PJM -L rscgrp=short-a         
#PJM -L node=1
#PJM -L elapse=00:30:00
#PJM -g  jh210022a             
#PJM -L jobenv=singularity
#PJM -j

source /etc/profile.d/modules.sh
module load singularity/3.7.3
module load cuda/12.0


ROOT=/work/01/jh210022o/q25030     
CODE=$ROOT/graph-vae
IMG=$CODE/images/gvae_cuda.sif     
DATA=$ROOT/datasets                
RUNS=$ROOT/runs                    
WANDB=$ROOT/wandb                  

mkdir -p "$DATA" "$RUNS" "$WANDB"

singularity exec --nv \
  -B "$CODE":/workspace/graph-vae \
  -B "$DATA":/dataset \
  -B "$RUNS":/workspace/runs \
  -B "$WANDB":/workspace/wandb \
  "$IMG" bash -c '
    set -e
    export PYTHONPATH=/workspace/graph-vae:$PYTHONPATH
    export TORCH_GEOMETRIC_HOME=/dataset
    export WANDB_MODE=offline

    ln -sf /workspace/runs/graphvae_ddp_amp.pt \
           /workspace/runs/graphvae_stable.pt

    export QM9_ROOT=/dataset/QM9
    export PYG_DISABLE_DOWNLOAD=1

    # PyG 2.4 用の互換リンク（無ければスキップ）
    [ -f /dataset/QM9/processed/data_v3.pt ] && \
      ln -sf /dataset/QM9/processed/data_v3.pt \
             /dataset/QM9/processed/data_molecule.pt || true

    python -m gvae.eval.eval_stable
  '
