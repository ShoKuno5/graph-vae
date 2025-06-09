#!/bin/bash
#PJM -L rscgrp=short-a
#PJM -L node=1
#PJM -L elapse=00:30:00
#PJM -g  jh210022a
#PJM -L jobenv=singularity
#PJM -j

set -e
source /etc/profile.d/modules.sh
module load singularity/3.7.3
module load cuda/12.0

# -------- host-side paths --------
ROOT=/work/01/jh210022o/q25030
CODE=$ROOT/graph-vae
IMG=$CODE/images/gvae_cuda.sif

DATA=$ROOT/datasets
RUNS=$CODE/runs               # ★ repo 内 runs/ に統一
WANDB=$CODE/wandb

mkdir -p "$DATA" "$RUNS" "$WANDB"

# -------- pick experiment dir --------
# 1) pjsub --sparam "EXP=20250601_213145" で明示指定
# 2) 未指定なら runs/ 内で最新タイムスタンプ dir を自動選択
if [[ -z "$EXP" ]]; then
  EXP=$(basename "$(ls -1td "$RUNS"/*/ | head -n1)")
fi
EXP_DIR=$RUNS/$EXP

if [[ ! -f "$EXP_DIR/graphvae_ddp_amp.pt" ]]; then
  echo "[ERROR] checkpoint not found: $EXP_DIR/graphvae_ddp_amp.pt"
  exit 1
fi

# -------- singularity exec --------
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

    export QM9_ROOT=/dataset/QM9
    export PYG_DISABLE_DOWNLOAD=1

    # PyG 2.4 互換シンボリックリンク（存在すれば）
    [ -f /dataset/QM9/processed/data_v3.pt ] && \
      ln -sf /dataset/QM9/processed/data_v3.pt \
             /dataset/QM9/processed/data_molecule.pt || true

    python -m gvae.eval.eval_stable \
           --ckpt /workspace/runs/'"$EXP"'/graphvae_ddp_amp.pt \
           --out  /workspace/runs/'"$EXP"'/eval.txt
  '
