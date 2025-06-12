#!/usr/bin/env python
"""
Sample graphs from a trained GraphVAE checkpoint.

Usage
-----
python experiments/sample.py RUN_DIR \
       [--num 8] [--th 0.5] [--cpu] [--seed 0]
"""

from __future__ import annotations
import argparse, yaml, torch, networkx as nx
from pathlib import Path
import matplotlib.pyplot as plt
import math

# ----------------------------------------------------------------------
# import your package
# ----------------------------------------------------------------------
from graphvae.models.model  import GraphVAE, vec_to_adj
from graphvae.eval.metrics import degree_histogram, clustering_coef        # optional

# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------
def load_model(run_dir: Path, device: str = "cpu") -> tuple[GraphVAE, dict]:
    ckpt = run_dir / "graphvae_ddp.pt"
    cfg  = yaml.safe_load(open(run_dir / "params.yaml"))

    # ── fall-back: 古い YAML に in_dim / max_nodes が無い場合は ckpt から推定 ──
    state = torch.load(ckpt, map_location="cpu")
    in_dim     = cfg.get("in_dim")     or state["conv1.lin.weight"].shape[1]
    #max_nodes  = cfg.get("max_nodes")  or state["dec_lin.weight"].size(0)  # (=U)≒N(N-1)/2 → 近似推定でも十分
    
    off_diag   = cfg.get("off_diag")   or state["dec_edge.2.weight"].size(0)
    # U = N(N−1)/2  →  N = (1+√(1+8U))/2
    max_nodes  = cfg.get("max_nodes")  or int((1 + math.isqrt(1 + 8*off_diag)) // 2)

    model = GraphVAE(
        in_dim    = in_dim,
        hid_dim   = cfg["hid_dim"],
        z_dim     = cfg["z_dim"],
        max_nodes = max_nodes,
        pool      = cfg.get("pool", "sum"),
    ).to(device)

    model.load_state_dict(state, strict=True)
    model.eval()
    return model, cfg

def save_graph(G: nx.Graph, out: Path):
    nx.write_gml(G,     out.with_suffix(".gml"))
    plt.figure(figsize=(3, 3))
    nx.draw(G, node_size=60, width=0.8)
    plt.tight_layout(); plt.savefig(out.with_suffix(".png"), dpi=150); plt.close()

# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------
def main(run_dir: Path, num: int, th: float, device: str, seed: int):
    torch.manual_seed(seed)

    model, cfg = load_model(run_dir, device)
    out_dir    = run_dir / "samples"; out_dir.mkdir(exist_ok=True)

    with torch.no_grad():
        z   = torch.randn(num, cfg["z_dim"], device=device)
        #vec = model.dec(z)                    # (B, U)
        vec_edge  = model.dec_edge(z)         # (B, U_off)
        vec_node  = model.dec_node(z)         # (B, N)        

    for i in range(num):
        # training と同じく自己ループの logit を極端に小さくする
        #adj_logits = vec_to_adj(vec[i], model.max_nodes, diag=False)
        #adj_logits.fill_diagonal_(-10.0)
        
        adj_logits = vec_to_adj(vec_edge[i], model.max_nodes, diag=False)
        #adj_logits.fill_diagonal_(vec_node[i])   # ← ノード対角を注入
        
        diag_idx = torch.arange(model.max_nodes, device=adj_logits.device)
        adj_logits[diag_idx, diag_idx] = vec_node[i]          # ← ベクトルを書き込む
        
        adj_prob = adj_logits.sigmoid()
        # ---------- quick dummy diagnostic -----------------
        deg     = adj_prob.sum(1)                # 各ノードの総確率
        R_int   = int((deg > 0.05).sum())        # “実ノード” 推定 (閾値は適宜)
        mask_dummy = torch.ones_like(adj_prob, dtype=torch.bool)
        mask_dummy[:R_int, :R_int] = False

        p_dummy = adj_prob[mask_dummy].mean().item()
        p_real  = adj_prob[:R_int, :R_int].triu(1).mean().item()
        print(f"[{i}]  P(edge|dummy)={p_dummy:.3f}  P(edge|real-real)={p_real:.3f}")
        # ---------------------------------------------------
        
        #adj_bin = (adj_prob > th).float()
        
        # “実ノード” を推定して dummy 部分だけ 0 にしても OK
        adj_bin = (adj_prob > th).float()        

        G = nx.from_numpy_array(adj_bin.cpu().numpy())
        save_graph(G, out_dir / f"sample_{i}")

        # quick metrics (optional)
        deg_hist = degree_histogram(G)
        C_avg    = clustering_coef(G)
        print(f"[{i}] nodes={G.number_of_nodes():3d} edges={G.number_of_edges():3d} "
              f"C={C_avg:.3f} max_deg={max(deg_hist)}")

    print(f"✔ {num} graphs saved to {out_dir}")

# ----------------------------------------------------------------------
if __name__ == "__main__":
    pa = argparse.ArgumentParser()
    pa.add_argument("run_dir", type=Path, help="runs/YYYYMMDD_HHMMSS")
    pa.add_argument("--num",  type=int,   default=4,   help="#samples to draw")
    pa.add_argument("--th",   type=float, default=0.5, help="edge-probability threshold")
    pa.add_argument("--cpu",  action="store_true",     help="force CPU")
    pa.add_argument("--seed", type=int,   default=0)
    args = pa.parse_args()

    device = "cpu" if args.cpu or not torch.cuda.is_available() else "cuda:0"
    main(args.run_dir, args.num, args.th, device, args.seed)

"""
  ### sample
pjsub --interact -g jh210022a -L rscgrp=interactive-a,jobenv=singularity

module load singularity/3.7.3 cuda/12.0 

ROOT=/work/jh210022o/q25030 \
CODE=$ROOT/graph-vae \
IMG=$CODE/images/gvae_cuda.sif \
DATA=$ROOT/datasets \
RUNS=$CODE/runs  

export PYTHONPATH=/workspace/graph-vae:$PYTHONPATH

  ### GPU で実行
singularity exec --nv \
  -B "$CODE":/workspace/graph-vae \
  -B "$DATA":/dataset \
  -B "$RUNS":/workspace/runs \
  "$IMG" \
  bash -c "
    python /workspace/graph-vae/experiments/sample.py /workspace/runs/20250612_171747 --num 8 --th 0.5
  "
"""