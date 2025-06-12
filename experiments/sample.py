#!/usr/bin/env python
"""
Sample graphs *without* dummy-node artefacts or self-loops.

Usage
-----
python experiments/sample.py RUN_DIR --num 8 --th_edge 0.5 --th_node 0.5
"""

from __future__ import annotations
import argparse, yaml, torch, networkx as nx, math
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from graphvae.models.model import GraphVAE, vec_to_adj      # model w/ node-head :contentReference[oaicite:1]{index=1}
from graphvae.eval.metrics   import degree_histogram, clustering_coef  # optional

# ----------------------------------------------------------------------
def _load_model(run_dir: Path, device="cpu") -> tuple[GraphVAE, dict]:
    ckpt = run_dir / "graphvae_ddp.pt"
    cfg  = yaml.safe_load(open(run_dir / "params.yaml"))
    state = torch.load(ckpt, map_location="cpu")

    # --- Fallbacks ------------------------------------------------------
    in_dim    = cfg.get("in_dim")    or state["conv1.lin.weight"].shape[1]
    off_diag  = cfg.get("off_diag")  or state["dec_edge.2.weight"].size(0)
    max_nodes = cfg.get("max_nodes") or int((1 + math.isqrt(1 + 8*off_diag)) // 2)

    model = GraphVAE(in_dim, cfg["hid_dim"], cfg["z_dim"],
                     max_nodes, pool=cfg.get("pool","sum")).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, cfg

def _save_graph(g: nx.Graph, out: Path):
    nx.write_gml(g, out.with_suffix(".gml"))
    plt.figure(figsize=(3,3))
    nx.draw(g, node_size=60, width=0.8)
    plt.tight_layout(); plt.savefig(out.with_suffix(".png"), dpi=150); plt.close()

# ----------------------------------------------------------------------
def main(run_dir: Path, num: int, th_edge: float, th_node: float,
         device: str, seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)    
    model, cfg = _load_model(run_dir, device)
    out_dir    = run_dir / "samples_fixed"; out_dir.mkdir(exist_ok=True)

    with torch.no_grad():
        z         = torch.randn(num, cfg["z_dim"], device=device)
        edge_log  = model.dec_edge(z)      # (B, U_off)
        node_log  = model.dec_node(z)      # (B, N)

    for i in range(num):
        # ----------------------------------------------------------------
        # 1) ノード存在確率 → dummy マスク
        node_prob   = node_log[i].sigmoid()          # (N,)
        node_exists = node_prob > th_node            # Bool mask
        R_int       = int(node_exists.sum().item())  # 推定実ノード数

        # ----------------------------------------------------------------
        # 2) エッジ logits (off-diag) → (N,N) ＋ dummy 行/列抑圧
        adj_log = vec_to_adj(edge_log[i], model.max_nodes, diag=False)
        adj_log[:, ~node_exists] = -10.0
        adj_log[~node_exists, :] = -10.0

        # 3) 自己ループはいらないので 0 に潰す
        adj_log.fill_diagonal_(-10.0)

        # ----------------------------------------------------------------
        # 4) 二値化 & NetworkX 化
        adj_bin = (adj_log.sigmoid() > th_edge).float()
        # ---- isolated dummy nodes を除去 ---------------------------
        
        g = nx.from_numpy_array(adj_bin.cpu().numpy())
        g.remove_nodes_from(list(nx.isolates(g)))

        # ---- quick diagnostic -------------------------------------
        mask_dummy = torch.ones_like(adj_bin, dtype=torch.bool)
        mask_dummy[:R_int, :R_int] = False
        p_dummy = adj_bin[mask_dummy].mean().item() if mask_dummy.any() else 0.0
        p_real  = adj_bin[:R_int, :R_int].triu(1).mean().item()
        print(f"[{i}] P(edge|dummy)={p_dummy:.3f}  P(edge|real)={p_real:.3f}")

        _save_graph(g, out_dir / f"sample_{i}")

        # quick metrics
        deg_hist = degree_histogram(g)
        C_avg    = clustering_coef(g)
        print(f"[{i}] nodes={g.number_of_nodes():3d} edges={g.number_of_edges():3d} "
              f"C={C_avg:.3f} max_deg={max(deg_hist)}")

    print(f"✔ {num} graphs saved to {out_dir}")

# ----------------------------------------------------------------------
if __name__ == "__main__":
    pa = argparse.ArgumentParser()
    pa.add_argument("run_dir", type=Path)
    pa.add_argument("--num",      type=int,   default=4)
    pa.add_argument("--th_edge",  type=float, default=0.5,
                    help="edge-probability threshold")
    pa.add_argument("--th_node",  type=float, default=0.5,
                    help="node-existence threshold")
    pa.add_argument("--cpu",      action="store_true")
    pa.add_argument("--seed",     type=int,   default=0)
    args = pa.parse_args()

    device = "cpu" if args.cpu or not torch.cuda.is_available() else "cuda:0"
    main(args.run_dir, args.num, args.th_edge, args.th_node, device, args.seed)
