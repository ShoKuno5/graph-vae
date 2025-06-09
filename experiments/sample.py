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
    max_nodes  = cfg.get("max_nodes")  or state["dec_lin.weight"].size(0)  # (=U)≒N(N-1)/2 → 近似推定でも十分

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
        vec = model.dec(z)                    # (B, U)

    for i in range(num):
        adj_prob = vec_to_adj(vec[i], model.max_nodes).sigmoid()
        adj_bin  = (adj_prob > th).float()

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
