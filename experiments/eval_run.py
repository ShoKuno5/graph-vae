#!/usr/bin/env python
"""
Evaluate a run directory that contains:
    - params.yaml
    - samples/*.gml (generated graphs)
Compare those samples with the training graphs used for that run.

Usage
-----
python experiments/eval_run.py RUN_DIR [--train_dir DATASET_DIR]

If --train_dir is omitted, the script tries to infer it from params.yaml
(e.g. ENZYMES will be loaded via torch_geometric). Feel free to adapt.
"""
import argparse, yaml, glob
from pathlib import Path
import networkx as nx
import numpy as np

from graphvae.eval import metrics as M


# --------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------- #
def load_gml_list(pattern: str):
    return [nx.read_gml(p) for p in glob.glob(pattern)]


def print_table(rows, headers):
    col_w = [max(len(str(x)) for x in col) for col in zip(*([headers] + rows))]
    fmt = "  ".join("{:<" + str(w) + "}" for w in col_w)
    print(fmt.format(*headers))
    print("-" * (sum(col_w) + 2 * (len(col_w) - 1)))
    for r in rows:
        print(fmt.format(*r))


# --------------------------------------------------------------------- #
def main(run_dir: Path, train_dir: Path | None):
    run_dir = run_dir.expanduser().resolve()
    params = yaml.safe_load(open(run_dir / "params.yaml"))

    # ---------- load generated graphs ----------
    gen_graphs = load_gml_list(str(run_dir / "samples" / "*.gml"))
    if not gen_graphs:
        raise RuntimeError("No samples/*.gml found — run sample.py first.")

    # ---------- load reference (training) graphs ----------
    if train_dir:
        ref_graphs = load_gml_list(str(Path(train_dir).expanduser() / "*.gml"))
    else:
        # quick fallback: load via torch_geometric if dataset known
        from torch_geometric.datasets import TUDataset
        from torch_geometric.utils import to_networkx
        td = TUDataset(root="/tmp/TUD", name=params["dataset"].upper())
        ref_graphs = [to_networkx(d, to_undirected=True) for d in td]

    # ---------- compute stats ----------
    rows = []
    for name, lst in [("Reference", ref_graphs), ("Generated", gen_graphs)]:
        rows.append(
            [
                name,
                len(lst),
                f"{M.batch_stat(M.clustering_coef, lst):.3f}",
                f"{M.batch_stat(M.aspl, lst):.3f}",
                f"{np.mean([g.number_of_nodes() for g in lst]):.1f}",
                f"{np.mean([g.number_of_edges() for g in lst]):.1f}",
            ]
        )

    # ---------- print ----------
    print(f"\nRun directory : {run_dir}")
    print(f"Train graphs  : {len(ref_graphs)}   |  Generated graphs : {len(gen_graphs)}\n")
    print_table(
        rows,
        headers=["Set", "#graphs", "C_avg", "ASPL", "N_nodes", "N_edges"],
    )

    # ---------- degree L1 distance ----------
    l1s = [M.degree_l1(g1, g2) for g1, g2 in zip(ref_graphs[: len(gen_graphs)], gen_graphs)]
    print(f"\nDegree-hist L1 (pair-wise, len={len(l1s)}):  mean={np.mean(l1s):.3f}")

    print("\n✔ evaluation finished")


# --------------------------------------------------------------------- #
if __name__ == "__main__":
    pa = argparse.ArgumentParser()
    pa.add_argument("run_dir", type=Path, help="runs/YYYYMMDD_HHMMSS")
    pa.add_argument("--train_dir", type=Path, help="optional dir with .gml of training graphs")
    args = pa.parse_args()

    main(args.run_dir, args.train_dir)
