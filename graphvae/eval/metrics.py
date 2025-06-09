"""
Reusable graph-level metrics for GraphVAE evaluation.
Feel free to extend!
"""
from __future__ import annotations
import networkx as nx
import numpy as np
from typing import Sequence


# --------------------------------------------------------------------- #
#  Degree distribution (histogram) & L1 distance
# --------------------------------------------------------------------- #
def degree_histogram(g: nx.Graph, max_deg: int | None = None) -> np.ndarray:
    """Return fixed-length degree histogram (0..max_deg)."""
    degs = [d for _, d in g.degree()]
    m = max_deg or max(degs, default=0)
    hist = np.bincount(degs, minlength=m + 1).astype(np.float32)
    return hist / hist.sum() if hist.sum() else hist  # normalised


def degree_l1(g1: nx.Graph, g2: nx.Graph, max_deg: int | None = None) -> float:
    h1 = degree_histogram(g1, max_deg)
    h2 = degree_histogram(g2, max_deg or len(h1) - 1)
    m = max(len(h1), len(h2))
    h1 = np.pad(h1, (0, m - len(h1)))
    h2 = np.pad(h2, (0, m - len(h2)))
    return np.abs(h1 - h2).sum() / 2.0


# --------------------------------------------------------------------- #
#  Global metrics
# --------------------------------------------------------------------- #
def clustering_coef(g: nx.Graph) -> float:
    return nx.average_clustering(g)


def aspl(g: nx.Graph) -> float:
    if nx.is_connected(g):
        return nx.average_shortest_path_length(g)
    # for disconnected graphs, take component-wise average
    sp = [nx.average_shortest_path_length(c) for c in (g.subgraph(cc) for cc in nx.connected_components(g))]
    return float(np.average(sp, weights=[len(c) for c in nx.connected_components(g)]))


# --------------------------------------------------------------------- #
#  Batch helpers
# --------------------------------------------------------------------- #
def batch_stat(fn, graphs: Sequence[nx.Graph]) -> float:
    """Compute mean(fn(g)) over a list of graphs."""
    return float(np.mean([fn(g) for g in graphs]))
