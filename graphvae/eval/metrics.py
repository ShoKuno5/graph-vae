import networkx as nx
import numpy as np
from scipy.stats import wasserstein_distance
import hashlib
from typing import List

# =============================================================================
# Utility
# =============================================================================

def to_nx(edge_index, num_nodes):
    """(PyG) edge_index → networkx.Graph"""
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    for src, dst in edge_index.t().tolist():
        if src != dst:  # ignore self-loops
            G.add_edge(src, dst)
    return G

# =============================================================================
# Validity family
# =============================================================================

def is_valid(G: nx.Graph, k_max: int = 6) -> bool:
    """Return True if the graph passes *all* hard constraints.

    Current constraints:
        • connected
        • no self‑loops
        • max degree ≤ k_max  (default 6)
    """
    return (
        nx.is_connected(G)
        and nx.number_of_selfloops(G) == 0
        and max(dict(G.degree()).values(), default=0) <= k_max
    )


def why_invalid(G: nx.Graph, k_max: int = 6) -> str:
    """Label the first rule that the graph violates."""
    if not nx.is_connected(G):
        return "disconnected"
    if nx.number_of_selfloops(G):
        return "selfloop"
    if max(dict(G.degree()).values(), default=0) > k_max:
        return "deg>k"
    return "other"


def validity(graphs: List[nx.Graph], k_max: int = 6) -> float:
    """Fraction of graphs that are *valid* under `is_valid`."""
    valids = [is_valid(g, k_max) for g in graphs]
    return sum(valids) / len(valids)

# =============================================================================
# Uniqueness family
# =============================================================================

def uniqueness(graphs: List[nx.Graph]) -> float:
    """Edge‑list based uniqueness."""
    canon = [tuple(sorted(g.edges())) for g in graphs]
    return len(set(canon)) / len(graphs)


# --- ISO‑aware uniqueness -----------------------------------------------------

def _wl_canonical_hash(G: nx.Graph) -> str:
    """Weisfeiler–Lehman hash (attr‑agnostic) shortened with MD5."""
    H = nx.convert_node_labels_to_integers(G)
    wl = nx.weisfeiler_lehman_graph_hash(H, iterations=3)
    return hashlib.md5(wl.encode()).hexdigest()


def uniqueness_iso(graphs: List[nx.Graph]) -> float:
    """Count graphs up to isomorphism using WL hash."""
    hashes = {_wl_canonical_hash(g) for g in graphs}
    return len(hashes) / len(graphs)

# =============================================================================
# Degree‑MMD  (1‑D Wasserstein‑1 / Earth‑Mover distance)
# =============================================================================

def degree_mmd(graphs_ref: List[nx.Graph],
               graphs_gen: List[nx.Graph],
               max_deg: int = 10) -> float:
    """W₁ distance between degree distributions of reference & generated sets."""
    def avg_hist(graphs):
        hist_sum = np.zeros(max_deg + 1)
        cnt = 0
        for g in graphs:
            degs = np.array([d for _, d in g.degree()])
            if degs.size == 0:
                continue
            h, _ = np.histogram(degs, bins=range(max_deg + 2))
            if h.sum() == 0:
                continue
            hist_sum += h / h.sum()   # normalize per‑graph hist
            cnt += 1
        return hist_sum / max(cnt, 1)

    X = avg_hist(graphs_ref)
    Y = avg_hist(graphs_gen)
    bins = np.arange(len(X))           # support positions 0,1,…,max_deg
    return float(
        wasserstein_distance(bins, bins, u_weights=X, v_weights=Y)
    )
