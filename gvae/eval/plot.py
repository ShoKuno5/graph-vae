"""Utility to visualize generated graphs."""

import networkx as nx
import matplotlib.pyplot as plt

from . import metrics as M


def plot_graph(edge_index, num_nodes):
    """Draw a single generated graph using NetworkX."""
    G = M.to_nx(edge_index, num_nodes)
    nx.draw(G, with_labels=True)
    plt.show()

