import networkx as nx, matplotlib.pyplot as plt
G = M.to_nx(edge_index_gen, data.num_nodes)
nx.draw(G, with_labels=True)
plt.show()
