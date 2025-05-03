# 1) 生成ループと同じ文脈で
ref, gen  = sample_graphs(model, loader, 1, device)
G0 = gen[0]

import networkx as nx, matplotlib.pyplot as plt
nx.draw(G0, with_labels=True)
plt.show()
