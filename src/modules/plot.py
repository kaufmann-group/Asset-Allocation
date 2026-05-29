import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps

"""
draws graph of communities
"""
def draw_graph(Graph, name, labels=None):
    pos = nx.spring_layout(Graph, seed=7)
    cmap = colormaps["tab10"]

    node_colors = "lightblue" if labels is None else [cmap(int(c) % 10) for c in labels]

    plt.figure(figsize=(8, 6))

    edges = list(Graph.edges(data=True))
    weights = np.array([d.get("weight", 0.0) for _, _, d in edges], dtype=float)
    if len(weights) > 0 and weights.max() != weights.min():
        norm_weights = (weights - weights.min()) / (weights.max() - weights.min())
    else:
        norm_weights = np.ones_like(weights) * 0.5

    nx.draw_networkx(Graph, pos=pos, node_color=node_colors, with_labels=True, node_size=500, font_size=9, edge_color=norm_weights, edge_cmap=plt.cm.Greys, width=2)

    plt.title("Graph")
    plt.axis("off")
    plt.savefig(name, dpi=300)
    plt.show()