import os

import networkx as nx
import matplotlib.pyplot as plt
import scanpy as sc


def _draw_network(G, title="", colors=None, layout=None, save=False, filename=None):
    """
    Draw the gene and cell backbone networks
    :param G: The backbone network.
    :param title: The title.
    :param colors: Discrete matplotlib colormap (optional).
    :param layout: Callable networkx layout function (optional).
    :param save: Whether to save to the figures/ directory.
    :param filename: The filename to save the figure to when the directory is set (optional).
    :return: Drawn figure.
    """
    graph_size = G.number_of_nodes() + G.number_of_edges()
    if graph_size > 7500 and not save:
        print("Plotting skipped, very large graph!")
        return

    if layout is None:
        layout = nx.spring_layout(G, k=1/(G.number_of_nodes()**.25))
    else:
        layout = layout(G)
    if not colors:
        type2color = {'ligand': 'blue', 'receptor': 'orange', 'other': 'lightgray', 'activates': 'green', 'inhibits': 'red', None: 'black'}
    else:
        type2color = {i: color for (i, color) in enumerate(colors)}
        type2color[None] = 'black'
    plt.figure()
    plt.title(title)
    nx.draw(G,
            node_color=[type2color[G.nodes[n].get('type', None)] for n in G],
            edge_color=[type2color[G.edges[e].get('type', None)] for e in G.edges],
            node_size=15, width=.5,
            pos=layout)
    if save:
        os.makedirs("figures/", exist_ok=True)
        if not filename:
            filename = title
        plt.savefig(os.path.join("figures/", filename + ".png"))
    else:
        plt.show()
    plt.clf()


def _make_simulated_adata_plots(adata, save=False):
    timepoint = adata.uns['last_timepoint']

    sc.pl.pca(adata, color=["true_labels", "batch"], title=[f'PCA at t={timepoint} (colored by true labels)',
                                                            f'PCA at t={timepoint} (colored by batch)'],
              show=not save, save=f"_t{timepoint}.png" if save else None)

    # Plot UMAP
    sc.pl.paga(adata, plot=True, title=f"PAGA network at t={timepoint}",
               show=not save is None, save=f"_t{timepoint}.png" if save else None)
    sc.tl.umap(adata, init_pos='paga')
    sc.pl.umap(adata, color=["true_labels", "batch"], title=[f"UMAP at t={timepoint} (colored by true labels)",
                                                             f"UMAP at t={timepoint} (colored by batch)"],
               show=not save is None, save=f"_labels_t{timepoint}.png" if save else None)
    sc.pl.umap(adata, color=['leiden', 'louvain'], title=[f"UMAP at t={timepoint} (colored by leiden clusters)",
                                                          f"UMAP at t={timepoint} (colored by louvain clusters)"],
               show=not save is None, save=f"_cluster_t{timepoint}.png" if save else None)
