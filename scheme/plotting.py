import os

import jax.numpy as jnp
import networkx as nx
import matplotlib.pyplot as plt
import scanpy as sc


def _draw_network(G, title="", colors=None, color_prop=None, layout=None, save=False, filename=None):
    """
    Draw the gene and cell backbone networks
    :param G: The backbone network.
    :param title: The title.
    :param colors: Discrete matplotlib colormap (optional).
    :param color_prop: The property to color the nodes by (optional).
    :param layout: Callable networkx layout function (optional).
    :param save: Whether to save to the figures/ directory.
    :param filename: The filename to save the figure to when the directory is set (optional).
    :return: Drawn figure.
    """
    plt.clf()
    #graph_size = G.number_of_nodes() + G.number_of_edges()
    #if graph_size > 7500 and not save:
    #    print("Plotting skipped, very large graph!")
    #    return

    if not color_prop:
        color_prop = 'type'

    # Create copy of graph so we can make weights the absolute value
    G_copy = G.copy()
    for edge in list(G_copy.edges):
        G_copy.edges[edge]['weight'] = abs(G_copy.edges[edge]['weight'])

    if layout is None:
        layout = nx.spring_layout(G, k=1/(G.number_of_nodes()**.25))
    else:
        layout = layout(G)
    if not colors:
        type2color = {'ligand': 'blue', 'receptor': 'red', 'ligand/receptor': 'orange', 'gene': 'lightgray', 'other': 'lightgray', 'activates': 'green', 'inhibits': 'red', None: 'black'}
    elif isinstance(colors, dict):
        type2color = colors
    else:
        type2color = {i: color for (i, color) in enumerate(colors)}

    if None not in type2color:
        type2color[None] = 'black'

    plt.figure()
    plt.title(title)
    nx.draw(G.reverse(),
            node_color=[type2color[G.nodes[n].get(color_prop, None)] for n in G],
            edge_color=[type2color[G.edges[e].get(color_prop, None)] for e in G.edges],
            node_size=15, width=.5,
            pos=layout)
    if save:
        os.makedirs("figures/", exist_ok=True)
        if not filename:
            filename = title
        plt.savefig(os.path.join("figures/", filename + ".png"))
    else:
        plt.show()


def _make_simulated_adata_plots(adata, save=False):
    plt.clf()

    timepoint = adata.uns['timestep']

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


def _draw_voronoi_slice(matrix: jnp.ndarray, slice_idx: int, title: str, save: bool = False):
    """
    Draw a voronoi slice of the given matrix.
    :param matrix: The matrix to draw the slice of.
    :param slice_idx: The index of the slice to draw.
    :param title: The title of the plot.
    :param save: Whether to save the plot.
    """
    plt.clf()
    plt.figure()
    plt.axis('off')
    plt.grid(False)
    plt.title(title)
    # Plot as pixel array, with a discrete colormap
    p = plt.imshow(matrix[slice_idx, :, :], cmap='tab20', interpolation='nearest')
    if save:
        os.makedirs("figures/", exist_ok=True)
        p.savefig(os.path.join("figures/", title + ".png"))
    return p


def _draw_voronoi_slices_animation(matrix: jnp.ndarray, title: str, save: bool = False, filename: str = "voronoi.gif"):
    """
    Draw an animation of the voronoi slices of the given matrix.
    :param matrix: The matrix to draw the slices of.
    :param title: The title of the plot.
    :param save: Whether to save the animation.
    :param filename: The filename to save the animation to.
    """
    import matplotlib.animation as animation
    plt.clf()
    fig = plt.figure()
    plt.axis('off')
    plt.grid(False)
    ims = []
    for i in range(matrix.shape[0]):
        im = plt.imshow(matrix[i, :, :],
                        cmap='tab20',
                        interpolation='nearest',
                        #title=f"Slice {i+1}/{matrix.shape[0]}"
                        )
        ims.append([im])
    ani = animation.ArtistAnimation(fig, ims, interval=750, blit=True, repeat_delay=750)
    plt.title(title)
    if save:
        os.makedirs("figures/", exist_ok=True)
        ani.save("figures/" + filename)
    return ani
