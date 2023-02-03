import itertools
from collections import defaultdict
from typing import Tuple, List, Dict

import jax
import jax.numpy as jnp
import networkx as nx
import numpy as np
from networkx.algorithms import community

from scheme.util import StatefulPRNGKey


def _make_strength(rng_key: StatefulPRNGKey, sign=None):
    strength = jax.random.normal(rng_key(), shape=())
    strength = jnp.clip(strength, -1, 1)

    if sign is not None:
        if jax.lax.ne(jax.numpy.sign(sign), jax.numpy.sign(strength)):
            strength *= -1

    return strength.item()


def duplication_divergence_graph(n_genes: int, p: float, rec_prop: float, rng_key: StatefulPRNGKey) -> nx.DiGraph:
    """
    Produces an approximately scale-free network.
    Inspired by:
    Nicolau, Miguel, and Marc Schoenauer. "On the evolution of scale-free topologies with a gene regulatory network model." Biosystems 98.3 (2009): 137-148.
    Ispolatov, Iaroslav, Pavel L. Krapivsky, and Anton Yuryev. "Duplication-divergence model of protein interaction network." Physical review E 71.6 (2005): 061911.
    :param n_genes: The number of genes to include.
    :param p: Mutation rate. Higher values lead to more mutations.
    :param rec_prop: Proportion of genes being a receptor.
    :param rng_key: The random number generator key.
    :return: Gene-regulatory network.
    """
    assert 0 <= p <= 1
    assert 0 < rec_prop < 1

    G = nx.DiGraph()

    def _add_edge(u, v):
        G.add_edge(u, v, weight=_make_strength(rng_key))

    G.add_node(0, is_ligand=False, is_receptor=False)
    G.add_node(1, is_ligand=False, is_receptor=False)
    G.add_node(2, is_ligand=False, is_receptor=False)

    # Add initial bidirectional edge
    _add_edge(0, 1)
    _add_edge(1, 0)

    # Add initial self-loop and unidirectional edge
    _add_edge(2, 2)  # Allow for self-loops
    _add_edge(2, 1)

    curr_gene = 2
    while G.number_of_nodes() < n_genes:
        # Select a node to duplicate
        node = jax.random.choice(rng_key(), jnp.arange(G.number_of_nodes(), dtype=int)).item()

        # Add new node
        G.add_node(curr_gene, **G.nodes[node])

        # Duplicate edges from new node to neighbors of duplicated node
        for i, neighbor in enumerate(G.predecessors(node)):
            if not jax.random.bernoulli(rng_key(), p, shape=()).item():
                G.add_edge(neighbor, curr_gene, **G.edges[neighbor, node])
                # Redraw the strength of the edge
                G.edges[neighbor, curr_gene]['weight'] = _make_strength(rng_key, G.edges[neighbor, node]['weight'])

        # Duplicate edges from neighbors of duplicated node to new node
        for i, neighbor in enumerate(G.successors(node)):
            if not jax.random.bernoulli(rng_key(), p, shape=()).item():
                G.add_edge(curr_gene, neighbor, **G.edges[node, neighbor])
                # Redraw the strength of the edge
                G.edges[curr_gene, neighbor]['weight'] = _make_strength(rng_key, G.edges[node, neighbor]['weight'])

        # Should we add a connection to the progenitor?
        if jax.random.bernoulli(rng_key(), p, shape=()).item():
            _add_edge(curr_gene, node)

        if not G.nodes[curr_gene]['is_receptor']:
            # Should this new gene be a receptor
            if jax.random.bernoulli(rng_key(), rec_prop, shape=()).item():
                G.nodes[curr_gene]['is_receptor'] = True
                # Annotate predecessors as ligands
                for i, neighbor in enumerate(G.predecessors(curr_gene)):
                    G.nodes[neighbor]['is_ligand'] = True

        if G.degree(curr_gene) == 0:
            # No neighbors, remove and continue
            G.remove_node(curr_gene)
            continue
        else:
            curr_gene += 1

        # Removals accounted for. Now add new edges
        # added_edges = jax.random.bernoulli(rng_key(), p, shape=(G.number_of_nodes(),))
        # for neighbor in jnp.argwhere(added_edges):
        #     neighbor = neighbor.item()
        #
        #     # If the edge exists, we replace it
        #     if G.has_edge(neighbor, curr_gene):
        #         G.remove_edge(neighbor, curr_gene)
        #
        #     _add_edge(neighbor, curr_gene)

    # Remove any ligands or receptors without partners
    for node in list(G.nodes):
        if G.nodes[node]['is_ligand']:
            # Require at least one successor to be a receptor
            if not any(G.nodes[neighbor]['is_receptor'] for neighbor in G.successors(node)):
                G.remove_node(node)
                continue
            else:
                G.nodes[node]['type'] = 'ligand'

        if G.nodes[node]['is_receptor']:
            # Require at least one predecessor to be a ligand
            if not any(G.nodes[neighbor]['is_ligand'] for neighbor in G.predecessors(node)):
                G.remove_node(node)
                continue
            else:
                G.nodes[node]['type'] = 'receptor'

        if G.nodes[node]['is_ligand'] and G.nodes[node]['is_receptor']:
            G.nodes[node]['type'] = 'ligand/receptor'

        if not G.nodes[node]['is_ligand'] and not G.nodes[node]['is_receptor']:
            G.nodes[node]['type'] = 'gene'

    return G


def count_gene_types(G: nx.DiGraph) -> Dict[str, int]:
    """
    Count the number of genes of each type.
    :param G: The gene network.
    :return: A dictionary of type counts.
    """
    gene_types = defaultdict(int)
    for node in G.nodes:
        gene_types[G.nodes[node]['type']] += 1

    return gene_types


def annotate_cell_type_specific_gene_modules(G: nx.DiGraph,
                                             n_cell_type: int,
                                             rng_key: StatefulPRNGKey,
                                             generic_gene_proportion: float = 0.6) -> nx.DiGraph:
    """
    Annotate gene regulatory network with cell type specific genes.
    This tries to split the gene network into clusters and then randomly assign genes to cell types.
    :param G: The initial gene network.
    :param n_cell_type: The number of cell types.
    :param rng_key: The random number generator key.
    :param generic_gene_proportion: The proportion of genes that are non-cell type specific.
    :return: The annotated gene network.
    """
    # Remove connections between ligand and receptors to force them to be bridges between cell types.
    clustering_graph = G.copy()
    for node in G.nodes:
        if G.nodes[node]['is_receptor']:
            for neighbor in G.predecessors(node):
                if G.nodes[neighbor]['is_ligand']:
                    clustering_graph.remove_edge(neighbor, node)

    # Cluster the graph
    communities = community.louvain_communities(clustering_graph,
                                                weight=None,  # Don't use edge weights, just consider topology
                                                resolution=1.1,  # Resolution > 1 so that the algorithm favors smaller communities
                                                seed=np.random.RandomState(rng_key()))

    # Assign genes to cell types
    sorted_communities = sorted(communities, key=lambda x: len(x), reverse=True)
    # First, assign generic genes by taking biggest communities and assigning them to cell type None
    accumulated_proportion = 0
    for cluster in sorted_communities:
        if accumulated_proportion < generic_gene_proportion:
            for gene in cluster:
                G.nodes[gene]['cell_type'] = None
            accumulated_proportion += len(cluster) / G.number_of_nodes()
        else:
            assigned_celltype = jax.random.choice(rng_key(), jnp.arange(n_cell_type, dtype=int)).item()
            for gene in cluster:
                G.nodes[gene]['cell_type'] = assigned_celltype

    return G


def produce_celltype_perturbed_gene_backbone(G: nx.DiGraph,
                                             cell_type: int,
                                             rng_key: StatefulPRNGKey,
                                             edge_remove_prob: float = 0.025,
                                             edge_add_prob: float = 0.025,
                                             edge_mutate_prob: float = 0.025) -> nx.DiGraph:
    """
    Produce a perturbed gene network backbone for a given cell type.
    :param G: The base gene network.
    :param cell_type: The cell type to produce the gene network for.
    :param rng_key: The random number generator key.
    :param edge_remove_prob: The probability of removing an edge.
    :param edge_add_prob: The probability of adding an edge.
    :param edge_mutate_prob: The probability of mutating an edge weight.
    :return: The cell-type specific gene network.
    """
    Gnew = G.copy()
    # Remove incorrect cell-type edges
    for node in G.nodes:
        ct = G.nodes[node]['cell_type']
        if ct is not None and ct != cell_type:
            # Delete input and output edges
            for neighbor in list(Gnew.predecessors(node)):
                Gnew.remove_edge(neighbor, node)
            for neighbor in list(Gnew.successors(node)):
                Gnew.remove_edge(node, neighbor)

    edges_to_remove = int(Gnew.number_of_edges() * edge_remove_prob)
    edges_to_add = int(Gnew.number_of_edges() * edge_add_prob)
    edges_to_mutate = int(Gnew.number_of_edges() * edge_mutate_prob)

    # Remove edges
    for _ in range(edges_to_remove):
        edge = jax.random.choice(rng_key(), jnp.arange(Gnew.number_of_edges(), dtype=int)).item()
        edge = list(Gnew.edges)[edge]
        Gnew.remove_edge(edge[0], edge[1])

    # Add edges
    for _ in range(edges_to_add):
        node1 = jax.random.choice(rng_key(), jnp.arange(Gnew.number_of_nodes(), dtype=int)).item()
        node2 = jax.random.choice(rng_key(), jnp.arange(Gnew.number_of_nodes(), dtype=int)).item()
        node1 = list(Gnew.nodes)[node1]
        node2 = list(Gnew.nodes)[node2]
        Gnew.add_edge(node1, node2, weight=_make_strength(rng_key))

    # Mutate edge weight
    for _ in range(edges_to_mutate):
        edge = jax.random.choice(rng_key(), jnp.arange(Gnew.number_of_edges(), dtype=int)).item()
        edge = list(Gnew.edges)[edge]
        Gnew[edge[0]][edge[1]]['weight'] = _make_strength(rng_key)

    return Gnew


def _sparsify_graph(G, sparsity_factor, rng_key: StatefulPRNGKey):  # Sparsify the graph
    # Remove edges according to sparsity factor
    # With a higher likelihood of removing edges from highly central nodes
    num_edges_to_remove = int(G.number_of_edges() * sparsity_factor)
    edges = list(G.edges)
    edge_probs = []
    for edge in edges:
        min_degree = min(G.degree(edge[0]), G.degree(edge[1]))
        edge_probs.append(min_degree)
    edge_probs = jnp.array(edge_probs)
    edge_probs = edge_probs / edge_probs.sum()
    edges_to_remove = jax.random.choice(rng_key(), jnp.arange(len(edges), dtype=int),
                                    shape=(num_edges_to_remove,),
                                    replace=False, p=edge_probs)

    for i in edges_to_remove:
        edge = edges[i]
        G.remove_edge(edge[0], edge[1])


# def _generate_gene_backbone(n_genes, sparsity_factor: float, rng_key: StatefulPRNGKey):
#     """
#     Produce a network representing a gene interaction backbone topology.
#     :param n_genes: The number of genes to include.
#     :param sparsity_factor: The percentage of edges to prune out after network generation to prevent a "hairball" type network.
#     :param rng_key: The random number generator key.
#     :return: The final network.
#     """
#     # Generate a random initial regulatory network
#     # Follows a scale-free network random expectation
#     base_gene_backbone = nx.scale_free_graph(n_genes,
#                                              alpha=0.3,  # Prob for adding a new node to an existing node
#                                              beta=0.65,  # Prob for adding a new edge to an existing node
#                                              gamma=0.05,  # Prob for adding a new edge to a new node
#                                              ).reverse()  # Flip the direction of edges for convenience
#
#     # Convert from MultiDiGraph to DiGraph to prevent multiple parallel edges
#     base_gene_backbone = nx.DiGraph(base_gene_backbone)
#
#     # Annotate nodes with gene names and type
#     for node in base_gene_backbone.nodes:
#         base_gene_backbone.nodes[node]["name"] = f"gene_{node}"
#         in_degree = base_gene_backbone.in_degree(node)
#         out_degree = base_gene_backbone.out_degree(node)
#         if in_degree <= 1 and out_degree > 0:
#             base_gene_backbone.nodes[node]["type"] = 'ligand'
#         else:
#             base_gene_backbone.nodes[node]["type"] = 'gene'
#     # Annotate receptors
#     lr_pairs = []  # Track the official ligand/receptor pairs
#     for node in base_gene_backbone.nodes:
#         if base_gene_backbone.nodes[node]["type"] == 'ligand':
#             for neighbor in base_gene_backbone.neighbors(node):
#                 if base_gene_backbone.nodes[neighbor]["type"] != 'ligand' and base_gene_backbone.out_degree(node) < 5:
#                     base_gene_backbone.nodes[neighbor]["type"] = 'receptor'
#                     lr_pairs.append((node, neighbor))
#
#     # Make network more sparse
#     _sparsify_graph(base_gene_backbone, sparsity_factor, rng_key=rng_key)
#
#     return base_gene_backbone, lr_pairs


def _generate_network_from_backbone(backbone: nx.DiGraph, lr_pairs: List[Tuple[int, int]], permutation_strength: float, max_count: int, rng_key: StatefulPRNGKey):
    """
    Generate a gene network from a given backbone network.
    :param backbone: The initial backbone network.
    :param lr_pairs: Ligand-receptor pair interactions that are enforced.
    :param permutation_strength: Hyperparameter that modulates the strength of the modifications from the backbone.
    :param max_count: The max random expected expression.
    :param rng_key: The random number generator key.
    :return: The new gene network.
    """
    # Copy so we can modify it
    backbone = backbone.copy()
    added = 0
    removed = 0

    # Decorate the graph with self loops
    SELF_LOOP_PROB = 0.01 * permutation_strength
    # Decorate the graph with random edges added and removed as noise
    RANDOM_EDGE_ADD_PROB = .001 * permutation_strength
    # Probability a new edge that impacts ligands/receptors is kept
    DECOY_PROB = 0.5 * permutation_strength
    for node in backbone.nodes:
        if jax.random.uniform(rng_key()) < SELF_LOOP_PROB:
            backbone.add_edge(node, node)
            added += 1
        for other_node in backbone.nodes:
            from_node = node
            to_node = other_node
            # Swap directions if the order is wrong
            if backbone.nodes[from_node]['type'] == 'receptor' and backbone.nodes[to_node]['type'] == 'ligand':
                temp = from_node
                from_node = to_node
                to_node = temp
            # If ligand/receptor interaction, make sure its a real interaction
            from_type = backbone.nodes[from_node]['type']
            to_type = backbone.nodes[to_node]['type']

            if len({from_type, to_type} & {'ligand', 'receptor'}) > 0:
                # Decoy conditions
                # Ligand/receptor interaction not known
                wrong_lr = (from_type == 'ligand' and to_type == 'receptor') and ((from_type, to_type) not in lr_pairs)
                # Interaction for ligand or receptor that is non L/R
                no_lr = not (from_type == 'ligand' and to_type == 'receptor')

                if wrong_lr or no_lr:
                    if jax.random.uniform(rng_key()) > DECOY_PROB:
                        continue

            if jax.random.uniform(rng_key()) < RANDOM_EDGE_ADD_PROB:
                backbone.add_edge(from_node, to_node)
                added += 1

    RANDOM_EDGE_REMOVE_PROB = (added / backbone.number_of_edges())
    for edge in list(backbone.edges()):
        if jax.random.uniform(rng_key()) < RANDOM_EDGE_REMOVE_PROB:
            if backbone.has_edge(*edge):
                backbone.remove_edge(*edge)
                removed += 1

    # Annotate edges with interaction types
    for edge in backbone.edges:
        edge_type = "activates" if jax.random.uniform(rng_key()) < 0.6 else "inhibits"
        backbone.edges[edge]["type"] = edge_type
        backbone.edges[edge]["connectivity"] = 1 if edge_type == "activates" else -1
        backbone.edges[edge]["weight"] = jax.random.uniform(rng_key())
        backbone.edges[edge]["effect_threshold"] = jax.random.randint(rng_key(), (), 0, max_count+1)

    print(f"Added {added} edges, removed {removed} edges!")

    return backbone

