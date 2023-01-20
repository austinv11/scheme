import itertools
from typing import Tuple, List

import jax
import jax.numpy as jnp
import networkx as nx

from scheme.util import StatefulPRNGKey


def duplication_divergence_graph(n_genes: int, p: float, lr_prob: float, rng_key: StatefulPRNGKey) -> nx.DiGraph:
    """
    Produces an approximately scale-free network.
    Inspired by:
    Nicolau, Miguel, and Marc Schoenauer. "On the evolution of scale-free topologies with a gene regulatory network model." Biosystems 98.3 (2009): 137-148.
    Ispolatov, Iaroslav, Pavel L. Krapivsky, and Anton Yuryev. "Duplication-divergence model of protein interaction network." Physical review E 71.6 (2005): 061911.
    :param n_genes: The number of genes to include.
    :param p: Mutation rate. Higher values lead to more mutations.
    :param lr_prob: Probability of a gene being mutated to a ligand/receptor.
    :param rng_key: The random number generator key.
    :return: Gene-regulatory network.
    """
    assert 0 <= p <= 1
    assert 0 < lr_prob < 1

    G = nx.DiGraph()

    def _make_strength(sign=None):
        strength = jax.random.normal(rng_key(), shape=())
        strength = jnp.clip(strength, -1, 1)

        if sign is not None:
            if jax.lax.ne(jax.numpy.sign(sign), jax.numpy.sign(strength)):
                strength *= -1

        return strength.item()

    def _add_edge(u, v):
        G.add_edge(u, v, weight=_make_strength())

    G.add_node(0, type='gene')
    G.add_node(1, type='gene')

    _add_edge(0, 1)
    _add_edge(1, 0)

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
                G.edges[neighbor, curr_gene]['weight'] = _make_strength(G.edges[neighbor, node]['weight'])

        # Duplicate edges from neighbors of duplicated node to new node
        for i, neighbor in enumerate(G.successors(node)):
            if not jax.random.bernoulli(rng_key(), p, shape=()).item():
                G.add_edge(curr_gene, neighbor, **G.edges[node, neighbor])
                # Redraw the strength of the edge
                G.edges[curr_gene, neighbor]['weight'] = _make_strength(G.edges[node, neighbor]['weight'])

        # Should we add a connection to the progenitor?
        if jax.random.bernoulli(rng_key(), p, shape=()).item():
            _add_edge(curr_gene, node)

        if G.nodes[curr_gene]['type'] == 'gene':
            # Should this new gene be a receptor
            if jax.random.bernoulli(rng_key(), lr_prob, shape=()).item():
                G.nodes[curr_gene]['type'] = 'receptor'
                # Annotate predecessors as ligands randomly
                for i, neighbor in enumerate(G.predecessors(curr_gene)):
                    if G.nodes[neighbor]['type'] == 'gene':
                        if jax.random.bernoulli(rng_key(), lr_prob, shape=()).item():
                            G.nodes[neighbor]['type'] = 'ligand'

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
        if G.nodes[node]['type'] == 'ligand':
            # Require at least one successor to be a receptor
            if not any(G.nodes[neighbor]['type'] == 'receptor' for neighbor in G.successors(node)):
                G.remove_node(node)

        elif G.nodes[node]['type'] == 'receptor':
            # Require at least one predecessor to be a ligand
            if not any(G.nodes[neighbor]['type'] == 'ligand' for neighbor in G.predecessors(node)):
                G.remove_node(node)

    return G

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


def _generate_gene_backbone(n_genes, sparsity_factor: float, rng_key: StatefulPRNGKey):
    """
    Produce a network representing a gene interaction backbone topology.
    :param n_genes: The number of genes to include.
    :param sparsity_factor: The percentage of edges to prune out after network generation to prevent a "hairball" type network.
    :param rng_key: The random number generator key.
    :return: The final network.
    """
    # Generate a random initial regulatory network
    # Follows a scale-free network random expectation
    base_gene_backbone = nx.scale_free_graph(n_genes,
                                             alpha=0.3,  # Prob for adding a new node to an existing node
                                             beta=0.65,  # Prob for adding a new edge to an existing node
                                             gamma=0.05,  # Prob for adding a new edge to a new node
                                             ).reverse()  # Flip the direction of edges for convenience

    # Convert from MultiDiGraph to DiGraph to prevent multiple parallel edges
    base_gene_backbone = nx.DiGraph(base_gene_backbone)

    # Annotate nodes with gene names and type
    for node in base_gene_backbone.nodes:
        base_gene_backbone.nodes[node]["name"] = f"gene_{node}"
        in_degree = base_gene_backbone.in_degree(node)
        out_degree = base_gene_backbone.out_degree(node)
        if in_degree <= 1 and out_degree > 0:
            base_gene_backbone.nodes[node]["type"] = 'ligand'
        else:
            base_gene_backbone.nodes[node]["type"] = 'gene'
    # Annotate receptors
    lr_pairs = []  # Track the official ligand/receptor pairs
    for node in base_gene_backbone.nodes:
        if base_gene_backbone.nodes[node]["type"] == 'ligand':
            for neighbor in base_gene_backbone.neighbors(node):
                if base_gene_backbone.nodes[neighbor]["type"] != 'ligand' and base_gene_backbone.out_degree(node) < 5:
                    base_gene_backbone.nodes[neighbor]["type"] = 'receptor'
                    lr_pairs.append((node, neighbor))

    # Make network more sparse
    _sparsify_graph(base_gene_backbone, sparsity_factor, rng_key=rng_key)

    return base_gene_backbone, lr_pairs


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

