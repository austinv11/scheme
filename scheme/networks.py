from typing import Tuple, List

import jax
import jax.numpy as jnp
import networkx as nx

from scheme.util import StatefulPRNGKey


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
            base_gene_backbone.nodes[node]["type"] = 'other'
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

