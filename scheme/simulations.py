import itertools
import os
from collections.abc import Iterable
from dataclasses import dataclass
from typing import List, Tuple, Union, Optional, Dict, Iterator

import anndata as ad
import networkx as nx
# Note that we are using jax to speed up calculations
import jax.numpy as jnp
import jax
from jax import lax
import treeo as to  # Helps define dataclasses as jax trees
from tqdm import tqdm
import scanpy as sc
import matplotlib.pyplot as plt

from scheme.util import StatefulPRNGKey, bimodal_normal, parameterized_normal, \
    negative_binomial, jax_jit, consumes_key, jax_vmap


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


@consumes_key(1, 'rng_key')
@jax_jit()
def _batch_effect_to_sparsity_factor(batch_effect, rng_key: StatefulPRNGKey):
    return (jnp.minimum(batch_effect, 1.5)) * jax.random.uniform(rng_key)/4


@consumes_key(1, 'rng_key', count=2)
@jax_jit()
def _emphasize_group(proportions, prioritized: int, rng_key: StatefulPRNGKey):
    """Emphasize a group of proportions from a list of proportions."""
    proportions /= jax.random.randint(rng_key[0], (), 25, 250)  # Down-weigh initial proportions
    proportions = proportions.at[prioritized].add(jax.random.uniform(rng_key[1]) * (1 - proportions.sum()))  # Make intracluster relationships more important
    return proportions


@consumes_key(2, 'rng_key', count=4)
@jax_jit(static_argnums=(0,))
def _randomized_proportions(groups: int, prioritize_group: Optional[int], rng_key: StatefulPRNGKey):
    """Generate random proportions that sum to 1"""
    # Bimodal distribution
    proportions = jnp.clip(bimodal_normal((.25, .75), .25, groups, rng_key[:2]), 0, 1)  # We want groups to have more varied sizes
    proportions /= proportions.sum()
    # For cell-cell-communication networks, we want to emphasize the intracluster relationships
    proportions = lax.cond(prioritize_group is None, lambda prop, prioritize, keys: prop, _emphasize_group,
                           proportions, prioritize_group, rng_key[2:4])
    return proportions


def _generate_cell_backbone(n_labels, n_cells, batch_effect, cell_type_interaction_probs, rng_key: StatefulPRNGKey):
    while True:
        cell_distribution = (_randomized_proportions(n_labels, None, rng_key=rng_key) * n_cells)

        # Batch effect can disrupt the scale of the number of cells per sample
        # Note that mean is 0, so 2^0= 1 so expected value is no change
        # We clip the value for numerical stability
        cell_distribution = jnp.clip(2**parameterized_normal(0, .2 * (1 + batch_effect), cell_distribution.shape, key=rng_key), .1, 1.5) * cell_distribution
        # Batch effect impacting cell communication probabilities:
        cell_type_interaction_probs = jnp.clip((jnp.clip(2**parameterized_normal(0, .2 * (1 + batch_effect), cell_type_interaction_probs.shape, key=rng_key), .1, 1.5) * cell_type_interaction_probs), 0, 1)

        cell_backbone = nx.stochastic_block_model(
            cell_distribution.astype(int).tolist(),  # Split cells into types
            cell_type_interaction_probs.tolist(),
            selfloops=True,
            directed=True
        )

        # Get the partitions as the "true" labels
        for partition, nodes in enumerate(cell_backbone.graph['partition']):
            for node in nodes:
                cell_backbone.nodes[node]['type'] = partition

        # Batch effect can disrupt the connections
        _sparsify_graph(cell_backbone, _batch_effect_to_sparsity_factor(batch_effect, rng_key=rng_key), rng_key=rng_key)

        # Annotate edges with random diffusion strengths
        for edge in list(cell_backbone.edges):
            diffusion = jnp.clip(parameterized_normal(.5, .1 * (1 + batch_effect), shape=(), key=rng_key), 0, 1)
            cell_backbone.edges[edge]['diffusion'] = diffusion
            if diffusion <= 0:
                cell_backbone.remove_edge(*edge)

        # Make sure that there are no outliers
        if min(deg for node, deg in nx.degree(cell_backbone)) > 0:
            print("Generated cell backbone with", n_cells, "cells")
            break
        else:
            print("Singleton cells detected, retrying")

    return cell_backbone


@dataclass(eq=True, unsafe_hash=True)
class BatchData(to.Tree):
    """
    Cached information holder for data batches.
    """
    # Number of cells in the batch
    n_cells: int = to.field(node=False, hash=True)
    # The base cell interaction network
    cell_backbone: nx.Graph = to.field(node=False, hash=False)
    # The base gene interaction networks
    gene_backbones: Tuple[nx.DiGraph, ...] = to.field(node=False, hash=False)
    # The connections between cells
    cell_connectivity: jnp.ndarray = to.field(node=True, hash=False, repr=False)
    # The probability of ligand diffusion between cells
    cell_diffusions: jnp.ndarray = to.field(node=True, hash=False, repr=False)
    # Mask representing which genes are markers that don't belong to a cell type
    anti_marker_mask: jnp.ndarray = to.field(node=True, hash=False, repr=False)
    # The probability of gene interaction effects once a threshold is reached
    effect_probs: jnp.ndarray = to.field(node=True, hash=False, repr=False)
    # The threshold for gene interaction effects
    effect_thresholds: jnp.ndarray = to.field(node=True, hash=False, repr=False)
    # Gene interactions within cells
    gene_connectivities: jnp.ndarray = to.field(node=True, hash=False, repr=False)


@dataclass(eq=True, unsafe_hash=True)
class ExperimentData(to.Tree):
    """
    Cached information holder for an experiment/batches.
    """
    # The number of genes in the experiment
    n_genes: int = to.field(node=False, hash=False)
    # The number of cell types in the experiment
    n_cell_types: int = to.field(node=False, hash=False)
    # Mask representing which genes are ligands
    ligand_mask: jnp.ndarray = to.field(node=True, hash=False, repr=False)
    # Ligand/receptor pairs
    ligand_receptor_pairs: Tuple[Tuple[int, int]] = to.field(node=False, hash=False, repr=False)
    # Cell types to their marker genes
    cell_type_markers: Tuple[Tuple[int]] = to.field(node=False, hash=False, repr=False)
    # The locations of the batch boundaries in the counts data
    batch_indices: Tuple[int] = to.field(node=False, hash=True, repr=False)
    # Data related to each batch
    batches: Tuple[BatchData, ...] = to.field(node=True, hash=True)

    @property
    def n_batches(self):
        return len(self.batches)

    @property
    def batch_ranges(self) -> Iterator[Tuple[int, int]]:
        return zip(self.batch_indices[:-1], self.batch_indices[1:])


@dataclass(eq=True, unsafe_hash=True)
class SimulatedExperimentResults(Iterable, to.Tree):
    """
    Cached information holder for a simulated experiment.
    """
    # The timesteps saved from the simulation
    timesteps: Tuple[int] = to.field(node=False, hash=True)
    # The counts data
    counts: Tuple[jnp.ndarray] = to.field(node=True, hash=False, repr=False)
    # Input experiment data
    experiment_data: ExperimentData = to.field(node=True, hash=True)

    def get_counts(self, timestep: int) -> jnp.ndarray:
        return self.counts[self.timesteps.index(timestep)]

    def __iter__(self):
        return iter(zip(self.timesteps, self.counts))


def make_batch_matrices(n_cell_types: int,
                        batch_effect: float,
                        gene_backbones: List[nx.Graph],
                        cell_backbones: List[nx.Graph],
                        lr_pairs: List[Tuple[int, int]],
                        rng_key: StatefulPRNGKey) -> ExperimentData:
    batches = []
    batch_indices = jnp.cumsum(jnp.array([0, *(n.number_of_nodes() for n in cell_backbones)]), dtype=int)

    ligands = len({l for (l, r) in lr_pairs})
    # Update the genes/cells since the random generation may not have made the expected number
    # Note that the number should still be consistent between backbones so we can just select an arbitrary one
    n_genes = gene_backbones[0].number_of_nodes()

    max_marker_genes = int(0.1 * n_genes)  # Number of potential marker genes per cell type
    # n_celltypes x n_genes: Marks which genes are markers for given cell types
    marker_mask = jnp.zeros((n_cell_types, n_genes))
    celltype2markers = list()
    for label in range(n_cell_types):
        markers = jax.random.randint(rng_key(), jax.random.randint(rng_key(), (1,), 1, max_marker_genes+1), 0, n_genes)
        marker_mask = marker_mask.at[label, markers].set(1)
        celltype2markers.append(tuple(markers))
    ligand_mask = jnp.zeros((n_genes,))
    ligand_mask = ligand_mask.at[ligands].set(1)

    # Shared batch info
    cell_type_anti_marker_gene_mask = jnp.zeros((n_cell_types, n_genes), dtype=float)
    for cell_type in range(n_cell_types):
        cell_mask = marker_mask[cell_type, :]
        anti_cell_mask = (jnp.delete(marker_mask, cell_type, axis=0).sum(axis=0) > 0).astype(float)
        # Remove any genes that are markers in the current cell type as well
        anti_cell_mask -= (anti_cell_mask * cell_mask).astype(float)
        cell_type_anti_marker_gene_mask = cell_type_anti_marker_gene_mask.at[cell_type, :].set(anti_cell_mask)
    # Downweigh marker genes by 90% if they are in the wrong cell type
    cell_type_anti_marker_gene_mask = cell_type_anti_marker_gene_mask * .9

    for batch, cell_backbone in enumerate(cell_backbones):
        batched_gene_backbones = []
        for gene_backbone in gene_backbones:
            batched_backbone = gene_backbone.copy(as_view=False)
            _sparsify_graph(batched_backbone, _batch_effect_to_sparsity_factor(batch_effect, rng_key=rng_key), rng_key=rng_key)
            batched_gene_backbones.append(batched_backbone)

        n_cells = cell_backbone.number_of_nodes()
        cell_connectivity = nx.to_numpy_array(cell_backbone)  # Cell to cell interactions
        cell_diffusions = nx.to_numpy_array(cell_backbone, weight="diffusion")   # Probability of exocrine ligand signaling

        cell_type_indices = dict()
        for label in range(n_cell_types):
            cell_type_indices[label] = [n for n in cell_backbone.nodes if cell_backbone.nodes[n]['type'] == label]

        gene_connectivities = jnp.zeros((n_cells, n_genes, n_genes), dtype=float)
        effect_thresholds = jnp.zeros((n_cells, n_genes, n_genes), dtype=float)
        effect_probs = jnp.zeros((n_cells, n_genes, n_genes), dtype=float)
        # n_cell x n_gene: matrix that assigns ones to genes that are markers for non markers
        anti_marker_gene_mask = jnp.zeros((n_cells, n_genes), dtype=float)
        for i in range(n_cell_types):
            backbone = gene_backbones[i]
            gene_connectivities = gene_connectivities.at[cell_type_indices[i], :, :].set(nx.to_numpy_array(backbone, weight='connectivity'))  # -1 for inhibits, 1 for activates
            effect_thresholds = effect_thresholds.at[cell_type_indices[i], :, :].set(nx.to_numpy_array(backbone, weight='effect_threshold'))  # Min required expression
            effect_probs = effect_probs.at[cell_type_indices[i], :, :].set(nx.to_numpy_array(backbone, weight='weight'))  # Probability of effect if threshold met
            anti_marker_gene_mask = anti_marker_gene_mask.at[cell_type_indices[i], :].set(cell_type_anti_marker_gene_mask[i, :])
        del backbone

        batches.append(BatchData(
            n_cells=n_cells,
            cell_backbone=nx.freeze(cell_backbone),
            gene_backbones=tuple(map(nx.freeze, batched_gene_backbones)),
            cell_connectivity=cell_connectivity,
            cell_diffusions=cell_diffusions,
            anti_marker_mask=anti_marker_gene_mask,  # convert to a hashable array
            effect_probs=effect_probs,
            effect_thresholds=effect_thresholds,
            gene_connectivities=gene_connectivities
        ))

    return ExperimentData(
        n_genes=n_genes,
        n_cell_types=n_cell_types,
        ligand_mask=ligand_mask,
        ligand_receptor_pairs=tuple(lr_pairs),
        cell_type_markers=tuple(celltype2markers),
        batch_indices=tuple(batch_indices.astype(int).tolist()),
        batches=tuple(batches)
    )


def _simulate_counts(cell_simulation_iterations, n_labels, cell_backbones, gene_backbones, max_count, count_prob,
                     noise_prob, lr_pairs, batch_effect, dropout_prob,
                     rng_key: StatefulPRNGKey) -> SimulatedExperimentResults:
    experiment_info = make_batch_matrices(n_labels, batch_effect, gene_backbones, cell_backbones, lr_pairs, rng_key)

    stored_simulations = dict()
    simulated = None

    for timestep in tqdm(range(max(cell_simulation_iterations)), position=0, desc="Simulating Time Steps"):
        # Simulate
        simulated = _simulate_batches(simulated, max_count, count_prob, experiment_info, rng_key=rng_key)

        # Store simulations according to requested time points
        if (timestep+1) in cell_simulation_iterations:
            stored_simulations[timestep+1] = simulated

    print("Inserting Random Noise")
    for timestep in stored_simulations.keys():
        simulation = stored_simulations[timestep]
        for (batch_start, batch_end) in experiment_info.batch_ranges:
            batch_shape = (batch_end - batch_start, simulation.shape[1])
            # Add noise to the counts matrix
            simulation = simulation.at[batch_start:batch_end, :].add(parameterized_normal(0, noise_prob * max_count, shape=batch_shape, key=rng_key))
            # Ensure non-negative counts
            simulation = simulation.at[batch_start:batch_end, :].max(0)
            # Add random technical dropout
            dropout_mask = jax.random.bernoulli(rng_key(), 1-dropout_prob, shape=batch_shape)
            simulation = simulation.at[batch_start:batch_end, :].multiply(dropout_mask)
            stored_simulations[timestep] = simulation.__array__()  # Convert to numpy array

    return SimulatedExperimentResults(
        timesteps=tuple(stored_simulations.keys()),
        counts=tuple(stored_simulations.values()),
        experiment_data=experiment_info,
    )


@jax_vmap(in_axes=0, out_axes=0)  # Vmap allows us to "batch" the multiplication
@jax_jit()
def _multiply_cells_by_genes(cells, gene_effects):
    return cells * gene_effects

# TODO: GPU benchmarks
@consumes_key(4, "rng_key", count=2)
@jax_jit(static_argnums=(0, 1))
def _calculate_expression_transition(experiment_info: ExperimentData,
                                     batch_i: int,
                                     first_iter: bool,
                                     counts_matrix: jnp.array,
                                     rng_key: StatefulPRNGKey) -> jnp.array:
    batch: BatchData = experiment_info.batches[batch_i]
    batch_start = experiment_info.batch_indices[batch_i]
    batch_end = experiment_info.batch_indices[batch_i + 1]

    # Remove incorrect initial markers if this is the first iteration
    remove_markers_fun = lambda x: jnp.around(x * (1 - batch.anti_marker_mask))
    counts_matrix = counts_matrix.at[batch_start:batch_end, ].set(lax.cond(first_iter, remove_markers_fun, lambda x: x, counts_matrix[batch_start:batch_end, ]))

    # Finally, simulations
    # Get Ligand expression only
    ligand_exp_mat = counts_matrix[batch_start:batch_end, ] * experiment_info.ligand_mask
    # Don't allow diffusion between self (would unnaturally increase ligand concentration)
    ligand_exp_mat = ligand_exp_mat.at[jnp.diag_indices(ligand_exp_mat.shape[0])].set(0)  # np.fill_diag(ligand_exp_mat, 0)
    # Random diffusion per cell/genes
    diffusion_mask = jnp.less(jax.random.uniform(rng_key[0], batch.cell_diffusions.shape), batch.cell_diffusions).astype(float)
    # Add surrounding ligands to counts
    effective_counts = counts_matrix[batch_start:batch_end, ] + (batch.cell_connectivity @ diffusion_mask @ ligand_exp_mat)
    del ligand_exp_mat
    del diffusion_mask

    effect_mask = jax.random.uniform(rng_key[1], batch.gene_connectivities.shape)  # Random draw
    effect_mask = jnp.less(effect_mask, batch.effect_probs)  # Check which effects were randomly selected

    # Get counts for each of the selected effects
    effects = _multiply_cells_by_genes(effective_counts, effect_mask)
    #effects = jnp.einsum("ij,ijk->ijk", effective_counts, effect_mask)
    del effective_counts
    del effect_mask

    # Select effects that meet a threshold
    effects = ((effects - batch.effect_thresholds) > 0)
    # Set the directionality of the effects
    effects = effects * batch.gene_connectivities
    # Add the effects together to get a n_cell x n_gene total effects matrix
    effects = effects.sum(axis=2)
    # Down weigh misplaced cell type markers
    effects = effects * (1 - batch.anti_marker_mask)

    # Finally, add the effects to the real counts matrix to update it
    counts_matrix = counts_matrix.at[batch_start:batch_end, ].add(jnp.around(effects))
    del effects

    # Make sure the update didnt introduce negative values
    counts_matrix = counts_matrix.at[batch_start:batch_end, ].max(0)
    return counts_matrix


def _simulate_batches(curr_counts, max_count, count_prob, experiment_info: ExperimentData, rng_key: StatefulPRNGKey):
    first_iter = curr_counts is None
    if first_iter:
        counts_matrix = negative_binomial(max_count, count_prob, shape=(experiment_info.batch_indices[-1], experiment_info.n_genes), key=rng_key).astype(float)
    else:
        counts_matrix = curr_counts.copy()

    for batch_i in tqdm(range(experiment_info.n_batches), position=1, desc="Simulating Batches", leave=False):
        counts_matrix = _calculate_expression_transition(experiment_info, batch_i, first_iter, counts_matrix, rng_key)

    return counts_matrix


def _simulations_to_adatas(simulation_results: SimulatedExperimentResults) -> Tuple[ad.AnnData, Tuple[ad.AnnData, ...]]:
    adatas = []
    ligands = {l for l,r in simulation_results.experiment_data.ligand_receptor_pairs}
    receptors = {r for l,r in simulation_results.experiment_data.ligand_receptor_pairs}
    for simulation, matrix in simulation_results:
        n_genes = matrix.shape[1]
        adata = ad.AnnData(matrix.astype(jnp.float32))
        # Metadata
        adata.uns['simulation'] = True
        adata.uns['last_timepoint'] = simulation
        adata.uns['lr_pairs'] = simulation_results.experiment_data.ligand_receptor_pairs
        adata.obs['simulation_timepoint'] = jnp.repeat(simulation, matrix.shape[0]).astype(int)
        adata.obs['batch'] = jnp.concatenate([
            jnp.repeat(i, batch_end-batch_start) for i, (batch_start, batch_end) in enumerate(simulation_results.experiment_data.batch_ranges)
        ])
        adata.obs['true_labels'] = jnp.array([
            simulation_results.experiment_data.batches[batch].cell_backbone.nodes[n]['type'] for (batch, n) in zip(adata.obs['batch'],
                                                                                                   itertools.chain.from_iterable(b.cell_backbone.nodes for b in simulation_results.experiment_data.batches))
        ], dtype=int)
        adata.var['is_ligand'] = [(g in ligands) for g in range(n_genes)]
        adata.var['is_receptor'] = [(g in receptors) for g in range(n_genes)]
        for ct, markers in enumerate(simulation_results.experiment_data.cell_type_markers):
            adata.var[f'is_label_{ct}_marker'] = [(g in markers) for g in range(n_genes)]

        adatas.append(adata)

    # Combine selected adatas to emulate pseudotime
    full_adata = ad.concat(adatas[::-1],  # Reverse so latest time is prioritized
                        merge="first",
                        uns_merge='first')
    full_adata.uns['last_timepoint'] = 'All'

    # Common pre-processing
    for adata in (adatas + [full_adata]):
        # Annotate cells by clustering to replicate unclear cell types
        sc.tl.pca(adata)
        # Prep for clustering
        sc.pp.neighbors(adata, n_pcs=40)
        # Cluster
        sc.tl.leiden(adata)
        sc.tl.louvain(adata)
        sc.tl.paga(adata)
        # Annotate cell type by clustering to replicate uncertain labelling
        adata.obs['cell_type'] = adata.obs['louvain']

    return full_adata, tuple(adatas)


def simulate_counts(
    # Ideal number of genes
    n_genes: int = 250,
    # Ideal number of cells per batch
    n_cells: int = 400,
    # Number of cell types
    n_labels: int = 5,
    # Number of batches
    n_batches: int = 5,
    # Time points to examine simulations for
    cell_simulation_iterations: Union[int, List[int]] = [1, 10, 25, 50, 100],
    # Whether to produce intermediate plots
    plot: bool = True,
    # If specified, save plots to the figures/ directory
    save_plots: bool = True,

    # Maximum background counts number
    max_count: int = 200,
    # Random expectation of the probability of counts out of the max counts
    count_prob: float = 0.5,
    # Expectation of the probability of random technical dropout
    dropout_prob: float = 0.3,
    # Proportion controlling the level of gaussian noise to apply to the final counts
    noise_prob: float = 0.25,
    # Scales batch effects
    batch_effect: float = .75,
    # Random seed
    seed: int = 0
) -> SimulatedExperimentResults:
    key = StatefulPRNGKey(seed)

    if isinstance(cell_simulation_iterations, int):
        cell_simulation_iterations = [cell_simulation_iterations]
    # Make sure its from lowest to largest
    cell_simulation_iterations.sort()

    # Generate the initial gene backbone
    base_gene_backbone, lr_pairs = _generate_gene_backbone(n_genes, .25, rng_key=key)

    ligands = {l for (l, r) in lr_pairs}
    receptors = {r for (l, r) in lr_pairs}
    title = f"Ligands: {len(ligands)}, Receptors: {len(receptors)}, Other Genes: {n_genes - len(ligands) - len(receptors)}"
    print(title)

    if plot:
        _draw_network(base_gene_backbone.reverse(), title, save=save_plots, filename="base_gene_backbone")

    # Progressively permute the network to make cell types more different, but still related
    gene_backbones = [None] * n_labels
    rand_idx = jax.random.permutation(key(), n_labels)  # Make the cell type relationships random
    backbone = base_gene_backbone
    for i in range(n_labels):
        backbone = _generate_network_from_backbone(backbone, lr_pairs, jax.random.uniform(key()), max_count, key)
        gene_backbones[rand_idx[i]] = backbone

    del backbone
    del rand_idx

    if plot:
        for i, backbone in enumerate(gene_backbones):
            _draw_network(backbone.reverse(), f"Gene Backbone for: {i}", save=save_plots, filename=f"gene_backbone_{i}")

    # Probability of connections between types
    cell_type_interaction_probs = jnp.array(
        [_randomized_proportions(n_labels, prioritize_group=i, rng_key=key).tolist() for i in range(n_labels)])

    cell_backbones = []
    for batch in range(n_batches):
        backbone = _generate_cell_backbone(n_labels, n_cells, batch_effect, cell_type_interaction_probs, rng_key=key)
        cell_backbones.append(backbone)
        if plot:
            _draw_network(backbone, f"Batch {batch} Backbone", colors=plt.cm.tab10.colors,
                          layout=nx.kamada_kawai_layout, save=save_plots, filename=f"cell_backbone_{batch}")

    simulations = _simulate_counts(cell_simulation_iterations, n_labels,
                                    cell_backbones, gene_backbones,
                                    max_count, count_prob,
                                    noise_prob, lr_pairs,
                                    batch_effect, dropout_prob,
                                    key)

    print("Annotating cells")
    full_adata, adatas = _simulations_to_adatas(simulations)

    if plot:
        for adata in (adatas + (full_adata,)):
            timepoint = adata.uns['last_timepoint']

            sc.pl.pca(adata, color=["true_labels", "batch"], title=[f'PCA at t={timepoint} (colored by true labels)',
                                                                    f'PCA at t={timepoint} (colored by batch)'],
                      show=not save_plots, save=f"_t{timepoint}.png" if save_plots else None)

            # Plot UMAP
            sc.pl.paga(adata, plot=True, title=f"PAGA network at t={timepoint}",
                       show=not save_plots is None, save=f"_t{timepoint}.png" if save_plots else None)
            sc.tl.umap(adata, init_pos='paga')
            sc.pl.umap(adata, color=["true_labels", "batch"], title=[f"UMAP at t={timepoint} (colored by true labels)",
                                                                     f"UMAP at t={timepoint} (colored by batch)"],
                       show=not save_plots is None, save=f"_labels_t{timepoint}.png" if save_plots else None)
            sc.pl.umap(adata, color=['leiden', 'louvain'], title=[f"UMAP at t={timepoint} (colored by leiden clusters)",
                                                                  f"UMAP at t={timepoint} (colored by louvain clusters)"],
                       show=not save_plots is None, save=f"_cluster_t{timepoint}.png" if save_plots else None)


if __name__ == "__main__":
    from time import time
    start_time = time()
    simulate_counts()
    print("END", time() - start_time)
