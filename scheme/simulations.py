import os
from typing import List, Tuple, Union, Optional
import os.path as osp

import anndata as ad
import networkx as nx
# Note that we are using jax to speed up calculations
import jax.numpy as jnp
import jax
from jax import lax
from matplotlib import pyplot as plt
from tqdm import tqdm

from scheme.data import ExperimentData, BatchData, SimulatedExperimentResults, _simulations_to_adatas, write_anndata, \
    select_lr_pairs
from scheme.networks import _sparsify_graph, \
    duplication_divergence_graph, annotate_cell_type_specific_gene_modules, produce_celltype_perturbed_gene_backbone
from scheme.plotting import _draw_network, _make_simulated_adata_plots
from scheme.util import StatefulPRNGKey, bimodal_normal, parameterized_normal, \
    negative_binomial, jax_jit, consumes_key, jax_vmap, lognormal, _randomized_proportions
from scheme.voronoi import jump_flooding


@consumes_key(1, 'rng_key')
@jax_jit()
def _batch_effect_to_sparsity_factor(batch_effect, rng_key: StatefulPRNGKey):
    return (jnp.minimum(batch_effect, 1.5)) * jax.random.uniform(rng_key)/4


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


# TODO: Optimize by removing the for loops
def _generate_cell_backbone_from_voronoi(n_labels,
                                         n_cells,
                                         batch_effect,
                                         cell_type_interaction_probs,
                                         rng_key: StatefulPRNGKey,
                                         max_seeds_per_type: int = 3):
    # 3D matrix, so we need to take the 3rd root of n_cells
    matrix_dimension = int(n_cells**(1/3))
    # Generate number of random seeds for each cell type
    n_seeds = jax.random.randint(rng_key(), (n_labels,), 1, max_seeds_per_type)
    # Make mapping of index to cell type by taking the cumulative sum
    celltypes2index = jnp.cumsum(n_seeds)

    # Generate voronoi topology
    voronoi = jump_flooding(matrix_dimension, jnp.sum(n_seeds, dtype=jnp.integer).item(), rng_key=rng_key)

    # Overwrite seed numbers with corresponding cell type
    for celltype, max_index in enumerate(celltypes2index):
        last_max_index = 0 if celltype == 0 else celltypes2index[celltype - 1]
        voronoi = jnp.where((voronoi >= last_max_index) & (voronoi < max_index), celltype, voronoi)

    # Generate cell backbone by building network from voronoi topology and cell type interaction probabilities and exponential decay
    cell_backbone = nx.DiGraph()
    for i in range(matrix_dimension):
        for j in range(matrix_dimension):
            for k in range(matrix_dimension):
                celltype = voronoi[i, j, k].item()
                # Add node to graph
                cell_label = f"{i}_{j}_{k}"
                cell_backbone.add_node(cell_label, type=celltype)

    # Log-normal distance for ligand emission
    ligand_emission_distances = lognormal(0.25, .25, (matrix_dimension, matrix_dimension, matrix_dimension), key=rng_key)
    # Round to integers
    ligand_emission_distances = jnp.round(ligand_emission_distances).astype(jnp.int32)

    # Populate edges
    for i in range(matrix_dimension):
        for j in range(matrix_dimension):
            for k in range(matrix_dimension):
                curr_celltype = voronoi[i, j, k]
                curr_cell_label = f"{i}_{j}_{k}"
                emission_distance = ligand_emission_distances[i, j, k]
                # Add edges to all cells within emission distance
                offsets = jnp.arange(-emission_distance, emission_distance + 1)
                all_coord_offsets = jnp.array(jnp.meshgrid(offsets, offsets, offsets, indexing='ij')).T.reshape(-1, 3)
                for coord_offset in all_coord_offsets:
                    neighbor_coord = jnp.array([i, j, k]) + coord_offset
                    # Skip out of bounds
                    if jnp.any(neighbor_coord < 0) or jnp.any(neighbor_coord >= matrix_dimension):
                        continue
                    neighbor_celltype = voronoi[neighbor_coord[0], neighbor_coord[1], neighbor_coord[2]]
                    neighbor_cell_label = f"{neighbor_coord[0]}_{neighbor_coord[1]}_{neighbor_coord[2]}"
                    diffusion = cell_type_interaction_probs[curr_celltype, neighbor_celltype].item()
                    cell_backbone.add_edge(curr_cell_label, neighbor_cell_label,
                                           diffusion=diffusion, weight=diffusion)

    # Replace string nodes with 0-indexed integers
    cell_backbone = nx.convert_node_labels_to_integers(cell_backbone, label_attribute='label')

    # Batch effect disrupting connections randomly
    _sparsify_graph(cell_backbone, _batch_effect_to_sparsity_factor(batch_effect, rng_key=rng_key), rng_key=rng_key)

    return cell_backbone


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


def _generate_gene_networks(n_genes: int,
                            n_labels: int,
                            key: StatefulPRNGKey) -> Tuple[List[Tuple[int, int]], nx.DiGraph, List[nx.DiGraph]]:
    # Generate the initial gene backbone annotated with type (ligand, receptor, gene)
    base_gene_backbone = duplication_divergence_graph(n_genes, .4, .05, key)
    # Annoate with random cell-type specific modules (under cell_type, if None then non-specific)
    base_gene_backbone = annotate_cell_type_specific_gene_modules(base_gene_backbone, n_labels, key)
    lr_pairs = select_lr_pairs(base_gene_backbone)

    #base_gene_backbone, lr_pairs = _generate_gene_backbone(n_genes, .25, rng_key=key)
    # Generate the gene backbones for each cell type
    gene_backbones = []
    for i in range(n_labels):
        gene_backbones.append(produce_celltype_perturbed_gene_backbone(base_gene_backbone, i, key))

    return lr_pairs, base_gene_backbone, gene_backbones


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

    lr_pairs, base_gene_backbone, gene_backbones = _generate_gene_networks(n_genes, n_labels, key)

    ligands = {l for (l, r) in lr_pairs}
    receptors = {r for (l, r) in lr_pairs}
    title = f"Ligands: {len(ligands)}, Receptors: {len(receptors)}, Other Genes: {n_genes - len(ligands) - len(receptors)}"
    print(title)

    if plot:
        _draw_network(base_gene_backbone.reverse(), title, save=save_plots, filename="base_gene_backbone")

        for i, backbone in enumerate(gene_backbones):
            _draw_network(backbone.reverse(), f"Gene Backbone for: {i}", save=save_plots, filename=f"gene_backbone_{i}")

    cell_backbones = []
    for batch in range(n_batches):
        # Probability of connections between types
        cell_type_interaction_probs = jnp.array([_randomized_proportions(n_labels, prioritize_group=i, rng_key=key).tolist() for i in range(n_labels)])
        backbone = _generate_cell_backbone_from_voronoi(n_labels, n_cells, batch_effect, cell_type_interaction_probs, rng_key=key)
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
    return simulations


def compile_experiment_data(
        simulation_results: SimulatedExperimentResults,
        output_dir: str = None,
        plot: bool = True,
        save_plots: bool = True) -> Tuple[ad.AnnData, Tuple[ad.AnnData, ...]]:
    print("Annotating cells")
    full_adata, adatas = _simulations_to_adatas(simulation_results)

    if plot:
        for adata in (adatas + (full_adata,)):
            _make_simulated_adata_plots(adata, save=save_plots)

    if output_dir:
        if not osp.exists(output_dir):
            os.makedirs(output_dir)
        write_anndata(full_adata, osp.join(output_dir, "full_adata.h5ad"))
        for adata in adatas:
            write_anndata(adata, osp.join(output_dir, f"t{adata.uns['last_timepoint']}.h5ad"))

    return full_adata, adatas


if __name__ == "__main__":
    from time import time
    start_time = time()
    res = simulate_counts(
        n_genes=250,
        n_cells=1000,
    )
    compile_experiment_data(res, output_dir="simulated_data")
    print("END", time() - start_time)
