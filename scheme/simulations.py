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

from scheme.cells import simulate_cells
from scheme.data import write_anndata, \
    select_lr_pairs, simulation_to_adata
from scheme.experiment import simulate_batch_effect
from scheme.networks import duplication_divergence_graph, annotate_cell_type_specific_gene_modules, produce_celltype_perturbed_gene_backbone, \
    gene_networks_to_tensor
from scheme.plotting import _draw_network, _make_simulated_adata_plots, _draw_voronoi_slice
from scheme.util import StatefulPRNGKey


def generate_gene_networks(n_genes: int,
                           n_labels: int,
                           p_network_mutation: float,
                           p_network_functional_mutation: float,
                           key: StatefulPRNGKey) -> Tuple[List[Tuple[int, int]], nx.DiGraph, List[nx.DiGraph]]:
    # Generate the initial gene backbone annotated with type (ligand, receptor, gene)
    base_gene_backbone = duplication_divergence_graph(n_genes, p_network_mutation, p_network_functional_mutation, key)
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
    n_cells: int = 1000,
    # Number of cell types
    n_celltypes: int = 5,
    # Number of batches
    n_batches: int = 4,
    # Model hyperparameters
    # Number of simulation steps
    n_steps: int = 100,
    # The max count of random background counts
    max_expression: int = 100,
    # The probability of a "mutation" occuring during gene network duplications
    p_network_mutation = 0.4,
    # The probability of a gene being annotated as a receptor or ligand during divergence
    p_gene_label = 0.15,
    # The probability of a random background gene being expressed
    p_expression = 0.15,
    # The probability of a random protein being degraded
    p_degradation = 0.1,
    # The probability of a cell dying
    p_death = 0.1,
    # Strength of dropout batch effect
    p_dropout = 0.1,
    # Strength of pcr bias batch effect
    pcr_bias_strength = 0.1,
    # Whether to produce intermediate plots
    plot: bool = True,
    # If specified, save plots to the figures/ directory
    save_plots: bool = True,
    # Random seed
    seed: int = 0
) -> List[List[ad.AnnData]]:
    key = StatefulPRNGKey(seed)

    # Generate gene regulatory networks
    lr_pairs, base_gene_backbone, gene_backbones = generate_gene_networks(n_genes, n_celltypes, p_network_mutation, p_gene_label, key)
    # Convert to matrices
    gene_info, gene_network_info = gene_networks_to_tensor(base_gene_backbone, gene_backbones)

    # Get information about the gene networks
    ligands = {l for (l, r) in lr_pairs}
    receptors = {r for (l, r) in lr_pairs}
    title = f"Ligands: {len(ligands)}, Receptors: {len(receptors)}, Other Genes: {n_genes - len(ligands) - len(receptors)}"
    print(title)

    if plot:
        _draw_network(base_gene_backbone.reverse(), title, save=save_plots, filename="base_gene_backbone")

        for i, backbone in enumerate(gene_backbones):
            _draw_network(backbone.reverse(), f"Gene Backbone for: {i}", save=save_plots, filename=f"gene_backbone_{i}")

    # Generate the counts matrices
    batch2spatial = []
    batch2counts = []
    for batch in tqdm(range(n_batches), 'Simulating batches...', total=n_batches):
        spatial, counts = simulate_cells(
            n_cells, n_celltypes,
            n_genes, p_degradation,
            p_death,
            p_expression, max_expression,
            n_steps,
            lr_pairs, gene_info,
            gene_network_info,
            key=key
        )

        if plot:
            for t, spatial_mat in enumerate(spatial):
                # Calculate true t
                t = 10 * (t + 1)
                _draw_voronoi_slice(spatial_mat, 3, title=f"Voronoi slice at t={t}")

        batch2spatial.append(spatial)
        batch2counts.append(counts)

    batch2adatas = []
    # Compile the simulation results
    for batch in tqdm(range(n_batches), 'Compiling simulation results...', total=n_batches):
        spatial = batch2spatial[batch]
        counts = batch2counts[batch]
        adatas = []

        for t, (spatial_mat, counts_mat) in enumerate(zip(spatial, counts)):
            # Calculate true t
            t = 10 * (t+1)

            if p_dropout > 0 or pcr_bias_strength > 0:
                counts_mat = simulate_batch_effect(counts_mat, p_dropout, pcr_bias_strength, key)

            adata = simulation_to_adata(counts_mat, spatial_mat, lr_pairs)
            adata.obs['batch'] = batch
            adata.uns['timestep'] = t
            adatas.append(adata)

            if plot:
                _make_simulated_adata_plots(adata, save=save_plots)

            if not osp.exists("adatas/"):
                os.makedirs("adatas/")

            write_anndata(adata, osp.join("adatas/", f"b{batch}_t{t}.h5ad"))

        batch2adatas.append(adatas)

    return batch2adatas


if __name__ == "__main__":
    from time import time
    start_time = time()
    res = simulate_counts(
        n_genes=250,
        n_cells=1000,
    )
    print("END", time() - start_time)
