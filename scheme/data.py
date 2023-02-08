import itertools
from dataclasses import dataclass
from typing import Tuple, Iterator, Iterable, List

import anndata as ad
import numpy as np
import scanpy as sc
import jax.numpy as jnp
import treeo as to
import networkx as nx


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


def select_lr_pairs(gene_backbone: nx.DiGraph) -> List[Tuple[int, int]]:
    lr_pairs = []
    for (u, v) in gene_backbone.edges:  # FIXME: Replace type checks with is_ligand/is_receptor
        if gene_backbone.nodes[u]['is_ligand'] and gene_backbone.nodes[v]['is_receptor']:
            lr_pairs.append((u, v))
    return lr_pairs


def _simulations_to_adatas(simulation_results: SimulatedExperimentResults) -> Tuple[ad.AnnData, Tuple[ad.AnnData, ...]]:
    adatas = []
    ligands = {l for l, r in simulation_results.experiment_data.ligand_receptor_pairs}
    receptors = {r for l, r in simulation_results.experiment_data.ligand_receptor_pairs}
    for simulation, matrix in simulation_results:
        n_genes = matrix.shape[1]
        adata = ad.AnnData(np.asarray(matrix.astype(jnp.float32)))
        # Metadata
        adata.uns['simulation'] = True
        adata.uns['last_timepoint'] = simulation
        adata.uns['ligands'] = [l for l, r in simulation_results.experiment_data.ligand_receptor_pairs]
        adata.uns['receptors'] = [r for l, r in simulation_results.experiment_data.ligand_receptor_pairs]
        adata.obs['simulation_timepoint'] = np.asarray(jnp.repeat(simulation, matrix.shape[0]).astype(int)).tolist()
        adata.obs['batch'] = np.asarray(jnp.concatenate([
            jnp.repeat(i, batch_end - batch_start) for i, (batch_start, batch_end) in
            enumerate(simulation_results.experiment_data.batch_ranges)
        ])).tolist()
        adata.obs['true_labels'] = np.asarray(jnp.array([
            simulation_results.experiment_data.batches[batch].cell_backbone.nodes[n]['type'] for (batch, n) in
            zip(adata.obs['batch'],
                itertools.chain.from_iterable(
                    b.cell_backbone.nodes for b in simulation_results.experiment_data.batches))
        ], dtype=int)).tolist()
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


def write_anndata(adata: ad.AnnData, file: str):
    """
    Write an AnnData object to a file with standardized compression schemes.
    """
    adata.write_h5ad(file,
                     compression='gzip',
                     compression_opts=9)
