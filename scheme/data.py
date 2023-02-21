import itertools
from dataclasses import dataclass
from typing import Tuple, Iterator, Iterable, List

import anndata as ad
import numpy as np
import scanpy as sc
import jax.numpy as jnp
import treeo as to
import networkx as nx


def select_lr_pairs(gene_backbone: nx.DiGraph) -> List[Tuple[int, int]]:
    lr_pairs = []
    for (u, v) in gene_backbone.edges:  # FIXME: Replace type checks with is_ligand/is_receptor
        if gene_backbone.nodes[u]['is_ligand'] and gene_backbone.nodes[v]['is_receptor']:
            lr_pairs.append((u, v))
    return lr_pairs


def simulation_to_adata(counts_mat: jnp.array, spatial_mat: jnp.array, lr_pairs: List[Tuple[int, int]]) -> ad.AnnData:
    ligands = {l for l, r in lr_pairs}
    receptors = {r for l, r in lr_pairs}
    from scheme.cells import _project_voronoi_to_expression_positions
    spatial_indices = _project_voronoi_to_expression_positions(spatial_mat)[0]
    n_genes = counts_mat.shape[1]
    adata = ad.AnnData(counts_mat.astype(jnp.float32).__array__())
    # Metadata
    adata.uns['simulation'] = True
    adata.uns['ligands'] = [l for l, r in lr_pairs]
    adata.uns['receptors'] = [r for l, r in lr_pairs]
    adata.obs['true_labels'] = spatial_mat[spatial_indices[:, 0], spatial_indices[:, 1], spatial_indices[:, 2]]\
        .astype(jnp.int32).__array__().tolist()
    adata.var['is_ligand'] = [(g in ligands) for g in range(n_genes)]
    adata.var['is_receptor'] = [(g in receptors) for g in range(n_genes)]

    # Common pre-processing
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

    return adata


def write_anndata(adata: ad.AnnData, file: str):
    """
    Write an AnnData object to a file with standardized compression schemes.
    """
    adata.write_h5ad(file,
                     compression='gzip',
                     compression_opts=9)
