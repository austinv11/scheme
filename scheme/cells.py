import jax
import jax.numpy as jnp

from scheme.data import SingleCellSimulationResults
from scheme.util import StatefulPRNGKey, DEFAULT_RNG_KEY
from scheme.voronoi import jump_flooding


def initialize_cells(n_cells: int,
                     n_celltypes: int,
                     max_voronoi_seeds_per_celltype: int,
                     key: StatefulPRNGKey) -> jnp.array:
    # We will use chebyshev distance to determine the voronoi tessellations for smoother borders
    minkowski_distance_parameter = jnp.inf
    # Calculate the 3d dimensions of the cube that will contain all cells
    cube_dimension = int(n_cells ** (1/3))
    # Number of voronoi seeds per cell type
    n_seeds = jax.random.randint(key(), (n_celltypes,), minval=1, maxval=max_voronoi_seeds_per_celltype+1)
    # Make an indexing array to determine cell types -> voronoi seeds
    celltypes2index = jnp.cumsum(n_seeds)

    # Perform the tesselation
    voronoi = jump_flooding(cube_dimension, jnp.sum(n_seeds, dtype=jnp.integer).item(), key, p=minkowski_distance_parameter)

    # Map the voronoi seeds to cell types
    for celltype, max_index in enumerate(celltypes2index):
        last_max_index = 0 if celltype == 0 else celltypes2index[celltype - 1]
        voronoi = jnp.where((voronoi >= last_max_index) & (voronoi < max_index), celltype, voronoi)

    return voronoi


def simulate_cells(
        n_cells: int,
        n_celltypes: int,
        max_initial_celltype_colonies: int = None,
        key: StatefulPRNGKey = DEFAULT_RNG_KEY,
) -> SingleCellSimulationResults:
    """
    Generate a spatial distribution of cells and simulate them given a GRN.

    :param n_cells: The target number of cells to simulate (can be reduced if there is no integer cubic root).
    :param n_celltypes: The number of cell types to simulate.
    :param max_initial_celltype_colonies: The maximum number of initial cell colonies to begin initial growth from.
        Higher numbers yield more heterogeneity. If not specified, will be set to the cube root of n_cells.
    :param key: The random number generator key.
    :return: The final expression matrix, and intermediate data.
    """
    # Handle default parameters
    if max_initial_celltype_colonies is None:
        max_initial_celltype_colonies = int(n_cells ** (1/3))

    # Initialize cells
    spatial_cell_distribution = initialize_cells(n_cells,
                                                 n_celltypes,
                                                 max_initial_celltype_colonies,
                                                 key)



    return SingleCellSimulationResults(

    )
