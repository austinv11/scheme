from typing import Iterable, Tuple

import jax
import jax.numpy as jnp

from scheme.data import SingleCellSimulationResults
from scheme.util import StatefulPRNGKey, DEFAULT_RNG_KEY, jax_jit, consumes_key, exponential, negative_binomial, \
    jax_vmap, swap_elements, swap_rows
from scheme.voronoi import jump_flooding


def initialize_cells(n_cells: int,
                     n_celltypes: int,
                     max_voronoi_seeds_per_celltype: int,
                     key: StatefulPRNGKey) -> jnp.array:
    """
    Generate a 3D map of cell types using a voronoi tessellation to create smooth pockets of cell groups.
    The resultant array is NxNxN where N is the cube root of the number of cells and the values are cell type annotations.
    """
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


@jax_jit()
def _move_cell_step_iteration(voronoi_map: jnp.array,
                    expression_matrix: jnp.array,
                    movement_distance_3d: jnp.array,
                    movement_list: jnp.array,
                    movement_list_offset: int) -> Tuple[jnp.array, jnp.array, jnp.array, jnp.array, int]:
    """
    Handle a single step of cell movement.
    Given a 3D voronoi map, associated 2D expression matrix, list of remaining movement steps for cells
    (using 3D coordinates), and a pre-generated list of movement vectors, move cells in the 3D map and
    update the expression matrix accordingly.
    """
    # Get first cell position to move
    cell_position = jnp.argwhere(movement_distance_3d > 0)[0]
    remaining_distance = movement_distance_3d[cell_position[0], cell_position[1], cell_position[2]]
    # Update the distance
    movement_distance_3d = movement_distance_3d.at[cell_position[0], cell_position[1], cell_position[2]].set(remaining_distance - 1)
    # Get the movement
    movement = movement_list[movement_list_offset]
    # Update the movement list offset
    movement_list_offset += 1
    # Get the destination position
    destination_position = cell_position + movement
    # Clip to range
    destination_position = jnp.clip(destination_position, 0, voronoi_map.shape[0]-1)
    # Swap cell positions in 3d space
    voronoi_map = swap_elements(voronoi_map, cell_position, destination_position)
    movement_distance_3d = swap_elements(movement_distance_3d, cell_position, destination_position)
    # Get 1-d index of cell position
    cell_index = _position_to_cell_index(cell_position, voronoi_map.shape)
    # Get 1-d index of destination position
    destination_index = _position_to_cell_index(destination_position, voronoi_map.shape)
    # Swap cell positions in expression matrix
    expression_matrix = swap_rows(expression_matrix, cell_index, destination_index)

    return voronoi_map, expression_matrix, movement_distance_3d, movement_list, movement_list_offset


@consumes_key(5, 'key', count=3)
@jax_jit()
def cell_movement_step(voronoi_map: jnp.array,
                       expression_matrix: jnp.array,
                       cell_type_mobility_propensities: jnp.array,
                       cell_type_mobility_range: jnp.array,
                       p_move: float,
                       key: StatefulPRNGKey) -> Tuple[jnp.array, jnp.array]:
    """
    Given the Spatial map of cell types and the corresponding expression matrix, randomly move cells.
    Additionally given is the propensity of each cell type to move (CT x 1 matrix) and the average rang
    of each cell type when moving (CT x 1 matrix). The p_move parameter defines the random probability of movement.
    """
    # Label cells by propensity to move
    n_celltypes = cell_type_mobility_propensities.shape[0]
    # For each cell type, select the celltype mask
    movement_propensities = jax.lax.fori_loop(0, n_celltypes, lambda i, x: (x + jnp.full(voronoi_map.shape, cell_type_mobility_propensities[i])), jnp.zeros_like(voronoi_map))
    # Normalize the movement propensities to be independent probabilities
    movement_propensities = movement_propensities / jnp.sum(movement_propensities)
    # Then multiply by the probability of moving
    movement_propensities = movement_propensities * p_move
    # Select cells to move by using propensities as a bernoulli mask
    movement_mask = jax.random.bernoulli(key[0], movement_propensities)
    # Get the coordinates of the cells to move
    movement_positions = jnp.argwhere(movement_mask)
    # Calculate the distance for each cell to move
    # First get mean distance for each cell type, then get random distance
    movement_distances = jax.random.poisson(key[1], cell_type_mobility_range[voronoi_map[movement_positions[:, 0], movement_positions[:, 1], movement_positions[:, 2]]])
    # Make 3D array to represent the movement distances
    movement_distances_3d = jnp.zeros_like(voronoi_map)
    movement_distances_3d = movement_distances_3d.at[movement_positions[:, 0], movement_positions[:, 1], movement_positions[:, 2]].set(movement_distances)

    # Move cells
    # Get total number of movements to allow for static mapping
    n_movements = jnp.sum(movement_distances)
    # Get the random movements in 3D space
    movement_list = jax.random.randint(key[2], (n_movements, 3), -1, 2)
    voronoi_map, expression_matrix, _, _, _ = jax.lax.while_loop(
        lambda _, __, movement_distances_3d, ___, ____: jnp.sum(movement_distances_3d) > 0,
        _move_cell_step_iteration,
        (voronoi_map, expression_matrix, movement_distances_3d, movement_list, 0)
    )

    return voronoi_map, expression_matrix


@jax_jit()
def _diffuse_ligand(index: int,
                    spatial_ligand_matrix: jnp.array,
                    diffusion_positions_list: jnp.array,
                    diffusion_vector_list: jnp.array):
    """
    Diffuse ligand concentration into space. Index refers to the iteration number for the pre-computed positions.
    The spatial_ligand_matrix is an NxNxNxL matrix describing spatial ligand concentrations. The diffusion_positions_list
    and diffusion_vector_list are the pre-computed positions and vectors for position offsets, respectively.
    """
    initial_position = diffusion_positions_list[index,]
    diffusion_vector = diffusion_vector_list[index,]
    # Add ligand_dimension to diffusion vector by adding placeholder 0 axis
    diffusion_vector = diffusion_vector[:, jnp.newaxis]
    # Get final position
    final_position = initial_position + diffusion_vector
    # Round to nearest integer
    final_position = jnp.round(final_position).astype(jnp.int32)
    # Clip to range
    final_position = jnp.clip(final_position, 0, spatial_ligand_matrix.shape)
    # Move ligand
    spatial_ligand_matrix = spatial_ligand_matrix.at[final_position].add(1)
    spatial_ligand_matrix = spatial_ligand_matrix.at[initial_position].add(-1)

    return spatial_ligand_matrix, diffusion_positions_list, diffusion_vector_list


@jax_jit()
def _celltype_expression_step(celltype: int,
                              flat_celltype_mask: jnp.array,
                              expression_matrix: jnp.array,
                              diffused_ligands: jnp.array,
                              gene_info: jnp.array,
                              gene_network_info: jnp.array):
    """
    Handle the expression for a given cell type.
    Given are: a mask of cell type labels for rows in the expression matrix (N x 1), the expression matrix (N x G),
    The expression matrix that only represents the diffused ligand concentrations (N x G), gene metadata (G x 2), and
    gene network interactions (CT x G x G).
    """
    # Filter to specific cell type
    cell_selector = jnp.argwhere(flat_celltype_mask == celltype)
    # Select the effective expression matrix for the cell type
    celltype_expression = expression_matrix[cell_selector, :] + diffused_ligands[cell_selector, :]
    # Select the gene network info for the cell type
    celltype_gene_network_info = gene_network_info[celltype, :, :]
    is_ligand = gene_info[0, :] > 0
    is_receptor = gene_info[1, :] > 0

    # Transition step

    # Delete non-engaged receptors to prevent activation, first make masks for l/rs
    ligand_mask = celltype_expression @ is_ligand
    receptor_mask = celltype_expression @ is_receptor
    # Then calculate the propagated effects by testing ligands through the network
    lr_transition = ligand_mask @ celltype_gene_network_info
    # Now select receptors with zero activation so that they may be removed
    non_engaged_receptor_mask = (receptor_mask @ lr_transition) == 0
    # Now remove non-engaged receptors
    celltype_expression -= non_engaged_receptor_mask * celltype_expression

    # Calculate downstream effects
    celltype_expression_effects = celltype_expression @ celltype_gene_network_info
    # Delete diffused ligands
    celltype_expression_effects -= diffused_ligands[cell_selector, :]
    # Add effects to expression matrix
    expression_matrix = expression_matrix.at[cell_selector, :].add(celltype_expression_effects)

    # Clip to 0
    expression_matrix = jnp.clip(expression_matrix, 0, None)

    return flat_celltype_mask, expression_matrix, diffused_ligands, gene_network_info


@consumes_key(6, 'key', count=4)
@jax_jit()
def gene_expression_step(voronoi_map: jnp.array,
                         expression_matrix: jnp.array,
                         ligand_indices: jnp.array,
                         ligand_emission_propensities: jnp.array,
                         gene_info: jnp.array,
                         gene_network_info: jnp.array,
                         p_degredation: float,
                         key: StatefulPRNGKey) -> jnp.array:
    """
    Simulate gene expression and protein degredation, taking into consideration ligand emission properties.
    Given are the celltype-labelled spatial voronoi map (NxNxN), the expression matrix (NxG), the
    indices of ligands in the expression matrix (Lx1), the emission propensities for these ligands (Lx1),
    gene metadata (Gx2), gene2gene interactions (CTxGxG), and the random probability of degredation.
    """
    # Position and gene dimensions
    all_positions, all_rows = _project_voronoi_to_expression_positions(voronoi_map)
    flat_celltype_mask = voronoi_map[all_positions[0], all_positions[1], all_positions[2]]
    spatial_aware_expression = jnp.zeros((voronoi_map.shape[0], voronoi_map.shape[1], voronoi_map.shape[2], expression_matrix.shape[1])).at[all_positions].set(expression_matrix[all_rows])
    # First, simulate ligand diffusion
    diffused_ligands = spatial_aware_expression[:, :, :, ligand_indices]
    diffused_ligands = diffused_ligands * ligand_emission_propensities.T
    # Remove diffused ligands from the base expression matrix
    expression_matrix = expression_matrix.at[:, ligand_indices].add(-1*_transform_spatial_ligand_to_expression_matrix(diffused_ligands, expression_matrix, all_positions, all_rows, ligand_indices))
    # Now, shuffle the diffused ligands
    # Note: This is hard coded to a mean range of 1 to make it rare for ligands to diffuse more than 1 cell
    diffusion_ranges = jax.random.poisson(key[0], 1, (ligand_indices.shape[0],))
    # Create proportions representing the distance of each ligand from the center
    diffusion_proportions = jax.random.uniform(key[1], (jnp.sum(diffused_ligands).item(),))
    # Convert proportions to distances
    diffusion_positions = jnp.argwhere(diffused_ligands > 0)
    diffusion_distances = diffusion_proportions * diffusion_ranges[diffusion_positions[:, 3]].T
    # Create random angles for each ligand
    diffusion_angles = jax.random.uniform(key[2], (diffusion_proportions.shape[0], 2), 0, 2*jnp.pi)
    # Replicate diffused ligands to match the other arrays
    diffusion_positions = jnp.repeat(diffusion_positions, diffused_ligands[diffusion_positions], axis=0)
    # Calculate the offset for each ligand to diffuse by converting spherical coordinates to cartesian
    diffusion_vectors = jnp.stack([
        diffusion_distances * jnp.sin(diffusion_angles[:, 0]) * jnp.cos(diffusion_angles[:, 1]),
        diffusion_distances * jnp.sin(diffusion_angles[:, 0]) * jnp.sin(diffusion_angles[:, 1]),
        diffusion_distances * jnp.cos(diffusion_angles[:, 0])
    ])
    # Use a fori loop to retrieve the pre-generated distance, angle, and starting point
    # To move ligands around
    diffused_ligands, _, _ = jax.lax.fori_loop(0, diffusion_vectors.shape[0], _diffuse_ligand, (diffused_ligands, diffusion_positions, diffusion_vectors))
    # Convert the spatial ligand matrix, to a flattened expression matrix
    diffused_ligands = _transform_spatial_ligand_to_expression_matrix(diffused_ligands, expression_matrix, all_positions, all_rows, ligand_indices)

    # Next, simulate ligand-receptor interactions and expression
    # TODO: Should there be explicit cell-type specific receptor expression for sub populations? ex CCR2+ vs CCR2-?
    _, expression_matrix, _, _ = jax.lax.fori_loop(0, jnp.max(voronoi_map), _celltype_expression_step, (flat_celltype_mask, expression_matrix, diffused_ligands, gene_info, gene_network_info))

    # Finally, simulate degradation. p_degredation represents the average number of proteins per gene per cell that are degraded per time step
    expression_matrix -= jax.random.poisson(key[3], p_degredation, expression_matrix.shape)

    return expression_matrix


@consumes_key(3, 'key', count=2)
@jax_jit()
def death_replication_step(voronoi_map: jnp.array,
                           expression_matrix: jnp.array,
                           p_death: float,
                           key: StatefulPRNGKey) -> Tuple[jnp.array, jnp.array]:
    """
    Perform a single death and replicaiton where cells are deleted and then replaced by daughters of nearby cells.
    The modified voronoi map and expression matrix are returned, with p_death modulating the proportion of cells to 'kill' each call.
    """
    # Select cells to die (true/false)
    death_mask = jax.random.bernoulli(key[0], p_death, voronoi_map.shape)
    # Get the coordinates of the cells to die
    death_positions = jnp.argwhere(death_mask)
    # Select cells that can replicate to fill in the death mask
    to_replicate = _select_neighbors(voronoi_map, death_positions, key[1])
    # Replace the cells that died with the cells that can replicate
    voronoi_map = voronoi_map.at[death_positions[:, 0], death_positions[:, 1], death_positions[:, 2]].set(voronoi_map[to_replicate[:, 0], to_replicate[:, 1], to_replicate[:, 2]])

    # Convert positions to rows in the expression matrix
    death_rows = _position_to_cell_index(death_positions, voronoi_map.shape)
    to_replicate_rows = _position_to_cell_index(to_replicate, voronoi_map.shape)
    expression_matrix = expression_matrix.at[death_rows, :].set(expression_matrix[to_replicate_rows, :])

    return voronoi_map, expression_matrix


def simulation_iteration(voronoi_map: jnp.array,
                         expression_matrix: jnp.array,
                         cell_type_mobility_propensities: jnp.array,
                         cell_type_mobility_range: jnp.array,
                         ligand_indices: jnp.array,
                         ligand_emission_propensisties: jnp.array,
                         gene_network_info: jnp.array,
                         p_move: float,
                         p_degredation: float,
                         p_death: float,
                         key: StatefulPRNGKey) -> Tuple[jnp.array, jnp.array]:
    """
    Perform a single iteration of the simulation.
    :param voronoi_map: The 3D mapping of cell types (NxNxN).
    :param expression_matrix: The expression matrix (NxG).
    :param cell_type_mobility_propensities: The propensities for each cell type to move (CTx1).
    :param cell_type_mobility_range: The average movement distance for each cell type (CTx1).
    :param ligand_indices: The indices of the ligands in the expression matrix (Lx1).
    :param ligand_emission_propensisties: The propensities for each ligand to be emitted (Lx1).
    :param gene_network_info: The gene network information (Gx2).
    :param p_move: The random probaility of cells to move each iteration.
    :param p_degredation: The probability for proteins to naturally degrade.
    :param p_death: The proportion of cells to kill each iteration.
    :param key: The random key.
    :return: The updated spatial map and expression matrix.
    """

    # Move cells in space.
    voronoi_map, expression_matrix = cell_movement_step(voronoi_map, expression_matrix,
                                                        cell_type_mobility_propensities,
                                                        cell_type_mobility_range, p_move, key)

    # Simulate gene expression.
    expression_matrix = gene_expression_step(voronoi_map, expression_matrix, ligand_indices,
                                             ligand_emission_propensisties, gene_network_info,
                                             p_degredation, key)

    # 'Kill' cells and replace them with daughters of nearby cells.
    voronoi_map, expression_matrix = death_replication_step(voronoi_map, expression_matrix, p_death, key)

    return voronoi_map, expression_matrix


def simulate_cells(
        n_cells: int,
        n_celltypes: int,
        n_genes: int,
        p_move: float,
        p_degredation: float,
        p_death: float,
        n_iter: int,
        lr_pairs: Iterable[Tuple[int, int]],
        max_initial_celltype_colonies: int = None,
        key: StatefulPRNGKey = DEFAULT_RNG_KEY,
) -> SingleCellSimulationResults:
    """
    Generate a spatial distribution of cells and simulate them given a GRN.

    :param n_cells: The target number of cells to simulate (can be reduced if there is no integer cubic root).
    :param n_celltypes: The number of cell types to simulate.
    :param n_genes: The number of genes to simulate.
    :param p_move: The probability of a cell moving each iteration.
    :param p_degredation: The probability for proteins to naturally degrade.
    :param p_death: The proportion of cells to 'kill' each iteration.
    :param n_iter: The number of iterations for simulations to run.
    :param lr_pairs: The ligand-receptor pairs in the gene network.
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

    # Initialize expression matrix
    expression_matrix = _initial_expression_matrix(n_genes, n_cells, key)

    # Initialize global meta properties
    celltype_mobility = _generate_celltype_mobility_info(n_celltypes, key)
    celltype_mobility_range = celltype_mobility[:, 0]
    celltype_mobility_propensity = celltype_mobility[:, 1]
    del celltype_mobility

    ligand_indices = _get_ligand_indices(n_genes, lr_pairs)
    n_ligands = jnp.sum(ligand_indices).item()
    ligand_emission_propensities = _generate_random_emission_propensities(n_ligands, key)

    # Perform simulation iterations
    for i in range(n_iter):
        spatial_cell_distribution, expression_matrix = simulation_iteration(spatial_cell_distribution,
                                                                            expression_matrix,
                                                                            celltype_mobility_propensity,
                                                                            celltype_mobility_range,
                                                                            ligand_indices,
                                                                            ligand_emission_propensities,
                                                                            gene_network_info,
                                                                            p_move,
                                                                            p_degredation,
                                                                            p_death,
                                                                            key)


    return SingleCellSimulationResults(

    )


# Utility functions for internal datastructures
@jax_jit(static_argnums=(0,))
def _make_cell_type_mask(n_celltypes: int, voronoi: jnp.array) -> jnp.array:
    """
    Given a voronoi tessellation, create a mask for cell types. Output matrix is:
    CT x N x N x N
    Where CT is the number of cell types, and N is the number of cells in each dimension.
    Values are 1 if a given cell matches the cell type of its index, and 0 otherwise.
    """
    #return jnp.array([jnp.where(voronoi == celltype, 1, 0) for celltype in range(n_celltypes)])
    return jax.vmap(lambda celltype, vor: jnp.where(vor == celltype, 1, 0), in_axes=(0,))(jnp.arange(n_celltypes), voronoi)


@consumes_key(1, 'key')
@jax_jit(static_argnums=(0,))
def _generate_random_emission_propensities(n_genes: int, key: StatefulPRNGKey) -> jnp.array:
    """
    Generate a random emission propensity matrix for a given number of genes.
    Matrix is G x 1, where G is the number of genes.
    """
    return jax.random.uniform(key, (n_genes, 1))


@consumes_key(1, 'key', count=2)
@jax_jit(static_argnums=(0,))
def _generate_celltype_mobility_info(n_celltypes: int, key: StatefulPRNGKey) -> jnp.array:
    """
    Generate a random mobility matrix for a given number of cell types.
    Matrix is CT x 2, where CT is the number of cell types.
    The first dimension is a mean mobility range (proportional to exponential(1))
    The second dimension is the random mobility propensity.
    """
    return jnp.stack([
        exponential(1, (n_celltypes,), key[0]),
        jax.random.uniform(key[1], (n_celltypes,))
    ], axis=1)


@consumes_key(2, 'key')
@jax_jit(static_argnums=(0,1))
def _initial_expression_matrix(n_genes: int, n_cells: int, key: StatefulPRNGKey, max_expression: int = 100, p_expression: int = 0.5) -> jnp.array:
    """
    Generate a random initial expression matrix distributed using a negative binomial distribution.
    The resultant array is N x G where N is the number of cells and G is the number of genes.
    """
    return negative_binomial(max_expression, p_expression, (n_cells, n_genes), key)


@consumes_key(2, 'key')
@jax_jit()
def _select_neighbors(voronoi_map: jnp.array, positions: jnp.array, key: StatefulPRNGKey) -> jnp.array:
    """
    Select random neighbors for each position in a positions array.
    Given a NxNxN voronoi map, and a Mx3 positions array,
    Return a Mx3 array of neighbor positions.
    """
    # Determine all possible 3d offsets
    offsets = jnp.meshgrid(jnp.arange(-1, 2), jnp.arange(-1, 2), jnp.arange(-1, 2))
    # Choose the 3d offsets
    selected_offsets = jax.random.randint(key, (positions.shape[0],), 0, len(offsets))
    # Map the selected offset indices to the true offsets
    selected_offsets = offsets[selected_offsets]
    # add offsets to positions and clip to voronoi map shape
    return jnp.clip((positions + selected_offsets), 0, voronoi_map.shape[0]-1)


@jax_vmap(in_axes=(0, None, None))  # For each index
@jax_jit(static_argnums=(1,))
def _cell_index_to_position(index: int, shape: tuple) -> tuple:
    """
    Given a cell index from the expression matrix and the shape of the cube, return the position of the cell in 3D space.
    The resultant array is 3 x 1 where each element is the position of the cell in that dimension.
    """
    return jnp.unravel_index(index, shape)


@jax_vmap(in_axes=(0, None, None))  # For each position
@jax_jit(static_argnums=(1,))
def _position_to_cell_index(position: tuple, shape: tuple) -> int:
    """
    Given a position in 3D space and the shape of the cube, return the cell expression matrix index.
    The result is a single integer.
    """
    return jnp.ravel_multi_index(position, shape).item()


@jax_jit()
def _project_voronoi_to_expression_positions(voronoi: jnp.array):
    """
    Given the voronoi spatial map, return a list of all possible positions, and the corresponding cell row index.
    """
    all_positions = jnp.meshgrid(jnp.arange(voronoi.shape[0]), jnp.arange(voronoi.shape[1]), jnp.arange(voronoi.shape[2]))
    all_rows = _position_to_cell_index(all_positions, voronoi.shape)
    return all_positions, all_rows


@jax_jit()
def _transform_spatial_ligand_to_expression_matrix(spatial_ligand_matrix: jnp.array,
                                                   expression_matrix: jnp.array,
                                                   all_positions: jnp.array,
                                                   all_rows: jnp.array,
                                                   ligand_indices: jnp.array):
    """
    Transform a spatially mapped ligand concentration matrix (NxNxNxL) to a traditional expression matrix (C x G).
    Given are the selected 3D positions, selected cells, and indices of ligand genes.
    """
    # Get ligand expression matrix
    ligand_expression_matrix = jnp.zeros_like(expression_matrix)
    ligand_expression_matrix = ligand_expression_matrix.at[all_rows, ligand_indices].set(spatial_ligand_matrix[all_positions[0], all_positions[1], all_positions[2], :])

    return ligand_expression_matrix


def _get_ligand_indices(n_genes: int, lr_pairs: Iterable[Tuple[int, int]]) -> jnp.array:
    """
    Given a list of ligand-receptor pairs and the total number of genes, return an array of ligand indices (G x 1 matrix, where G is number of Genes).
    """
    # TODO: Should we make this jit-able?
    sorted_distinct_ligands = sorted(set([lr_pair[0] for lr_pair in lr_pairs]))
    ligand_indices = jnp.zeros((n_genes, 1), dtype=jnp.integer)
    ligand_indices = ligand_indices.at[sorted_distinct_ligands].set(1)
    return ligand_indices
