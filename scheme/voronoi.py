import jax.numpy as jnp
import math
from jax import lax
import jax

from scheme.plotting import _draw_voronoi_slice, _draw_voronoi_slices_animation
from scheme.util import consumes_key, jax_jit, StatefulPRNGKey, jax_vmap


@jax_jit()
def _chebyshev_distance(x, y):
    return jnp.max(jnp.abs(x - y)).astype(float)


@jax_jit()
def _minkowski_distance(a: jnp.ndarray, b: jnp.ndarray, p: float = 2) -> jnp.ndarray:
    """
    Generate a distance value between two vectors.
    :param a: Vector one.
    :param b: Vector two,
    :param p: Hyper parameter (p=1 is Manhattan distance, p=2 is Euclidean distance, p=inf is Chebyshev distance)
    :return: Distance between the two vectors.
    """
    # Minkowski distance, but if p=inf, we use Chebyshev distance as a more direct calculation
    return lax.cond(jnp.isinf(p),
                    lambda x, y, _: _chebyshev_distance(x, y),
                    lambda x, y, p_: jnp.sum(jnp.abs(x - y) ** p_) ** (1 / p_),
                    a, b, p)


# Simulate cellular networks using voronoi tesselation via the jump flooding algorithm
@jax_jit()
def _make_neighbor_offsets(step: int = 1) -> jnp.ndarray:
    return jnp.array([
        [step, 0, 0],
        [0, step, 0],
        [0, 0, step],
        [step, step, 0],
        [0, step, step],
        [step, 0, step],
        [step, step, step],
        [-step, 0, 0],
        [0, -step, 0],
        [0, 0, -step],
        [-step, 0, 0],
        [0, -step, 0],
        [0, 0, -step],
        [-step, -step, 0],
        [0, -step, -step],
        [-step, 0, -step],
        [-step, -step, -step],
        [step, -step, 0],
        [-step, step, 0],
        [step, 0, -step],
        [-step, 0, step],
        [0, step, -step],
        [0, -step, step],
        [step, step, -step],
        [step, -step, step],
        [-step, step, step],
        [step, -step, -step],
        [-step, -step, step],
        [-step, step, -step],
    ], dtype=jnp.int32)


@jax_jit()
def __select_closest_color(curr_val: int, neighbor_val: int, dist_mat: jnp.ndarray, coordinate: jnp.array) -> int:
    new_dist = dist_mat[neighbor_val, coordinate[0], coordinate[1], coordinate[2]]
    curr_dist = dist_mat[curr_val, coordinate[0], coordinate[1], coordinate[2]]
    return lax.cond(curr_dist > new_dist,
                    lambda: neighbor_val,
                    lambda: curr_val)


@jax_jit()
def __maybe_take_color(curr_val: int, neighbor_val: int) -> int:
    return lax.cond(curr_val > -1,
                    lambda: curr_val,
                    lambda: neighbor_val)


@jax_jit()
def __compute_recolor(coordinate: jnp.array, neighbor_coordinate: jnp.array, mat: jnp.ndarray, dist_mat: jnp.ndarray) -> jnp.array:
    curr_val = mat[coordinate[0], coordinate[1], coordinate[2]]
    neighbor_val = mat[neighbor_coordinate[0], neighbor_coordinate[1], neighbor_coordinate[2]]
    return lax.cond(jnp.logical_and(curr_val > -1, neighbor_val > -1),
                    lambda: __select_closest_color(curr_val, neighbor_val, dist_mat, coordinate),
                    lambda: __maybe_take_color(curr_val, neighbor_val))


@jax_vmap(in_axes=(None, 0, None, None))  # For each neighbor
@jax_jit()
def __foreach_neighbor(coordinate: jnp.array, neighbor_offset: jnp.array, mat: jnp.ndarray, dist_mat: jnp.ndarray) -> jnp.array:
    neighbor_coordinate = coordinate + neighbor_offset
    # If the neighbor is out of bounds, don't color
    all_in_bounds = jnp.logical_and(jnp.all(neighbor_coordinate >= 0),
                                    jnp.any(neighbor_coordinate - mat.shape[0] < 0))
    curr_val = mat[coordinate[0], coordinate[1], coordinate[2]]
    return lax.cond(all_in_bounds,
                    lambda: __compute_recolor(coordinate, neighbor_coordinate, mat, dist_mat),
                    lambda: curr_val)


@jax_vmap(in_axes=(0, None, None, None))  # For each coordinate
@jax_jit()
def __foreach_pixel(coordinate: jnp.array, neighbor_offsets: jnp.ndarray, mat: jnp.ndarray, dist_mat: jnp.ndarray) -> jnp.array:
    return __foreach_neighbor(coordinate, neighbor_offsets, mat, dist_mat)


@jax_vmap(in_axes=(None, 3, None), out_axes=1)  # For each possible value
@jax_vmap(in_axes=(0, None, None))  # For each coordinate
@jax_jit()
def __map_to_dists(coordinate: jnp.array, mat: jnp.ndarray, dist_mat: jnp.ndarray) -> jnp.array:
    seed = mat[coordinate[0], coordinate[1], coordinate[2]]
    return dist_mat[seed, coordinate[0], coordinate[1], coordinate[2]]


@jax_jit()
def __jump_flooding_iter(i: int, coordinate_map: jnp.ndarray, neighbor_offsets: jnp.ndarray, mat: jnp.ndarray, dist_mat: jnp.ndarray, steps: jnp.array) -> jnp.ndarray:
    step = steps[i]
    step_neighbor_offsets = neighbor_offsets * step

    pixels2possible = __foreach_pixel(coordinate_map, step_neighbor_offsets, mat, dist_mat)
    # Reshape to make 4D by prepending 3 dimensions matching the size of mat
    pixels2possible = jnp.reshape(pixels2possible, (mat.shape[0], mat.shape[1], mat.shape[2], -1))

    # Get all the distances for each possible value
    possible_distances = __map_to_dists(coordinate_map, pixels2possible, dist_mat)
    possible_distances = jnp.reshape(possible_distances, (mat.shape[0], mat.shape[1], mat.shape[2], -1))

    # Only select values that changed
    changed_mask = jnp.not_equal(pixels2possible, mat[..., None])
    pixels2possible = jnp.where(changed_mask, pixels2possible, -1)
    possible_distances = jnp.where(changed_mask, possible_distances, jnp.inf)
    # This array is our matrix, but with all the possible values for each pixel.
    # Reduce by selecting the changed values that are closest to the pixel
    selected = jnp.argmin(possible_distances, axis=-1)
    # Get the new values from the selected indices
    selected_pixels = jnp.take_along_axis(pixels2possible, selected[..., None], axis=-1)
    # Flatten the last dimension
    selected_pixels = jnp.reshape(selected_pixels, mat.shape)

    # Only update the pixels that changed
    return jnp.where(selected_pixels > -1, selected_pixels, mat)


@jax_jit(static_argnums=(0, 1))
def _make_step_size_array(max_step: int, step_array_size: int) -> jnp.ndarray:
    step_range = jnp.arange(1, step_array_size+1, dtype=jnp.int32)
    return jnp.maximum(jnp.ones_like(step_range), max_step / (2**step_range)).astype(jnp.int32)


@jax_vmap(in_axes=(0, None, None))  # For each seed
@jax_vmap(in_axes=(None, 0, None))  # For each coordinate
@jax_jit()
def _make_distance_matrix(seed_coordinate: jnp.array, coord: jnp.array, p: float) -> jnp.array:
    return _minkowski_distance(seed_coordinate, coord, p)


@consumes_key(3, 'rng_key')
@jax_jit(static_argnums=(0, 1))
def _jump_flooding_impl(size: int, seeds: int, step_sizes_array: jnp.array, rng_key: StatefulPRNGKey, p: float = 2) -> jnp.ndarray:
    mat = (jnp.zeros((size, size, size), dtype=jnp.int32) - 1)
    # Add uniquely numbered seeds
    seed_coordinates = jax.random.randint(rng_key, (seeds, 3), minval=0, maxval=size)
    mat = mat.at[seed_coordinates[:, 0], seed_coordinates[:, 1], seed_coordinates[:, 2]] \
        .set(jnp.arange(0, seeds))
    # Perform the jump-flooding algorithm
    xvals, yvals, zvals = jnp.meshgrid(jnp.arange(0, size),
                                       jnp.arange(0, size),
                                       jnp.arange(0, size),
                                       indexing='ij')
    # flatten and combine the x, y, z values into a single array
    coordinate_map = jnp.stack([xvals.flatten(), yvals.flatten(), zvals.flatten()], axis=1)

    # Base offsets to check
    neighbor_offsets = _make_neighbor_offsets()

    # Make distance matrices to each seed
    dist_mat = _make_distance_matrix(seed_coordinates, coordinate_map, p).reshape((seeds, size, size, size))

    # Iterate over various stepsizes
    mat = lax.fori_loop(0, step_sizes_array.shape[0],
                        lambda i, m: __jump_flooding_iter(i, coordinate_map, neighbor_offsets, m, dist_mat, step_sizes_array),
                        mat)


    return mat


def jump_flooding(size: int, seeds: int, rng_key: StatefulPRNGKey, p: float = jnp.inf) -> jnp.ndarray:
    """
    Perform the Jump-Flooding-Algorithm (JFA) to generate a Voronoi tesselation.
    See https://en.wikipedia.org/wiki/Jump_flooding_algorithm for more details.
    Note that this creates a three-dimensional array.
    :param size: Determines the N which represents the NxNxN output matrix that gets returned.
    :param seeds: The number of "seeds" to start the tesselations from.
    :param rng_key: The random key used for seeding.
    :param p: The p hyperparameter for minkowski distance (p=1 is Manhattan distance, p=2 is Euclidean distance, p=inf is Chebyshev distance)
    :return: A three-dimensional matrix of size NxNxN.
    """
    # Stepsizes
    step_sizes = _make_step_size_array(size, int(math.log2(size)))
    # Step sizes are dynamically generated, so defer to jitted implementation
    # We are going to use the 1+JFA variant to improve performance
    # So we must prepend a 1 to the step sizes
    step_sizes = jnp.concatenate([jnp.array([1]), step_sizes])
    #print(step_sizes)

    return _jump_flooding_impl(size, seeds, step_sizes, rng_key, p)


"""
Generate cell-cell interactions via a voronoi tesselation.
"""

if __name__ == "__main__":
    voronoi_mat = jump_flooding(100, 35, StatefulPRNGKey(0), p=3)
    _draw_voronoi_slice(voronoi_mat, 25, title="Voronoi slice at z=25")
    _draw_voronoi_slice(voronoi_mat, 50, title="Voronoi slice at z=50")
    _draw_voronoi_slice(voronoi_mat, 75, title="Voronoi slice at z=75")
    _draw_voronoi_slices_animation(voronoi_mat, title="Voronoi slices animation")

