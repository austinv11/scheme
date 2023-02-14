import random
from typing import Tuple, TypeVar, Callable, Optional

import jax
import treeo as to
from numpyro.distributions import Exponential, VonMises
from numpyro.distributions.conjugate import NegativeBinomial


# JAX utilities
F = TypeVar("F", bound=Callable)  # Allow for type inspection with decorators


def jax_jit(**kwargs) -> Callable[[F], F]:
    """
    Jit a function using JAX.
    :param kwargs: The arguments to pass to jax.jit()
    """
    def decorator(func: F) -> F:
        #return func  # To ignore jit, uncomment this line
        return jax.jit(func, **kwargs)
    return decorator


def jax_vmap(**kwargs) -> Callable[[F], F]:
    """
    Vmap a function using JAX.
    :param kwargs: The arguments to pass to jax.vmap()
    """
    def decorator(func: F) -> F:
        return jax.vmap(func, **kwargs)
    return decorator


class StatefulPRNGKey(to.Tree):
    """
    Stateful PRNG key for JAX for simplicity.
    """

    key: jax.random.PRNGKey = to.node()
    seed: int = to.static()
    call_count: int = to.node()

    def __init__(self, seed: int = 0):
        self.seed = seed
        self.call_count = 0
        self.key = jax.random.PRNGKey(seed)

    def __call__(self):
        self.key, subkey = jax.random.split(self.key, 2)
        self.call_count = 1+self.call_count
        return subkey

    def __repr__(self):
        return f"StatefulPRNGKey(seed={self.seed}, call_count={self.call_count})"

    def __hash__(self):
        return hash(self.seed) ^ hash(self.call_count)

    def __eq__(self, other):
        return self.key == other.key

    # Dummy index to make it drop-in compatible with code using the @consumes_key decorator
    def __getitem__(self, n):
        self.key, *subkeys = jax.random.split(self.key, n + 1)
        self.call_count += n
        return subkeys


DEFAULT_RNG_KEY = StatefulPRNGKey(random.randint(0, 2 ** 32 - 1))


def consumes_key(argnum: int, argname: str, count: int = 1) -> Callable[[F], F]:
    """
    Decorator to consume a key from a function input. This is useful for jitting functions that use random numbers.
    :param argnum: The numerical index of the argument to consume.
    :param argname: The name of the argument to consume.
    :param count: The number of random keys to extract.
    """
    def decorator(func: F) -> F:
        def wrapper(*args, **kwargs):
            if len(args) > argnum:
                key = args[argnum]
                if isinstance(key, StatefulPRNGKey):
                    if count > 1:
                        subkeys = key[count]
                    else:
                        subkeys = key()
                    args = (*args[:argnum], subkeys, *args[argnum+1:])
            if argname in kwargs:
                key = kwargs[argname]
                if isinstance(key, StatefulPRNGKey):
                    if count > 1:
                        subkeys = key[count]
                    else:
                        subkeys = key()
                    kwargs = {**kwargs, argname: subkeys}
            return func(*args, **kwargs)
        return wrapper
    return decorator


@jax_jit()
def _emphasize_group(proportions, prioritized: int):
    """Emphasize a group of proportions from a list of proportions."""
    proportions = proportions.at[prioritized].mul(2)
    return proportions


@consumes_key(2, 'rng_key', count=2)
@jax_jit(static_argnums=(0,))
def _randomized_proportions(groups: int, prioritize_group: Optional[int], rng_key: StatefulPRNGKey):
    """Generate random propensities of interaction that sum to 1"""
    # Bimodal distribution
    proportions = jax.numpy.clip(bimodal_normal((.25, .75), .25, groups, rng_key[:2]), 0, 1)  # We want groups to have more varied sizes
    # For cell-cell-communication networks, we want to emphasize the intracluster relationships
    proportions = jax.lax.cond(prioritize_group is None, lambda prop, prioritize: prop, _emphasize_group,
                           proportions, prioritize_group)
    proportions /= proportions.sum()  # Normalize
    return proportions


@consumes_key(3, 'key')
@jax_jit(static_argnums=(2,))
def lognormal(mean, std, shape, key: StatefulPRNGKey):
    """
    Sample from a parameterized log-normal distribution.
    """
    return jax.numpy.exp(parameterized_normal(mean, std, shape, key))


@consumes_key(3, 'key')
@jax_jit(static_argnums=(2,))
def parameterized_normal(mean, std, shape, key: StatefulPRNGKey):
    """
    Sample from a parameterized normal distribution
    """
    return jax.random.normal(key, shape) * std + mean


@consumes_key(3, 'key', count=2)
@jax_jit(static_argnums=(2,))
def bimodal_normal(means: Tuple[float, float], std: float, samples: int, key: StatefulPRNGKey):
    """
    Sample from a bivariate normal distribution
    """
    # Sample from multivariate normal
    return (parameterized_normal(means[0], std, (samples,), key[0]) + parameterized_normal(means[1], std, (samples,), key[1])) / 2


@consumes_key(3, 'key')
@jax_jit(static_argnums=(2,))
def negative_binomial(total_count, probs, shape, key: StatefulPRNGKey):
    """
    Sample from a negative binomial distribution
    """
    # We have to use a numpyro distribution since jax has not implemented negative binomial
    distribution = NegativeBinomial(total_count=total_count, probs=probs)
    return distribution.sample(key, shape)


@consumes_key(2, 'key')
@jax_jit(static_argnums=(1,))
def exponential(lmda, shape, key: StatefulPRNGKey):
    """
    Sample from an exponential distribution
    """
    distribution = Exponential(rate=lmda)
    return distribution.sample(key, shape)


@jax_jit()
def generalized_logistic(x, L=1, k=1, x0=0):
    """
    Apply a logistic function. Default is equivalent to a sigmoid.
    :param x: The input to the logistic function.
    :param L: The maximum value of the logistic function (0-L).
    :param k: The steepness of the logistic function.
    :param x0: The x-value of the midpoint of the logistic function.
    """
    return L / (1 + jax.numpy.exp(-k * (x - x0)))


@jax_jit()
def swap_elements(array: jax.numpy.array, index1, index2) -> jax.numpy.array:
    """
    Swap two elements in an array
    :param array: The array to swap elements in.
    :param index1: The index of the first element to swap.
    :param index2: The index of the second element to swap.
    :return: The updated array.
    """
    temp = array[index1]
    array = array.at[index1].set(array[index2])
    array = array.at[index2].set(temp)
    return array


@jax_jit()
def swap_rows(array: jax.numpy.array, row1, row2) -> jax.numpy.array:
    """
    Swap two rows (0th dimension) in an array.
    :param array: The array to swap rows in.
    :param row1: The index of the first row to swap.
    :param row2: The index of the second row to swap.
    :return: The updated array.
    """
    temp = array[row1, :]
    array = array.at[row1, :].set(array[row2, :])
    array = array.at[row2, :].set(temp)
    return array
