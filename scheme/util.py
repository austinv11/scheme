import random
from typing import Tuple, TypeVar, Callable

import jax
import treeo as to
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
    :param kwargs: The arguments to pass to jax.jit()
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

