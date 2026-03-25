import abc
from typing import Sequence

import equinox as eqx
import jax
import jax.numpy as jnp


class Sampler(eqx.Module):

    @abc.abstractmethod
    def __call__(self, key: jnp.ndarray) -> jnp.ndarray: ...


class UniformSampler(Sampler):
    """
    Samples uniformly from the hypercube defined by `low` and `high`.

    Args:
        n_dim: The dimensionality of the hypercube.
        n_samples: The number of samples to draw from the hypercube on each call.
        low: The lower bounds of the hypercube.
        high: The upper bounds of the hypercube.
    """

    low: jnp.ndarray
    """The lower bounds of the hypercube."""
    high: jnp.ndarray
    """The upper bounds of the hypercube."""

    n_dim: int = eqx.field(static=True)
    """The dimensionality of the hypercube."""
    n_samples: int = eqx.field(static=True)
    """The number of samples to draw from the hypercube on each call."""

    def __init__(
        self, n_dim: int, n_samples: int, low: Sequence[float], high: Sequence[float]
    ) -> None:

        self.n_dim = n_dim
        self.n_samples = n_samples

        self.low = jnp.asarray(low)
        self.high = jnp.asarray(high)

    def __call__(self, key: jnp.ndarray) -> jnp.ndarray:
        return jax.random.uniform(
            key, shape=(self.n_samples, self.n_dim), minval=self.low, maxval=self.high
        )
