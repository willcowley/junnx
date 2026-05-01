import abc
from dataclasses import dataclass
from typing import Sequence

import jax
import jax.numpy as jnp

from junnx.datasets import TensorDataset


@dataclass(frozen=True)
class Sampler:

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

    n_dim: int
    """The dimensionality of the hypercube."""
    n_samples: int
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


class DataSampler(Sampler):
    """
    Samples randomly from a dataset.

    Args:
        n_samples: The number of samples to draw from the dataset on each call.
    """

    n_samples: int
    """The number of samples to draw from the dataset on each call."""
    data: TensorDataset
    """The dataset from which to sample."""

    def __init__(self, n_samples: int, data: TensorDataset) -> None:
        self.n_samples = n_samples
        self.data = data

    def __call__(self, key: jnp.ndarray) -> jnp.ndarray:
        idxs = jax.random.permutation(key, jnp.arange(len(self.data)))[: self.n_samples]
        return self.data.x[idxs]
