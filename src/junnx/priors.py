import abc
from typing import Sequence

import equinox as eqx
import jax.numpy as jnp
import tensorflow_probability.substrates.jax as tfp

from junnx.net import StochasticNet
from junnx.variational import VariationalDistribution

_JITTER = 1e-6
_EPS = 1e-12


class Prior(eqx.Module):
    """Abstract base class for priors."""

    @abc.abstractmethod
    def __call__(
        self, x: jnp.ndarray, n_samples: int, *, key: jnp.ndarray
    ) -> tfp.distributions.Distribution: ...


class SampleStochasticNetPrior(Prior):
    """Prior based on a `DenseStochasticNet` and `VariationalDistribution`."""

    net: StochasticNet
    variational_dist: VariationalDistribution

    def __call__(
        self, x: jnp.ndarray, n_samples: int, key: jnp.ndarray
    ) -> tfp.distributions.Distribution:
        # x: [N, D]
        predf = self.net.predict_f_samples(x, n_samples, key=key)  # [S, N, O]
        return self.variational_dist(predf)  # [O, N]


class DirichletPrior(Prior):
    """Dirichlet distribution prior."""

    concentration: jnp.ndarray  # [D,]  (D >= 2)
    """
    The concentration parameters of the Dirichlet distribution. Must be strictly positive.
    """

    def __call__(
        self, x: jnp.ndarray, n_samples: int, key: jnp.ndarray
    ) -> tfp.distributions.Dirichlet:
        # x: [N, D]
        return tfp.distributions.Dirichlet(concentration=self.concentration)


def squared_distance_matrix(x: jnp.ndarray) -> jnp.ndarray:
    # x: [..., N, D]
    x2 = jnp.sum(x**2, axis=-1, keepdims=True)  # [..., N, 1]
    x2T = jnp.swapaxes(x2, -2, -1)  # [..., 1, N]
    xT = jnp.swapaxes(x, -2, -1)  # [..., D, N]
    r2 = x2 + x2T - 2.0 * x @ xT  # [..., N, N]
    return jnp.maximum(r2, 0.0)  # [..., N, N]


class IsotropicStationaryKernelPrior(Prior):

    lengthscales: jnp.ndarray  # [D,]
    """The kernel lengthscales for each input dimension."""
    variance: jnp.ndarray  # [,]
    """THe kernel variance. Defaults to 1.0."""

    def __init__(self, lengthscales: Sequence[float], variance: float = 1.0) -> None:
        self.variance = jnp.asarray(variance)
        self.lengthscales = jnp.asarray(lengthscales)

    def scale(self, x: jnp.ndarray) -> jnp.ndarray:
        # x: [N, D]
        return x / self.lengthscales

    def scaled_squared_euclid_dist(self, x: jnp.ndarray) -> jnp.ndarray:
        # x: [N, D]
        return squared_distance_matrix(self.scale(x))  # [N, N]

    @abc.abstractmethod
    def kernel(self, x: jnp.ndarray) -> jnp.ndarray: ...

    def __call__(
        self, x: jnp.ndarray, n_samples: int, *, key: jnp.ndarray
    ) -> tfp.distributions.Distribution:
        # x: [N, D]
        N, *_ = x.shape
        cov = self.kernel(x)  # [N, N]
        cov = 0.5 * (cov + cov.T)  # enforce symmetry
        L = jnp.linalg.cholesky(cov + _JITTER * jnp.eye(N, dtype=cov.dtype))  # [N, N]
        return tfp.distributions.MultivariateNormalTriL(
            loc=jnp.zeros((N,), dtype=L.dtype), scale_tril=L
        )


class RBFPrior(IsotropicStationaryKernelPrior):
    """
    Radial Basis Function (RBF) kernel prior. Also known as the squared exponential
    kernel.
    """

    def kernel(self, x: jnp.ndarray) -> jnp.ndarray:
        # x: [N, D]
        r2 = self.scaled_squared_euclid_dist(x)  # [N, N]
        return self.variance * jnp.exp(-0.5 * r2)


class Matern52Prior(IsotropicStationaryKernelPrior):
    """Matérn 5/2 kernel prior."""

    def kernel(self, x: jnp.ndarray) -> jnp.ndarray:
        # X: [N, D]
        r = jnp.sqrt(self.scaled_squared_euclid_dist(x) + _EPS)  # [N, N]
        sqrt5 = jnp.sqrt(5.0)
        return (
            self.variance
            * (1.0 + sqrt5 * r + (5.0 / 3.0) * jnp.square(r))
            * jnp.exp(-sqrt5 * r)
        )
