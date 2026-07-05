import abc
from typing import Sequence

import equinox as eqx
import jax.numpy as jnp
import tensorflow_probability.substrates.jax as tfp

from junnx.net import StochasticNet, TractableStochasticNet
from junnx.variational import GaussianVariationalDistribution, VariationalDistribution

_JITTER = 1e-6
_EPS = 1e-12


class Prior(eqx.Module):
    """Abstract base class for priors."""

    @abc.abstractmethod
    def __call__(
        self, x: jnp.ndarray, *, key: jnp.ndarray
    ) -> tfp.distributions.Distribution: ...


class SampleStochasticNetPrior(Prior):
    """Prior based on a `DenseStochasticNet` and `VariationalDistribution`."""

    net: StochasticNet
    variational_dist: VariationalDistribution
    n_samples: int = eqx.field(static=True)

    def __call__(self, x: jnp.ndarray, *, key: jnp.ndarray) -> tfp.distributions.Distribution:
        # x: [N, D]
        predf = self.net.predict_f_samples(x, self.n_samples, key=key)  # [S, N, O]
        return self.variational_dist(predf)  # [O, N]


class TractableStochasticNetPrior(Prior):

    net: TractableStochasticNet

    def __init__(self, net: TractableStochasticNet, log_var: float | None = None):
        if log_var is not None:
            net = eqx.tree_at(
                lambda n: n.last_layer.w_log_var,
                net,
                jnp.full_like(net.last_layer.w_log_var, log_var),
            )
        self.net = net

    def __call__(self, x: jnp.ndarray, *, key: jnp.ndarray) -> tfp.distributions.Distribution:
        mean, cov = self.net.tractable_f_mean_cov(x, key=key)
        return GaussianVariationalDistribution().from_mean_cov(mean, cov)


class DirichletPrior(Prior):
    """Dirichlet distribution prior."""

    concentration: jnp.ndarray  # [D,]  (D >= 2)
    """
    The concentration parameters of the Dirichlet distribution. Must be strictly positive.
    """

    def __call__(self, x: jnp.ndarray, *, key: jnp.ndarray) -> tfp.distributions.Dirichlet:
        # x: [N, D]
        return tfp.distributions.Dirichlet(concentration=self.concentration)


def square_distance(x: jnp.ndarray, x2: jnp.ndarray | None = None) -> jnp.ndarray:
    # x: [..., N, D], x2: [..., M, D] or None (defaults to x, i.e. self-distance)
    if x2 is None:
        x2 = x
    xs = jnp.sum(x**2, axis=-1, keepdims=True)  # [..., N, 1]
    x2s = jnp.sum(x2**2, axis=-1, keepdims=True)  # [..., M, 1]
    x2sT = jnp.swapaxes(x2s, -2, -1)  # [..., 1, M]
    x2T = jnp.swapaxes(x2, -2, -1)  # [..., D, M]
    r2 = xs + x2sT - 2.0 * x @ x2T  # [..., N, M]
    return r2


class IsotropicStationaryKernelPrior(Prior):

    lengthscales: jnp.ndarray  # [D,]
    """The kernel lengthscales for each input dimension."""
    variance: jnp.ndarray  # [,]
    """The kernel variance. Defaults to 1.0."""

    def __init__(self, lengthscales: Sequence[float], variance: float = 1.0) -> None:
        self.variance = jnp.asarray(variance)
        self.lengthscales = jnp.asarray(lengthscales)

    def scale(self, x: jnp.ndarray) -> jnp.ndarray:
        # x: [N, D]
        return x / self.lengthscales

    def scaled_squared_euclid_dist(
        self, x: jnp.ndarray, x2: jnp.ndarray | None = None
    ) -> jnp.ndarray:
        # x: [N, D], x2: [M, D] or None
        x2_scaled = self.scale(x2) if x2 is not None else None
        return square_distance(self.scale(x), x2_scaled)  # [N, N] or [N, M]

    @abc.abstractmethod
    def kernel(self, x: jnp.ndarray, x2: jnp.ndarray | None = None) -> jnp.ndarray: ...

    def mean_function(self, x: jnp.ndarray) -> jnp.ndarray:
        # x: [N, D]
        N, *_ = x.shape
        return jnp.zeros((N,), dtype=x.dtype)

    def __call__(self, x: jnp.ndarray, *, key: jnp.ndarray) -> tfp.distributions.Distribution:
        # x: [N, D]
        N, *_ = x.shape
        cov = self.kernel(x)  # [N, N]
        cov = 0.5 * (cov + cov.T)  # enforce symmetry
        L = jnp.linalg.cholesky(cov + _JITTER * jnp.eye(N, dtype=cov.dtype))  # [N, N]
        return tfp.distributions.MultivariateNormalTriL(
            loc=self.mean_function(x), scale_tril=L
        )


class RBFPrior(IsotropicStationaryKernelPrior):
    """
    Radial Basis Function (RBF) kernel prior. Also known as the squared exponential
    kernel.
    """

    def kernel(self, x: jnp.ndarray, x2: jnp.ndarray | None = None) -> jnp.ndarray:
        # x: [N, D], x2: [M, D] or None
        r2 = self.scaled_squared_euclid_dist(x, x2)  # [N, N] or [N, M]
        return self.variance * jnp.exp(-0.5 * r2)


class Matern52Prior(IsotropicStationaryKernelPrior):
    """Matérn 5/2 kernel prior."""

    def kernel(self, x: jnp.ndarray, x2: jnp.ndarray | None = None) -> jnp.ndarray:
        # x: [N, D], x2: [M, D] or None
        r = jnp.sqrt(self.scaled_squared_euclid_dist(x, x2) + _EPS)  # [N, N] or [N, M]
        sqrt5 = jnp.sqrt(5.0)
        return (
            self.variance
            * (1.0 + sqrt5 * r + (5.0 / 3.0) * jnp.square(r))
            * jnp.exp(-sqrt5 * r)
        )
