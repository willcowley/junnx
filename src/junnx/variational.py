import abc

import equinox as eqx
import jax
import jax.numpy as jnp
import tensorflow_probability.substrates.jax as tfp

from junnx.utils import sample_mean_cov

_JITTER = 1e-6
_EPS = 1e-12


class VariationalDistribution(eqx.Module):

    @abc.abstractmethod
    def __call__(self, pred_f_samples: jnp.ndarray) -> tfp.distributions.Distribution: ...


class GaussianVariationalDistribution(VariationalDistribution):
    """
    Approximates a Gaussian distribution given samples from the predictive distribution.
    """

    def __call__(
        self, pred_f_samples: jnp.ndarray
    ) -> tfp.distributions.MultivariateNormalTriL:
        mean, cov = sample_mean_cov(pred_f_samples)  # mean: [O, N], cov: [O, N, N]
        return self.from_mean_cov(mean, cov)

    def from_mean_cov(
        self, mean: jnp.ndarray, cov: jnp.ndarray
    ) -> tfp.distributions.MultivariateNormalTriL:
        # mean: [O, N], cov: [O, N, N]
        _jitter = _JITTER * jnp.eye(cov.shape[-1], dtype=cov.dtype)[None, ...]  # [1, N, N]
        Lq = jnp.linalg.cholesky(cov + _jitter)  # [O, N, N]
        return tfp.distributions.MultivariateNormalTriL(loc=mean, scale_tril=Lq)


class DirichletVariationalDistribution(VariationalDistribution):
    """
    Approximates a Dirichlet distribution given samples from the predictive distribution.
    """

    def __call__(self, pred_f_samples: jnp.ndarray) -> tfp.distributions.Dirichlet:
        # pred_f_samples: [S, N, O]
        prob_samples = (1 - 2 * _EPS) * jax.nn.softmax(
            pred_f_samples, axis=-1
        ) + _EPS  # [S, N, O]
        mean = jnp.mean(prob_samples, axis=0)  # [N, O]
        var = jnp.var(prob_samples, axis=0) + _EPS  # [N, O]
        precision = mean * (1 - mean) / var - 1  # [N, O]
        concentration = mean * precision  # [N, O]
        return tfp.distributions.Dirichlet(concentration=concentration)
