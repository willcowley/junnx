import equinox as eqx
import tensorflow_probability.substrates.jax as tfp
from jax import numpy as jnp

from junnx.likelihoods import Likelihood
from junnx.net import StochasticNet
from junnx.priors import Prior
from junnx.samplers import Sampler
from junnx.variational import VariationalDistribution

_JITTER = 1e-6


class TrainingModel(eqx.Module):
    """
    Training Model for JUNNX, containing all the components necessary for FSVI.
    """

    net: StochasticNet
    """The stochastic network that defines a distribution over functions, f."""
    likelihood: Likelihood
    """The likelihood of observing the data given function values, P(Y|f)."""
    prior: Prior
    """The prior distribution over the function values, P(f)."""
    sampler: Sampler
    """
    Sampler from the input space used to compute the KL divergence between the variational
    distribution over f and the prior.
    """
    variational_dist: VariationalDistribution
    """The variational distribution used to approximate the predictive distribution over f."""

    def partition(self) -> tuple["TrainingModel", "TrainingModel"]:
        """Partition the model into trainable and static parts."""
        trainable, static = eqx.partition(self, eqx.is_inexact_array)

        # Explicitly move prior to static
        static = eqx.tree_at(lambda m: m.prior, static, self.prior)
        trainable = eqx.tree_at(lambda m: m.prior, trainable, None)

        # Explicitly move sampler to static
        static = eqx.tree_at(lambda m: m.sampler, static, self.sampler)
        trainable = eqx.tree_at(lambda m: m.sampler, trainable, None)

        # Explicitly move var_dist to static
        static = eqx.tree_at(lambda m: m.variational_dist, static, self.variational_dist)
        trainable = eqx.tree_at(lambda m: m.variational_dist, trainable, None)

        return trainable, static

    def predict_ydist(self, f_samples: jnp.ndarray) -> tfp.distributions.Distribution:
        # f_samples: [S, N, O]
        n_samples, *_ = f_samples.shape
        # transpose for tfp.distributions.MixtureSameFamily, which expects the mixture
        # components to be in the last event_shape dimension
        f_samples = jnp.transpose(f_samples, (1, 0, 2))  # [N, S, O]
        ydist = self.likelihood(f_samples)  # [N, S, O]
        mixture_distribution = tfp.distributions.Categorical(
            logits=jnp.zeros(n_samples)
        )  # [S,]
        return tfp.distributions.MixtureSameFamily(mixture_distribution, ydist)  # [N, O]

    def predict_y_mean_var(self, f_samples: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        """
        Predicts the mean and variance of the predictive distribution over observations, Y,
        given samples from the predictive distribution over f.
        """
        mixture = self.predict_ydist(f_samples)  # [N, O]
        mean = mixture.mean()  # [N, O]
        var = mixture.variance()  # [N, O]
        return mean, var

    def variational(
        self, x: jnp.ndarray, n_samples: int, *, key: jnp.ndarray
    ) -> tfp.distributions.Distribution:
        """
        Computes the variational distribution at the input locations `x`.
        """
        # x: [N, D]
        predf = self.net.predict_f_samples(x, n_samples, key=key)  # [S, N, O]
        return self.variational_dist(predf)
