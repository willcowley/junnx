import abc
from typing import Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
import tensorflow_probability.substrates.jax as tfp

_EPS = 1e-12


class Likelihood(eqx.Module):

    @abc.abstractmethod
    def __call__(self, x: jnp.ndarray) -> tfp.distributions.Distribution: ...


class GaussianLikelihood(Likelihood):
    """Likelihood that assumes a Gaussian distribution."""

    _scale: jnp.ndarray
    bijector: tfp.bijectors.Bijector = eqx.field(static=True)
    """Bijector that ensures positivity of the scale paremeter."""

    def __init__(
        self,
        scale_init: Sequence[float],
        bijector: tfp.bijectors.Bijector,
    ) -> None:
        self.bijector = bijector
        self._scale = self.bijector.inverse(jnp.asarray(scale_init) - _EPS)

    @property
    def scale(self) -> jnp.ndarray:
        """The scale parameter of the Gaussian likelihood."""
        # small scale values can destabilise training, so we add a small epsilon here
        return self.bijector(self._scale) + _EPS

    def __call__(self, x: jnp.ndarray) -> tfp.distributions.MultivariateNormalDiag:
        # x: [S, N, O]
        return tfp.distributions.MultivariateNormalDiag(loc=x, scale_diag=self.scale)


class ClassificationLikelihood(Likelihood):

    @abc.abstractmethod
    def probs(self, x: jnp.ndarray) -> jnp.ndarray:
        """The probabilities of each class, given logits `x`."""


class BernoulliLikelihood(ClassificationLikelihood):
    """Likelihood that assumes a Bernoulli distribution."""

    def probs(self, x: jnp.ndarray) -> jnp.ndarray:
        p = tfp.bijectors.SoftmaxCentered()(x) * (1 - 2 * _EPS) + _EPS
        return p

    def __call__(self, x: jnp.ndarray) -> tfp.distributions.Bernoulli:
        # x: [S, N, 1]
        probs = self.probs(x)  # [S, N, 2]
        return tfp.distributions.Bernoulli(probs=probs[..., 1:])


class CategoricalLikelihood(ClassificationLikelihood):
    """Likelihood that assumes a Categorical distribution."""

    def probs(self, x: jnp.ndarray) -> jnp.ndarray:
        return jax.nn.softmax(x, axis=-1) * (1 - 2 * _EPS) + _EPS

    def __call__(self, x: jnp.ndarray) -> tfp.distributions.Categorical:
        # x: [S, N, O]
        probs = self.probs(x)  # [S, N, O]
        # expand so tfp dist has correct shape for batch and event dims
        probs = jnp.expand_dims(probs, axis=-2)  # [S, N, 1, O]
        return tfp.distributions.Categorical(probs=probs)
