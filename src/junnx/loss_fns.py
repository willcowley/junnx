"""
Loss functions for performing approximate Bayesian inference.
"""

import abc

import equinox as eqx
import jax
import jax.numpy as jnp
import tensorflow_probability.substrates.jax as tfp

from junnx.model import TrainingModel
from junnx.net import TractableStochasticNet
from junnx.variational import GaussianVariationalDistribution


class LossFn(eqx.Module):
    """Abstract base class for loss functions."""

    @abc.abstractmethod
    def __call__(
        self,
        m: TrainingModel,
        x: jnp.ndarray,
        y: jnp.ndarray,
        mask: jnp.ndarray | None,
        context_x: jnp.ndarray | None,
        n_batches_per_epoch: int,
        *,
        key: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """
        Evaluates a scalar loss at `x`

        Args:
            m: The model
            x: The input data
            y: The target data
            context_x: Context input data used to evaluate the KL divergernce term
            n_batches_per_epoch: The number of training data batches per epoch
            key: A psuedo-random key for the loss evaluation

        Returns:
            - The scalar loss value
            - Sampled function values from model `m` at points `x`
        """


class NLLLoss(LossFn):
    """
    A loss that computes the negative log likelihood
    """

    n_samples_nll: int = eqx.field(static=True)
    """The number of samples to use to evaluate the negative log likelihood."""

    def __call__(
        self,
        m: TrainingModel,
        x: jnp.ndarray,
        y: jnp.ndarray,
        mask: jnp.ndarray | None,
        context_x: jnp.ndarray | None,
        n_batches_per_epoch: int,
        *,
        key: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        batch_size, *_ = x.shape

        # compute NLL
        predf = m.net.predict_f_samples(x, self.n_samples_nll, key=key)  # [SN, N, O]
        predy = m.likelihood(predf)
        if mask is not None:
            y = jnp.where(mask, y, 0.0)
        log_prob_y = predy.log_prob(y[None])  # [SN, N, O]
        if mask is not None:
            log_prob_y = jnp.where(mask[None], log_prob_y, 0.0)  # [SN, N, O]
            log_prob_y = log_prob_y.sum(axis=-1) / mask.sum()  # [SN, N]
        else:
            log_prob_y = log_prob_y.mean(axis=-1)
        nll_loss = -log_prob_y.sum(axis=-1).mean()  # [,]  per-batch NLL

        return nll_loss / batch_size, predf  # [,], [SN, N, O]


class SampleFSVILoss(NLLLoss):
    """
    A loss that performs functions-space variational inference (FSVI) using an MC sample approach to the KL
    divergence term.
    """

    n_samples_kl: int = eqx.field(static=True)
    """The  number of samples to use to approximate the KL divergence term."""

    def __call__(
        self,
        m: TrainingModel,
        x: jnp.ndarray,
        y: jnp.ndarray,
        mask: jnp.ndarray | None,
        context_x: jnp.ndarray | None,
        n_batches_per_epoch: int,
        *,
        key: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        assert context_x is not None
        batch_size, *_ = x.shape
        nll_key, kl_key = jax.random.split(key, 2)
        nll_loss, predf = super().__call__(
            m, x, y, mask, context_x, n_batches_per_epoch, key=nll_key
        )
        # compute KL div
        klq_key, klp_key = jax.random.split(kl_key, 2)
        q = m.variational(context_x, self.n_samples_kl, key=klq_key)  # [O,]
        p = m.prior(context_x, key=klp_key)
        kl_div = tfp.distributions.kl_divergence(q, p)  # [O,]

        kl_loss = kl_div.mean()  # [,]
        kl_loss /= n_batches_per_epoch
        kl_loss /= batch_size
        return nll_loss + kl_loss, predf


class TractableFSVILoss(NLLLoss):
    """
    A loss that performs functions-space variational inference (FSVI) using the tractable approximation of Rudner et al.
    to compute the KL divergence term.
    """

    def __call__(
        self,
        m: TrainingModel,
        x: jnp.ndarray,
        y: jnp.ndarray,
        mask: jnp.ndarray | None,
        context_x: jnp.ndarray | None,
        n_batches_per_epoch: int,
        *,
        key: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        assert context_x is not None
        batch_size, *_ = x.shape
        nll_key, kl_key = jax.random.split(key, 2)
        nll_loss, predf = super().__call__(
            m, x, y, mask, context_x, n_batches_per_epoch, key=nll_key
        )
        # compute KL div
        klq_key, klp_key = jax.random.split(kl_key, 2)
        m_net = m.net
        assert isinstance(m_net, TractableStochasticNet)
        mean, cov = m_net.tractable_f_mean_cov(context_x, key=klq_key)  # [O, M], [O, M, M]
        m_variational_dist = m.variational_dist
        assert isinstance(m_variational_dist, GaussianVariationalDistribution)
        q = m_variational_dist.from_mean_cov(mean, cov)  # [O,]
        p = m.prior(context_x, key=klp_key)
        kl_div = tfp.distributions.kl_divergence(q, p)  # [O,]

        kl_loss = kl_div.mean()  # [,]
        kl_loss /= n_batches_per_epoch
        kl_loss /= batch_size
        return nll_loss + kl_loss, predf
