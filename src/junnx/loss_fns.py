import abc

import equinox as eqx
import jax
import jax.numpy as jnp
import tensorflow_probability.substrates.jax as tfp

from junnx.model import TrainingModel
from junnx.net import TractableStochasticNet
from junnx.variational import GaussianVariationalDistribution


class LossFn(eqx.Module):

    @abc.abstractmethod
    def __call__(
        self,
        m: TrainingModel,
        x: jnp.ndarray,
        y: jnp.ndarray,
        context_x: jnp.ndarray | None,
        n_batches_per_epoch: int,
        *,
        key: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]: ...


class NLLLoss(LossFn):

    n_samples_nll: int = eqx.field(static=True)

    def __call__(
        self,
        m: TrainingModel,
        x: jnp.ndarray,
        y: jnp.ndarray,
        context_x: jnp.ndarray | None,
        n_batches_per_epoch: int,
        *,
        key: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        batch_size, *_ = x.shape

        # compute NLL
        predf = m.net.predict_f_samples(x, self.n_samples_nll, key=key)  # [SN, N, O]
        predy = m.likelihood(predf)
        nll_loss = -predy.log_prob(y[None]).sum(axis=-1).mean()  # [,]  per-batch NLL

        return nll_loss / batch_size, predf  # [,], [SN, N, O]


class SampleFSVILoss(NLLLoss):

    n_samples_kl: int = eqx.field(static=True)

    def __call__(
        self,
        m: TrainingModel,
        x: jnp.ndarray,
        y: jnp.ndarray,
        context_x: jnp.ndarray | None,
        n_batches_per_epoch: int,
        *,
        key: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        assert context_x is not None
        batch_size, *_ = x.shape
        nll_key, kl_key = jax.random.split(key, 2)
        nll_loss, predf = super().__call__(
            m, x, y, context_x, n_batches_per_epoch, key=nll_key
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

    def __call__(
        self,
        m: TrainingModel,
        x: jnp.ndarray,
        y: jnp.ndarray,
        context_x: jnp.ndarray | None,
        n_batches_per_epoch: int,
        *,
        key: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        assert context_x is not None
        batch_size, *_ = x.shape
        nll_key, kl_key = jax.random.split(key, 2)
        nll_loss, predf = super().__call__(
            m, x, y, context_x, n_batches_per_epoch, key=nll_key
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
