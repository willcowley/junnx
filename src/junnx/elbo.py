import equinox as eqx
import jax
import jax.numpy as jnp
import optax
import tensorflow_probability.substrates.jax as tfp

from junnx.train import TrainingModel


@eqx.filter_value_and_grad
def elbo(
    trainable: TrainingModel,
    static: TrainingModel,
    x: jnp.ndarray,
    y: jnp.ndarray,
    n_samples_nll: int,
    n_samples_kl: int,
    n_batches_per_epoch: int,
    *,
    key: jnp.ndarray,
) -> jnp.ndarray:
    """
    Computes the function-space variational inference loss for a batch of data.

    Args:
        trainable: The trainable part(s) of the model that gradients are to be computed with
            respect to.
        static: The static part(s) of the model that is not updated during training.
        x: The input data batch.
        y: The target data batch.
        n_samples_nll: The number of model realisations to use when estimating the negative
            log-likelihood
        n_samples_kl: The number of model realisations to use when estimating the KL divergence
        n_batches_per_epoch: The number of data batches per epoch, used to scale the KL
            divergence.

    Returns:
        The scalar per-batch loss.
    """
    # x: [N, D]
    # y: [N, O]
    batch_size, *_ = x.shape
    m = eqx.combine(trainable, static)

    nll_key, kl_key = jax.random.split(key, 2)

    # compute NLL
    predf = m.net.predict_f_samples(x, n_samples_nll, key=nll_key)  # [SN, N, O]
    predy = m.likelihood(predf)
    nll_loss = -predy.log_prob(y).sum(axis=-1).mean()  # [,]  per-batch NLL

    # compute KL div
    context_key, klq_key, klp_key = jax.random.split(kl_key, 3)
    # TODO: move context point generation outside of jax.jit
    context_points = m.sampler(context_key)  # [M, D]
    q = m.variational(context_points, n_samples_kl, key=klq_key)  # [O,]
    p = m.prior(context_points, n_samples_kl, key=klp_key)
    kl_div = tfp.distributions.kl_divergence(q, p)  # [O,]

    kl_loss = kl_div.mean()  # [,]
    kl_loss /= n_batches_per_epoch  # per-batch KL

    return (nll_loss + kl_loss) / batch_size  # [,]


@eqx.filter_jit
def train_step(
    trainable: TrainingModel,
    static: TrainingModel,
    x: jnp.ndarray,
    y: jnp.ndarray,
    n_samples_nll: int,
    n_samples_kl: int,
    n_batches_per_epoch: int,
    opt: optax.GradientTransformation,
    opt_state: optax.OptState,
    *,
    key: jnp.ndarray,
) -> tuple[jnp.ndarray, TrainingModel, optax.OptState]:
    """
    Performs a single function-space variational inference training step given a batch of data.
    Computes the scalar loss and updates the trainable parameters accordingly using the
    provided optimiser and optimiser state.

     Args:
        trainable: The trainable part(s) of the model that gradients are to be computed with
            respect to.
        static: The static part(s) of the model that is not updated during training.
        x: The input data batch.
        y: The target data batch.
        n_samples_nll: The number of model realisations to use when estimating the negative
            log-likelihood
        n_samples_kl: The number of model realisations to use when estimating the KL divergence
        n_batches_per_epoch: The number of data batches per epoch, used to scale the KL
            divergence.
        opt: The optax optimiser to use for updating the trainable parameters.
        opt_state: The current state of the optimiser.
        key: JAX LRNG key to seed sampling (keyword-only argument).

    Returns:
        A tuple containing:
        - The scalar per-batch loss.
        - The updated trainable part(s) of the model.
        - The updated optimiser state.
    """
    loss, grads = elbo(
        trainable, static, x, y, n_samples_nll, n_samples_kl, n_batches_per_epoch, key=key
    )
    update, opt_state = opt.update(grads, opt_state)
    trainable = eqx.apply_updates(trainable, update)
    return loss, trainable, opt_state
