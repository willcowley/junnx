import equinox as eqx
import jax.numpy as jnp
import jax.random as jaxr
import numpy.testing as npt
import pytest
import tensorflow_probability.substrates.jax as tfp

from junnx.likelihoods import GaussianLikelihood
from junnx.loss_fns import LossFn, NLLLoss, SampleFSVILoss, TractableFSVILoss
from junnx.model import TrainingModel
from junnx.net import DenseStochasticNet
from junnx.priors import Matern52Prior
from junnx.variational import GaussianVariationalDistribution


@pytest.fixture(scope="module")
def model_and_data() -> tuple[TrainingModel, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    key = jaxr.PRNGKey(0)
    key_init, key_x, key_context_x, key_y = jaxr.split(key, 4)
    model = TrainingModel(
        net=DenseStochasticNet(n_in=2, n_out=1, n_hidden=8, depth=1, key=key_init),
        likelihood=GaussianLikelihood(scale_init=(1.0,), bijector=tfp.bijectors.Softplus()),
        prior=Matern52Prior(lengthscales=(1.0, 1.0)),
        variational_dist=GaussianVariationalDistribution(),
    )

    x_batch = jaxr.normal(key_x, (32, 2))
    context_x_batch = jaxr.normal(key_context_x, (32, 2))
    y_batch = jaxr.normal(key_y, (32, 1))
    return model, x_batch, y_batch, context_x_batch


@pytest.fixture(
    name="fsvi_loss_fn",
    scope="module",
    params=[SampleFSVILoss(4, 4), TractableFSVILoss(4)],
    ids=["sample-fsvi", "tractable-fsvi"],
)
def _fsvi_loss_fn(request: pytest.FixtureRequest) -> LossFn:
    return request.param


def test_nll_loss(
    model_and_data: tuple[TrainingModel, jnp.ndarray, jnp.ndarray, jnp.ndarray],
) -> None:
    key = jaxr.PRNGKey(42)
    model, x_batch, y_batch, _ = model_and_data
    loss_fn = NLLLoss(n_samples_nll=4)

    loss, predf = loss_fn(model, x_batch, y_batch, None, None, 1, key=key)

    assert loss.shape == ()
    assert predf.shape == (4, 32, 1)


class _NormalLikelihood(GaussianLikelihood):

    def __call__(self, x: jnp.ndarray) -> tfp.distributions.Normal:
        # x: [S, N, O]
        return tfp.distributions.Normal(loc=x, scale=self.scale)


def test_nll_loss_masked(
    model_and_data: tuple[TrainingModel, jnp.ndarray, jnp.ndarray, jnp.ndarray],
) -> None:
    key = jaxr.PRNGKey(42)
    model, x_batch, y_batch, _ = model_and_data

    model = eqx.tree_at(
        lambda m: m.likelihood,
        model,
        _NormalLikelihood(scale_init=(1.0,), bijector=tfp.bijectors.Softplus()),
    )

    loss_fn = NLLLoss(n_samples_nll=4)

    mask = jnp.ones_like(y_batch)
    loss, _ = loss_fn(model, x_batch, y_batch, None, None, 1, key=key)
    loss_masked, _ = loss_fn(model, x_batch, y_batch, mask, None, 1, key=key)
    npt.assert_allclose(loss, loss_masked)

    mask = jnp.ones_like(y_batch)
    mask = mask.at[:16].set(0.0)
    loss, _ = loss_fn(model, x_batch[16:], y_batch[16:], None, None, 1, key=key)
    loss_masked, _ = loss_fn(
        model, x_batch, y_batch.at[:16].set(jnp.nan), mask, None, 1, key=key
    )
    npt.assert_allclose(loss, loss_masked)


def test_nll_loss_masked_gradable(
    model_and_data: tuple[TrainingModel, jnp.ndarray, jnp.ndarray, jnp.ndarray],
) -> None:

    key = jaxr.PRNGKey(42)
    model, x_batch, y_batch, _ = model_and_data

    model = eqx.tree_at(
        lambda m: m.likelihood,
        model,
        _NormalLikelihood(scale_init=(1.0,), bijector=tfp.bijectors.Softplus()),
    )

    loss_fn = NLLLoss(n_samples_nll=4)

    fn = eqx.filter_jit(eqx.filter_value_and_grad(loss_fn, has_aux=True))
    mask = jnp.ones_like(y_batch)
    mask = mask.at[:16].set(0.0)
    (loss, _), grads = fn(model, x_batch, y_batch.at[:16].set(jnp.nan), mask, None, 1, key=key)
    # (loss, _), grads = fn(model, x_batch, y_batch, mask, None, 1, key=key)

    assert not jnp.any(jnp.isnan(grads.likelihood.scale)).item()


def test_fsvi_loss(
    model_and_data: tuple[TrainingModel, jnp.ndarray, jnp.ndarray, jnp.ndarray],
    fsvi_loss_fn: LossFn,
) -> None:
    key = jaxr.PRNGKey(42)
    model, x_batch, y_batch, context_x_batch = model_and_data

    loss, predf = fsvi_loss_fn(model, x_batch, y_batch, None, context_x_batch, 1, key=key)

    assert loss.shape == ()
    assert predf.shape == (4, 32, 1)


def test_fsvi_loss_raises(
    model_and_data: tuple[TrainingModel, jnp.ndarray, jnp.ndarray, jnp.ndarray],
    fsvi_loss_fn: LossFn,
) -> None:
    key = jaxr.PRNGKey(42)
    model, x_batch, y_batch, _ = model_and_data

    with pytest.raises(AssertionError):
        _ = fsvi_loss_fn(model, x_batch, y_batch, None, 1, key=key)


def test_fsvi_kl_loss_positive(
    model_and_data: tuple[TrainingModel, jnp.ndarray, jnp.ndarray, jnp.ndarray],
    fsvi_loss_fn: LossFn,
) -> None:

    key = jaxr.PRNGKey(42)
    key_nll, _ = jaxr.split(key, 2)
    model, x_batch, y_batch, context_x_batch = model_and_data

    nll_loss, _ = NLLLoss(n_samples_nll=4)(model, x_batch, y_batch, None, 1, key=key_nll)

    fsvi_loss, _ = fsvi_loss_fn(model, x_batch, y_batch, context_x_batch, 1, key=key)

    kl_loss = fsvi_loss - nll_loss

    npt.assert_array_less(-kl_loss, 0.0)


def test_fsvi_kl_loss_scales(
    model_and_data: tuple[TrainingModel, jnp.ndarray, jnp.ndarray, jnp.ndarray],
    fsvi_loss_fn: LossFn,
) -> None:

    key = jaxr.PRNGKey(42)
    key_nll, _ = jaxr.split(key, 2)  # mimics split within fsvi losses
    model, x_batch, y_batch, context_x_batch = model_and_data

    nll_loss, _ = NLLLoss(n_samples_nll=4)(model, x_batch, y_batch, None, None, 1, key=key_nll)

    fsvi_loss1, _ = fsvi_loss_fn(model, x_batch, y_batch, None, context_x_batch, 1, key=key)

    kl_loss1 = fsvi_loss1 - nll_loss

    fsvi_loss2, _ = fsvi_loss_fn(model, x_batch, y_batch, None, context_x_batch, 2, key=key)

    kl_loss2 = fsvi_loss2 - nll_loss

    npt.assert_allclose(kl_loss1, 2 * kl_loss2, rtol=1e-6, atol=1e-6)
