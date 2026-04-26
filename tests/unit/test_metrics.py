from typing import Callable

import jax.nn
import jax.numpy as jnp
import jax.random as jaxr
import numpy.testing as npt
import pytest
import sklearn.metrics
import tensorflow_probability.substrates.jax as tfp

from junnx.metrics import ECE, MSE, NLL, RMSE, Accuracy, Brier, Metric, _Average


def test_ece() -> None:

    probs = jnp.array(
        [
            [0.15, 0.55, 0.15, 0.15],
            [0.24, 0.24, 0.27, 0.24],
            [0.0, 0.0, 0.01, 0.99],
            [0.0, 0.0, 0.01, 0.99],
        ]
    )

    labels = jnp.array([1, 0, 3, 2])

    ece = ECE.from_samples(probs, labels)
    result = ece.compute()

    # |0 - 0.27|  [2]
    # |1 - 0.55|  [5]
    # |1 - 1.98|  [9]
    expected = jnp.array([0.0, 0.0, 0.27, 0.0, 0.0, 0.45, 0, 0, 0, 0.98]).sum() / 4

    npt.assert_allclose(result, expected)

    ece0 = ECE.empty()
    ece1 = ECE.from_samples(probs[:2], labels[:2])
    ece2 = ece0.merge(ece1)
    ece3 = ECE.from_samples(probs[2:], labels[2:])
    ece4 = ece2.merge(ece3)

    result = ece4.compute()
    npt.assert_allclose(result, expected)


@pytest.mark.parametrize("metric_cls", [MSE, RMSE, NLL, Accuracy, ECE, Brier])
def test_average_metric_empty(metric_cls: type[_Average]) -> None:
    metric = metric_cls.empty()

    npt.assert_allclose(metric.total, 0)
    npt.assert_allclose(metric.count, 0)


def _from_batches(metrics_cls: type[Metric], preds, trues) -> jnp.ndarray:
    # preds: [Nbatches, Nbatch, Nout]
    # trues: [Nbatches, Nbatch, Nout] or [Nbatches, Nbatch] if classification
    m = metrics_cls.empty()
    for pred_batch, true_batch in zip(preds, trues):
        _m = metrics_cls.from_samples(pred_batch, true_batch)
        m = m.merge(_m)
    return m.compute()


@pytest.mark.parametrize(
    "metric_cls, impl",
    [
        (MSE, sklearn.metrics.mean_squared_error),
        (RMSE, sklearn.metrics.root_mean_squared_error),
    ],
)
def test_regression_compute(
    metric_cls: type[Metric], impl: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]
) -> None:
    key = jaxr.PRNGKey(0)
    key_pred, key_true = jaxr.split(key, 2)

    preds = jaxr.normal(key_pred, shape=(100, 1))
    trues = jaxr.normal(key_true, shape=(100, 1))

    metric = metric_cls.from_samples(preds, trues)
    result = metric.compute()

    expected = impl(trues, preds)

    npt.assert_allclose(result, expected)

    result_from_batches = _from_batches(
        metric_cls, preds.reshape(5, 20, 1), trues.reshape(5, 20, 1)
    )
    npt.assert_allclose(result_from_batches, expected, atol=1e-6, rtol=1e-6)


def _gaussian_nll(x, mu, sigma):
    return 0.5 * jnp.square((x - mu) / sigma) + jnp.log(sigma) + 0.5 * jnp.log(2 * jnp.pi)


def test_nll_compute() -> None:

    key = jaxr.PRNGKey(0)
    key_pred, key_true = jaxr.split(key, 2)

    preds_mean = jaxr.normal(key_pred, shape=(100, 1))
    pred_std = jnp.ones(shape=(1,))
    ydist = tfp.distributions.MultivariateNormalDiag(loc=preds_mean, scale_diag=pred_std)
    trues = ydist.sample(seed=key_true)

    metric = NLL.from_ydist(ydist, trues)
    result = metric.compute()

    expected = _gaussian_nll(trues, preds_mean, pred_std).mean()

    npt.assert_allclose(result, expected)


def test_nll_raises() -> None:
    key = jaxr.PRNGKey(0)
    key_pred, key_true = jaxr.split(key, 2)

    preds_mean = jaxr.normal(key_pred, shape=(100, 1))
    pred_std = jnp.ones(shape=(1,))
    ydist = tfp.distributions.MultivariateNormalDiag(loc=preds_mean, scale_diag=pred_std)
    preds = ydist.sample(seed=key_pred)
    trues = ydist.sample(seed=key_true)

    with pytest.raises(NotImplementedError):
        NLL.from_samples(preds, trues)


@pytest.mark.parametrize(
    "metric_cls, impl",
    [
        (Accuracy, sklearn.metrics.accuracy_score),
        (Brier, sklearn.metrics.brier_score_loss),
    ],
)
def test_classification_compute(
    metric_cls: type[Metric], impl: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]
) -> None:
    key = jaxr.PRNGKey(0)
    key_pred, key_true = jaxr.split(key, 2)

    preds = jax.nn.softmax(jaxr.normal(key_pred, shape=(100, 3)), axis=-1)
    trues = jaxr.randint(key_true, shape=(100,), minval=0, maxval=3)

    metric = metric_cls.from_samples(preds, trues)
    result = metric.compute()

    if isinstance(metric, Accuracy):
        impl_preds = jnp.argmax(preds, axis=-1)
    else:
        impl_preds = preds
    expected = impl(trues, impl_preds)

    npt.assert_allclose(result, expected)

    result_from_batches = _from_batches(
        metric_cls, preds.reshape(5, 20, 3), trues.reshape(5, 20)
    )
    npt.assert_allclose(result_from_batches, expected, atol=1e-6, rtol=1e-6)
