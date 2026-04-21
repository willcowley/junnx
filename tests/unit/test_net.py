import equinox as eqx
import jax.random as jaxr
import numpy.testing as npt
import pytest

from junnx.layers import DenseStochasticLayer
from junnx.net import (
    DenseStochasticNet,
    MCDropoutLeNet,
    MCDropoutMLP,
    StochasticLeNet,
    StochasticNet,
)


def test_dense_stochastic_net():
    key = jaxr.PRNGKey(0)
    key_init, key_x, key_call0, key_call1 = jaxr.split(key, 4)
    net = DenseStochasticNet(
        n_in=3,
        n_out=2,
        n_hidden=32,
        depth=2,
        use_bias=True,
        key=key_init,
    )

    assert net.n_in == 3
    assert net.n_out == 2
    assert net.n_hidden == 32
    assert net.depth == 2
    assert len(net.layers) == 3

    assert net.layers[0].n_in == 3
    assert net.layers[0].n_out == 32
    assert net.layers[1].n_in == 32
    assert net.layers[1].n_out == 32
    assert net.layers[2].n_in == 32
    assert net.layers[2].n_out == 2

    x = jaxr.normal(key_x, shape=(10, 3))

    predict_f_fn = eqx.filter_jit(net.predict_f_samples)
    samples0 = predict_f_fn(x, n_samples=5, key=key_call0)
    assert samples0.shape == (5, 10, 2)

    # check that samples are different
    for i in range(1, 5):
        with pytest.raises(AssertionError):
            npt.assert_allclose(samples0[0], samples0[i])

    # check that different keys give different samples
    samples1 = predict_f_fn(x, n_samples=5, key=key_call1)
    with pytest.raises(AssertionError):
        npt.assert_allclose(samples0, samples1)

    # check that same key gives same samples
    samples2 = predict_f_fn(x, n_samples=5, key=key_call0)
    npt.assert_allclose(samples0, samples2)


def test_dense_stochastic_net_is_tractable():
    key = jaxr.PRNGKey(42)
    key_init, key_x, key_samples = jaxr.split(key, 3)
    net = DenseStochasticNet(
        n_in=3,
        n_out=2,
        n_hidden=32,
        depth=2,
        use_bias=True,
        key=key_init,
    )
    x = jaxr.normal(key_x, shape=(10, 3))
    tractable_f_mean_cov_fn = eqx.filter_jit(net.tractable_f_mean_cov)
    mean, cov = tractable_f_mean_cov_fn(x, key=key_x)

    assert mean.shape == (2, 10)
    assert cov.shape == (2, 10, 10)

    # check last layer
    last_layer = net.last_layer
    assert isinstance(last_layer, DenseStochasticLayer)
    assert last_layer.n_in == 32
    assert last_layer.n_out == 2


def test_dense_mcdropout_net():
    key = jaxr.PRNGKey(0)
    key_init, key_x, key_call0, key_call1 = jaxr.split(key, 4)
    net = MCDropoutMLP(
        n_in=3,
        n_out=2,
        n_hidden=32,
        depth=2,
        use_bias=True,
        key=key_init,
    )

    assert net.n_in == 3
    assert net.n_out == 2
    assert net.n_hidden == 32
    assert net.depth == 2
    assert len(net.layers) == 3

    assert net.layers[0].in_features == 3
    assert net.layers[0].out_features == 32
    assert net.layers[1].in_features == 32
    assert net.layers[1].out_features == 32
    assert net.layers[2].in_features == 32
    assert net.layers[2].out_features == 2

    x = jaxr.normal(key_x, shape=(10, 3))

    predict_f_fn = eqx.filter_jit(net.predict_f_samples)
    samples0 = predict_f_fn(x, n_samples=5, key=key_call0)
    assert samples0.shape == (5, 10, 2)

    # check that samples are different
    for i in range(1, 5):
        with pytest.raises(AssertionError):
            npt.assert_allclose(samples0[0], samples0[i])

    # check that different keys give different samples
    samples1 = predict_f_fn(x, n_samples=5, key=key_call1)
    with pytest.raises(AssertionError):
        npt.assert_allclose(samples0, samples1)

    # check that same key gives same samples
    samples2 = predict_f_fn(x, n_samples=5, key=key_call0)
    npt.assert_allclose(samples0, samples2)


@pytest.mark.parametrize("model_type", ["stochastic", "mcdropout"])
def test_lenet(model_type: str):
    key = jaxr.PRNGKey(42)
    if model_type == "stochastic":
        m: StochasticNet = StochasticLeNet(key=key)
    elif model_type == "mcdropout":
        m = MCDropoutLeNet(key=key)
    else:
        raise ValueError(f"Unknown model: {model_type}")

    image = jaxr.normal(key, (1, 28, 28))

    key0, key1 = jaxr.split(key, 2)
    out0 = m(image, key=key0)

    assert out0.shape == (10,)

    out1 = m(image, key=key1)

    with pytest.raises(AssertionError):
        npt.assert_allclose(out0, out1)

    out2 = m(image, key=key0)
    npt.assert_allclose(out0, out2)

    out_samples = m.predict_f_samples(image[None], n_samples=3, key=key0)
    assert out_samples.shape == (3, 1, 10)


def test_stochastic_lenet_is_tractable():
    key = jaxr.PRNGKey(42)
    key_init, key_image, key_call = jaxr.split(key, 3)

    m = StochasticLeNet(key=key_init)
    images = jaxr.uniform(key_image, (8, 1, 28, 28))

    mean, cov = m.tractable_f_mean_cov(images, key=key_call)

    assert mean.shape == (10, 8)
    assert cov.shape == (10, 8, 8)

    # check last layer
    last_layer = m.last_layer
    assert isinstance(last_layer, DenseStochasticLayer)
    assert last_layer.n_in == 84
    assert last_layer.n_out == 10
