import equinox as eqx
import jax.random as jaxr
import numpy.testing as npt
import pytest

from junnx.net import DenseStochasticNet, MCDropoutMLP, StochasticLeNet


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


def test_stochastic_lenet():
    key = jaxr.PRNGKey(42)
    m = StochasticLeNet(key=key)

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
