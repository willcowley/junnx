import jax
import jax.numpy as jnp
import jax.random as jaxr
import numpy.testing as npt
import pytest

from junnx.layers import DenseStochasticLayer


def test_dense_stochastic_layer_smoke() -> None:

    key_init = jaxr.PRNGKey(0)
    n_in = 4
    n_out = 3
    use_bias = True
    layer = DenseStochasticLayer(n_in, n_out, use_bias, key=key_init)

    assert layer.w_mean.shape == layer.w_log_var.shape == (n_out, n_in)
    if use_bias:
        assert layer.bias is not None
        assert layer.bias.shape == (n_out,)
    else:
        assert layer.bias is None
    assert layer.n_in == n_in
    assert layer.n_out == n_out
    assert layer.use_bias == use_bias


def test_dense_stochastic_layer_is_stochastic() -> None:

    key_init = jaxr.PRNGKey(0)
    n_in = 4
    n_out = 3
    use_bias = True
    layer = DenseStochasticLayer(n_in, n_out, use_bias, key=key_init)

    x = jaxr.normal(jaxr.PRNGKey(1), shape=(n_in,))
    key1, key2 = jaxr.split(jaxr.PRNGKey(2), 2)

    out1 = layer(x, key1)
    out2 = layer(x, key2)

    assert out1.shape == (n_out,)
    assert out2.shape == (n_out,)
    with pytest.raises(AssertionError):
        npt.assert_allclose(out1, out2)


def test_dense_stochastic_layer_initialisation() -> None:

    key = jaxr.PRNGKey(0)
    key_init, key_call = jaxr.split(key, 2)
    n_in = 400
    n_out = 400
    use_bias = False
    layer = DenseStochasticLayer(n_in, n_out, use_bias, key=key_init)

    x = jnp.eye(n_in)
    call_fn = jax.jit(jax.vmap(lambda _x: layer(_x, key_call)))
    out = call_fn(x)

    assert out.shape == (n_in, n_out)
    npt.assert_allclose(out.std(), jnp.sqrt(2 / (n_in + n_out)), atol=1e-3, rtol=1e-3)
