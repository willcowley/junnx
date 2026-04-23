import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jaxr
import numpy.testing as npt
import pytest
import tensorflow_probability.substrates.jax as tfp

from junnx.net import DenseStochasticNet
from junnx.priors import DirichletPrior, TractableStochasticNetPrior


@pytest.mark.parametrize("nd", [2, 4])
def test_dirichlet_prior(nd: int) -> None:
    concentration = jnp.ones(shape=(nd,))
    prior = DirichletPrior(concentration=concentration)

    key = jax.random.PRNGKey(0)
    key_x, key_call = jaxr.split(key, 2)
    x = jaxr.normal(key_x, shape=(16, nd))  # [N, O]

    dist = prior(x, key=key_call)

    assert isinstance(dist, tfp.distributions.Distribution)
    assert dist.concentration.shape == x.shape[-1:]
    npt.assert_allclose(dist.concentration, concentration)


def test_tractable_prior() -> None:
    key = jax.random.PRNGKey(0)
    key_init, key_call = jaxr.split(key, 2)
    net = DenseStochasticNet(n_in=1, n_hidden=8, n_out=1, depth=2, key=key_init)
    prior = TractableStochasticNetPrior(net)

    x = jax.random.normal(key, shape=(16, 1))
    dist = prior(x, key=key_call)
    assert isinstance(dist, tfp.distributions.MultivariateNormalTriL)

    assert dist.loc.shape == (1, 16)
    assert dist.scale_tril.shape == (1, 16, 16)


def test_tractable_prior_updates_logvar() -> None:
    key = jax.random.PRNGKey(0)
    key_init, key_call = jaxr.split(key, 2)
    net = DenseStochasticNet(n_in=1, n_hidden=8, n_out=1, depth=2, key=key_init)
    prior = TractableStochasticNetPrior(net, log_var=0.0)

    npt.assert_allclose(prior.net.last_layer.w_log_var, 0.0)
    with pytest.raises(AssertionError):
        npt.assert_allclose(prior.net.last_layer.w_log_var, net.last_layer.w_log_var)


def test_tractable_prior_net_is_frozen() -> None:
    key = jax.random.PRNGKey(0)
    key_init, key_call = jaxr.split(key, 2)
    net = DenseStochasticNet(n_in=1, n_hidden=8, n_out=1, depth=2, key=key_init)
    prior = TractableStochasticNetPrior(net)

    prior_net = prior.net
    assert isinstance(prior_net, DenseStochasticNet)
    npt.assert_allclose(net.layers[0].w_mean, prior_net.layers[0].w_mean)

    # update net weights but prior net weights remain the same
    net = eqx.tree_at(lambda n: n.layers[0].w_mean, net, net.layers[0].w_mean + 2.0)
    npt.assert_allclose(net.layers[0].w_mean, prior_net.layers[0].w_mean + 2.0)
