import jax
import jax.numpy as jnp
import jax.random as jaxr
import numpy.testing as npt
import pytest
import tensorflow_probability.substrates.jax as tfp

from junnx.priors import DirichletPrior


@pytest.mark.parametrize("nd", [2, 4])
def test_dirichlet_prior(nd: int) -> None:
    concentration = jnp.ones(shape=(nd,))
    prior = DirichletPrior(concentration=concentration)

    key = jax.random.PRNGKey(0)
    key_x, key_call = jaxr.split(key, 2)
    x = jaxr.normal(key_x, shape=(16, nd))  # [N, O]

    dist = prior(x, 5, key=key_call)

    assert isinstance(dist, tfp.distributions.Distribution)
    assert dist.concentration.shape == x.shape[-1:]
    npt.assert_allclose(dist.concentration, concentration)
