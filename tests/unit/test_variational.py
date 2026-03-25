import jax.numpy as jnp
import jax.random as jaxr
import numpy.testing as npt
import tensorflow_probability.substrates.jax as tfp

from junnx.variational import DirichletVariationalDistribution, GaussianVariationalDistribution


def test_gaussian_variational_distribution():

    key = jaxr.PRNGKey(0)
    key_mean, key_cov, key_samples = jaxr.split(key, 3)
    mean = jaxr.normal(key_mean, shape=(1, 16))  # [1, N]
    cov = jaxr.normal(key_cov, shape=(16, 16))
    cov = cov @ cov.T + 16 * jnp.eye(16)  # [N, N]
    scale_tril = jnp.linalg.cholesky(cov)[None]  # [1, N, N]

    target_dist = tfp.distributions.MultivariateNormalTriL(loc=mean, scale_tril=scale_tril)

    f_samples = target_dist.sample(2_048, key_samples)  # [S, O, N]
    f_samples = jnp.swapaxes(f_samples, 1, 2)  # [S, N, O]

    variational_dist = GaussianVariationalDistribution()
    q_dist = variational_dist(f_samples)
    assert isinstance(q_dist, tfp.distributions.MultivariateNormalTriL)
    assert q_dist.loc.shape == mean.shape
    assert q_dist.scale_tril.shape == scale_tril.shape

    kl_div = tfp.distributions.kl_divergence(q_dist, target_dist)
    assert kl_div.shape == (1,)
    npt.assert_array_less(kl_div, 0.04)


def test_dirichlet_variational_distribution():

    key = jaxr.PRNGKey(0)
    key_samples = jaxr.split(key, 1)[0]
    concentration = jnp.array([[0.5, 1.0, 2.0]])  # [1, O]
    target_dist = tfp.distributions.Dirichlet(concentration=concentration)

    exp_f_samples = target_dist.sample(2_048, key_samples)  # [S, N, O]
    f_samples = jnp.log(exp_f_samples)

    variational_dist = DirichletVariationalDistribution()
    q_dist = variational_dist(f_samples)
    assert isinstance(q_dist, tfp.distributions.Dirichlet)
    assert q_dist.concentration.shape == (1, 3)

    kl_div = tfp.distributions.kl_divergence(q_dist, target_dist)
    assert kl_div.shape == (1,)
    npt.assert_array_less(kl_div, 2e-3)
