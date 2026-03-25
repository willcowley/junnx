import jax.numpy as jnp
import jax.random as jaxr
import numpy.testing as npt
import tensorflow_probability.substrates.jax as tfp

from junnx.likelihoods import BernoulliLikelihood, CategoricalLikelihood, GaussianLikelihood


def test_gaussian_likelihood() -> None:

    scale_init = (0.1, 0.2)
    bijector = tfp.bijectors.Softplus()
    likelihood = GaussianLikelihood(scale_init=scale_init, bijector=bijector)

    dummy_x = jaxr.normal(jaxr.PRNGKey(0), shape=(32, 16, 2))  # [S, N, O]

    dist = likelihood(dummy_x)

    assert isinstance(dist, tfp.distributions.MultivariateNormalDiag)
    assert dist.loc.shape == (32, 16, 2)
    assert dist.scale.diag.shape == (2,)

    npt.assert_allclose(dist.loc, dummy_x)
    npt.assert_allclose(dist.scale.diag, jnp.asarray(scale_init))


def test_bernoulli_likelihood() -> None:

    likelihood = BernoulliLikelihood()

    dummy_x = jaxr.normal(jaxr.PRNGKey(0), shape=(32, 16, 1))  # [S, N, O]

    dist = likelihood(dummy_x)

    _ = dist.log_prob(dummy_x)
    assert isinstance(dist, tfp.distributions.Bernoulli)
    assert dist.probs.shape == (32, 16, 1)

    npt.assert_allclose(dist.probs, likelihood.probs(dummy_x)[..., 1:])
    # probs in (0, 1)
    npt.assert_array_less(dist.probs, 1.0)
    npt.assert_array_less(-dist.probs, 0.0)


def test_categorical_likelihood() -> None:

    likelihood = CategoricalLikelihood()

    dummy_x = jaxr.normal(jaxr.PRNGKey(0), shape=(32, 16, 4))  # [S, N, O]

    dist = likelihood(dummy_x)
    _ = dist.log_prob(dummy_x)

    assert isinstance(dist, tfp.distributions.Categorical)
    assert dist.probs.shape == (32, 16, 1, 4)

    npt.assert_allclose(dist.probs[..., 0, :], likelihood.probs(dummy_x))
    # probs in (0, 1) and sum to 1 along last axis
    npt.assert_array_less(dist.probs, 1.0)
    npt.assert_array_less(-dist.probs, 0.0)
    npt.assert_allclose(jnp.sum(dist.probs, axis=-1), 1.0, atol=1e-6, rtol=0.0)
