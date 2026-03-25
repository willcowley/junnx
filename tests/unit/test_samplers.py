import jax.random as jaxr
import numpy.testing as npt

from junnx.samplers import UniformSampler


def test_uniform_sampler() -> None:

    key = jaxr.PRNGKey(0)
    key_x, key_call = jaxr.split(key, 2)
    low = (-1.0, 1.0)
    high = (1.0, 3.0)

    sampler = UniformSampler(2, 8, low, high)

    assert sampler.n_dim == 2
    assert sampler.n_samples == 8

    samples = sampler(key_call)
    assert samples.shape == (8, 2)

    samples01 = (samples - sampler.low) / (sampler.high - sampler.low)
    npt.assert_array_less(samples01, 1.0)
    npt.assert_array_less(-samples01, 0.0)
