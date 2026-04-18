import jax.numpy as jnp
import numpy.testing as npt

from junnx.metrics import ECE


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
