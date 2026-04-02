from typing import Optional

import jax.numpy as jnp
from sklearn.datasets import make_moons as sk_make_moons

from junnx.datasets import TensorDataset


class MakeMoonsDataset(TensorDataset):
    """Dataset dervied from the sci-kit learn `make_moons` function."""

    def __init__(self, n_samples: int | tuple[int, int], noise: Optional[float], seed: int):

        x, y = sk_make_moons(n_samples=n_samples, noise=noise, shuffle=True, random_state=seed)
        # centre x on origin
        x[..., 1:] -= 0.25
        x[..., :1] -= 0.5
        x = jnp.asarray(x)  # (N, 2)
        y = jnp.asarray(y)[:, None]  # (N, 1)
        super().__init__(x, y)
