from pathlib import Path

import jax.numpy as jnp

from junnx.datasets import TensorDataset


def _load_snelson05() -> tuple[jnp.ndarray, jnp.ndarray]:
    filename = str(Path(__file__).parent / "snelson05.npy")
    data = jnp.load(filename)
    x = data[:, :1] * 2 / 3 - 2  # scale to [-2, 2]
    y = data[:, 1:] * 4 / 3 + 2 / 3  # scale to [-2, 2]
    mask = (x >= -1) & (x < 0)  # remove points in [-1, 0)
    return x[~mask][:, None], y[~mask][:, None]


class SnelsonDataset(TensorDataset):
    """
    Dataset derived from Figure 1 of Snelson & Gaharami 2005 "Sparse Gaussian Processes using
    Pseudo-inputs".
    """

    def __init__(self) -> None:
        x, y = _load_snelson05()
        super().__init__(x, y)
