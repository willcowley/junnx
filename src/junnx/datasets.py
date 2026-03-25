from enum import Enum, unique
from pathlib import Path
from typing import Iterator, Mapping, Optional, Protocol

import jax
import jax.numpy as jnp
from sklearn.datasets import make_moons as sk_make_moons


@unique
class _ToyData(str, Enum):
    SNELSON05 = "snelson"


class _ToyDataLoader(Protocol):
    def __call__(self) -> tuple[jnp.ndarray, jnp.ndarray]: ...


def _load_snelson05() -> tuple[jnp.ndarray, jnp.ndarray]:
    filename = str(Path(__file__).parent / "data" / "snelson05.npy")
    data = jnp.load(filename)
    x = data[:, :1] * 2 / 3 - 2  # scale to [-2, 2]
    y = data[:, 1:] * 4 / 3 + 2 / 3  # scale to [-2, 2]
    mask = (x >= -1) & (x < 0)  # remove points in [-1, 0)
    return x[~mask][:, None], y[~mask][:, None]


_TOY_DATA_FN: Mapping[_ToyData, _ToyDataLoader] = {_ToyData.SNELSON05: _load_snelson05}


class Dataset:
    """Abstract dataset. Follows the PyTorch Dataset inteface."""

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, idx: jnp.ndarray):
        # idx [Nidxs,]  integer array
        raise NotImplementedError


class TensorDataset(Dataset):
    """Dataset wrapping x and y tensors."""

    def __init__(self, x: jnp.ndarray, y: jnp.ndarray) -> None:
        xshape, _ = x.shape
        yshape, _ = y.shape
        if xshape != yshape:
            raise ValueError(
                f"Expected x and y to have the same leading dimension, got {x.shape} and {y.shape}"
            )
        self._x = x
        self._y = y

    @property
    def x(self) -> jnp.ndarray:
        return self._x

    @property
    def y(self) -> jnp.ndarray:
        return self._y

    def __len__(self) -> int:
        return len(self._x)

    def __getitem__(self, idx: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        return self._x[idx], self._y[idx]


class SnelsonDataset(TensorDataset):
    """
    Dataset derived from Figure 1 of Snelson & Gaharami 2005 "Sparse Gaussian Processes using
    Pseudo-inputs".
    """

    def __init__(self) -> None:
        x, y = _TOY_DATA_FN[_ToyData.SNELSON05]()
        super().__init__(x, y)


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


class DataLoader:
    """
    DataLoader samples from a Dataset in batches and provides an iterable over the given
    dataset.

    Args:
        data: Dataset to sample from.
        batch_size: Number of samples per batch.
        shuffle: Whether to shuffle the data at the start of each iteration.
        key: JAX random key for shuffling the data.
    """

    def __init__(
        self,
        data: Dataset,
        batch_size: int = 32,
        shuffle: bool = True,
        *,
        key: jnp.ndarray,
    ) -> None:
        self.data = data
        self.idxs = jnp.arange(len(data))
        self._batch_size = batch_size
        self.shuffle = shuffle
        self.key = key

    @property
    def batch_size(self) -> int:
        return self._batch_size

    def __iter__(self) -> Iterator[tuple[jnp.ndarray, jnp.ndarray]]:
        # iterates over the data in batches of size `batch_size`
        if self.shuffle:
            idxs = jax.random.permutation(self.key, self.idxs)
            (self.key,) = jax.random.split(self.key, 1)
        else:
            idxs = self.idxs
        start = 0
        end = self._batch_size
        while end <= len(self.idxs):
            perm_idxs = idxs[start:end]
            yield self.data[perm_idxs]
            start = end
            end = start + self.batch_size
