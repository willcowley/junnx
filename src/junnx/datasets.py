from typing import Iterator

import jax
import jax.numpy as jnp


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
        xshape, *_ = x.shape
        yshape, *_ = y.shape
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

    @property
    def batches_per_epoch(self) -> int:
        return len(self.data) // self.batch_size

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
