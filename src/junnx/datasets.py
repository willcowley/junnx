import abc
from dataclasses import dataclass
from typing import Iterator, Sequence

import jax
import jax.numpy as jnp
import tensorflow_probability.substrates.jax.bijectors as tfpb

_EPS = 1e-12


class Dataset:
    """Abstract dataset. Follows the PyTorch Dataset interface."""

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

    def __add__(self, other: "TensorDataset") -> "TensorDataset":
        if not isinstance(other, TensorDataset):
            raise TypeError
        if self.x.shape[1:] != other.x.shape[1:]:
            raise ValueError
        if self.y.shape[1:] != other.y.shape[1:]:
            raise ValueError
        x = jnp.concatenate([self.x, other.x], axis=0)
        y = jnp.concatenate([self.y, other.y], axis=0)
        return TensorDataset(x, y)


class TransformedTensorDataset(TensorDataset):

    def __init__(
        self,
        ds: TensorDataset,
        xbijector: tfpb.Bijector,
        ybijector: tfpb.Bijector | None = None,
    ) -> None:
        self._xbijector = xbijector
        self._ybijector = ybijector
        x = xbijector(ds.x)
        y = ds.y
        if ybijector is not None:
            y = ybijector(y)
        super().__init__(x, y)

    @property
    def xbijector(self) -> tfpb.Bijector:
        return self._xbijector

    @property
    def ybijector(self) -> tfpb.Bijector | None:
        return self._ybijector


@dataclass(frozen=True)
class DataTransformFn(abc.ABC):

    @abc.abstractmethod
    def fit(self, x: jnp.ndarray) -> tfpb.Bijector: ...


@dataclass(frozen=True)
class StandardizeTransformFn(DataTransformFn):

    axis: int | Sequence[int] | None = None
    keepdims: bool = True
    batch_axis: int | None = 0

    def fit(self, x: jnp.ndarray) -> tfpb.Bijector:
        # x: [B, ...]
        mean = jnp.mean(x, axis=self.axis, keepdims=self.keepdims)
        std = jnp.std(x, axis=self.axis, keepdims=self.keepdims) + _EPS
        if self.batch_axis is not None:
            mean = jnp.squeeze(mean, axis=self.batch_axis)
            std = jnp.squeeze(std, axis=self.batch_axis)
        # tfpb.Chain applies bijectors from right to left
        return tfpb.Chain([tfpb.Scale(1 / std), tfpb.Shift(-mean)])


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
        self._key = key

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def batches_per_epoch(self) -> int:
        return len(self.data) // self.batch_size

    @property
    def key(self) -> jnp.ndarray:
        return self._key

    @key.setter
    def key(self, key: jnp.ndarray) -> None:
        self._key = key

    def __iter__(self) -> Iterator[tuple[jnp.ndarray, jnp.ndarray]]:
        # iterates over the data in batches of size `batch_size`
        if self.shuffle:
            idxs = jax.random.permutation(self._key, self.idxs)
            (self._key,) = jax.random.split(self._key, 1)
        else:
            idxs = self.idxs
        start = 0
        end = self._batch_size
        while end <= len(self.idxs):
            perm_idxs = idxs[start:end]
            yield self.data[perm_idxs]
            start = end
            end = start + self.batch_size
