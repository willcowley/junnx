import jax.numpy as jnp
import tensorflow_datasets as tfds

from junnx.datasets import TensorDataset


def _process_mnist(x, y) -> tuple[jnp.ndarray, jnp.ndarray]:
    x = jnp.asarray(x) / 255.0  # (N, 28, 28, 1)
    x = x.transpose(0, 3, 1, 2)  # (N, 1, 28, 28)  # eqx expects channel first format
    y = jnp.asarray(y)  # (N,)
    return x, y


class MNISTDataset(TensorDataset):

    def __init__(self, split: str = "train") -> None:
        x, y = tfds.load("mnist", split=split, batch_size=-1, as_supervised=True)
        x, y = _process_mnist(x, y)
        super().__init__(x, y)


class MNISTCorruptedDataset(TensorDataset):

    def __init__(self, corruption: str, split: str = "train") -> None:
        if corruption not in {
            "shot_noise",
            "impulse_noise",
            "glass_blur",
        }:
            raise ValueError(f"Invalid corruption: {corruption}")
        x, y = tfds.load(
            f"mnist_corrupted/{corruption}", split=split, batch_size=-1, as_supervised=True
        )
        x, y = _process_mnist(x, y)
        super().__init__(x, y)


class FashionMNISTDataset(TensorDataset):

    def __init__(self, split: str = "train") -> None:
        x, y = tfds.load("fashion_mnist", split=split, batch_size=-1, as_supervised=True)
        x, y = _process_mnist(x, y)
        super().__init__(x, y)


class EMNISTDataset(TensorDataset):

    def __init__(self, split: str = "train") -> None:
        x, y = tfds.load("emnist/mnist", split=split, batch_size=-1, as_supervised=True)
        x, y = _process_mnist(x, y)
        super().__init__(x, y)
