import jax.numpy as jnp
import tensorflow_datasets as tfds

from junnx.datasets import TensorDataset


class MNISTDataset(TensorDataset):

    def __init__(self):
        x, y = tfds.load("mnist", split="train", batch_size=-1, as_supervised=True)
        x = jnp.asarray(x) / 255.0  # (N, 28, 28, 1)
        x = x.transpose(0, 3, 1, 2)  # (N, 1, 28, 28)  # eqx expects channel first format
        y = jnp.asarray(y)[:, None]  # (N, 1)
        super().__init__(x, y)
