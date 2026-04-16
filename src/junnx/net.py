import abc
from typing import Sequence

import equinox as eqx
import jax
from jax import numpy as jnp

from junnx.layers import DenseStochasticLayer

_ACTIVATIONS = {
    "silu": jax.nn.silu,
    "swish": jax.nn.swish,
}


class StochasticNet(eqx.Module):

    @abc.abstractmethod
    def __call__(self, x: jnp.ndarray, key: jnp.ndarray) -> jnp.ndarray: ...

    def predict_f_samples(
        self, x: jnp.ndarray, n_samples: int, *, key: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Returns `n_samples` samples from the 'distribution over functions' represented by
        the `StochasticNet`  at input locations `x`.
        """
        # x: [N, D]
        keys = jax.random.split(key, n_samples)
        predf = jax.vmap(jax.vmap(self, in_axes=(0, None)), in_axes=(None, 0))(x, keys)
        return predf  # [S, N, O]


class DenseStochasticNet(StochasticNet):
    """
    Standard Multi-Layer Perceptron (feed-forward network) with stochastic weights.

    Args:
        n_in: The number of input features.
        n_out: The number of output features.
        n_hidden: The number of units in each hidden layer.
        depth: The number of hidden layers, including the output layer.
        use_bias: Whether to use a bias term in the layers. Defaults to True.
        activation: The non-linear activation function to use. Defaults to "silu".
        key: JAX PRNG key to provide randomness for stochastic weight initialization
            (keyword-only argument).
    """

    layers: Sequence[DenseStochasticLayer]
    """The layers of the MLP."""

    depth: int = eqx.field(static=True)
    """The number of hidden layers, including the output layer."""
    n_hidden: int = eqx.field(static=True)
    """The number of units in each hidden layer."""
    n_in: int = eqx.field(static=True)
    """The number of input features."""
    n_out: int = eqx.field(static=True)
    """The number of output features."""
    activation: str = eqx.field(static=True)
    """The non-linear activation function to use. Defaults to "silu"."""

    def __init__(
        self,
        n_in: int,
        n_out: int,
        n_hidden: int,
        depth: int,
        use_bias: bool = True,
        activation: str = "silu",
        *,
        key: jnp.ndarray,
    ) -> None:
        self.n_in = n_in
        self.n_out = n_out
        self.n_hidden = n_hidden
        self.depth = depth
        self.activation = activation

        layers = []
        for i in range(depth + 1):
            layer_key, key = jax.random.split(key, 2)
            if i == 0:
                layer = DenseStochasticLayer(n_in, n_hidden, use_bias, key=layer_key)
            elif i == depth:
                layer = DenseStochasticLayer(n_hidden, n_out, use_bias, key=layer_key)
            else:
                layer = DenseStochasticLayer(n_hidden, n_hidden, use_bias, key=layer_key)
            layers.append(layer)
        self.layers = layers

    def __call__(self, x: jnp.ndarray, key: jnp.ndarray) -> jnp.ndarray:
        for layer in self.layers[:-1]:
            layer_key, key = jax.random.split(key, 2)
            x = layer(x, layer_key)
            x = _ACTIVATIONS[self.activation](x)
        layer_key, _ = jax.random.split(key, 2)
        x = self.layers[-1](x, layer_key)
        return x


class MCDropoutMLP(StochasticNet):

    layers: Sequence[eqx.nn.Linear]
    """The layers of the MLP."""

    drop: eqx.nn.Dropout
    """The dropout layer."""
    depth: int = eqx.field(static=True)
    """The number of hidden layers, including the output layer."""
    n_hidden: int = eqx.field(static=True)
    """The number of units in each hidden layer."""
    n_in: int = eqx.field(static=True)
    """The number of input features."""
    n_out: int = eqx.field(static=True)
    """The number of output features."""
    activation: str = eqx.field(static=True)
    """The non-linear activation function to use. Defaults to "silu"."""

    def __init__(
        self,
        n_in: int,
        n_out: int,
        n_hidden: int,
        depth: int,
        use_bias: bool = True,
        activation: str = "silu",
        dropout_rate: float = 0.1,
        *,
        key: jnp.ndarray,
    ) -> None:
        self.n_in = n_in
        self.n_out = n_out
        self.n_hidden = n_hidden
        self.depth = depth
        self.activation = activation

        layers = []
        for i in range(depth + 1):
            layer_key, key = jax.random.split(key, 2)
            if i == 0:
                layer = eqx.nn.Linear(n_in, n_hidden, use_bias, key=layer_key)
            elif i == depth:
                layer = eqx.nn.Linear(n_hidden, n_out, use_bias, key=layer_key)
            else:
                layer = eqx.nn.Linear(n_hidden, n_hidden, use_bias, key=layer_key)
            layers.append(layer)
        self.layers = layers

        self.drop = eqx.nn.Dropout(p=dropout_rate)

    def __call__(self, x: jnp.ndarray, key: jnp.ndarray) -> jnp.ndarray:
        for layer in self.layers[:-1]:
            dropout_key, key = jax.random.split(key, 2)
            x = layer(x)
            x = _ACTIVATIONS[self.activation](x)
            x = self.drop(x, inference=False, key=dropout_key)
        x = self.layers[-1](x)
        return x


class StochasticLeNet(StochasticNet):
    """
    A stochastic version of the LeNet convolutional architecture.
    """

    conv1: eqx.nn.Conv2d
    pool1: eqx.nn.AvgPool2d
    conv2: eqx.nn.Conv2d
    pool2: eqx.nn.AvgPool2d
    fc1: DenseStochasticLayer
    fc2: DenseStochasticLayer
    fc3: DenseStochasticLayer

    def __init__(self, *, key: jnp.ndarray) -> None:
        conv1_key, conv2_key, fc1_key, fc2_key, fc3_key = jax.random.split(key, 5)
        self.conv1 = eqx.nn.Conv2d(1, 6, kernel_size=5, padding=2, key=conv1_key)
        self.pool1 = eqx.nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv2 = eqx.nn.Conv2d(6, 16, kernel_size=5, padding=0, key=conv2_key)
        self.pool2 = eqx.nn.AvgPool2d(kernel_size=2, stride=2)
        self.fc1 = DenseStochasticLayer(16 * 5 * 5, 120, use_bias=True, key=fc1_key)
        self.fc2 = DenseStochasticLayer(120, 84, use_bias=True, key=fc2_key)
        self.fc3 = DenseStochasticLayer(84, 10, use_bias=True, key=fc3_key)

    def _conv_backbone(self, x: jnp.ndarray) -> jnp.ndarray:
        x = self.pool1(jax.nn.silu(self.conv1(x)))
        x = self.pool2(jax.nn.silu(self.conv2(x)))
        return x.flatten()

    def _stochastic_mlp_wout_last_layer(self, x: jnp.ndarray, key: jnp.ndarray) -> jnp.ndarray:
        fc1_key, fc2_key = jax.random.split(key, 2)
        x = jax.nn.silu(self.fc1(x, fc1_key))
        x = jax.nn.silu(self.fc2(x, fc2_key))
        return x

    def _stochastic_mlp(self, x: jnp.ndarray, key: jnp.ndarray) -> jnp.ndarray:
        fc1_fc2_key, fc3_key = jax.random.split(key, 2)
        x = self._stochastic_mlp_wout_last_layer(x, fc1_fc2_key)
        x = self.fc3(x, fc3_key)
        return x

    def __call__(self, x: jnp.ndarray, key: jnp.ndarray) -> jnp.ndarray:
        x = self._conv_backbone(x)
        x = self._stochastic_mlp(x, key)
        return x

    def predict_f_samples(
        self, x: jnp.ndarray, n_samples: int, *, key: jnp.ndarray
    ) -> jnp.ndarray:
        x = jax.vmap(self._conv_backbone)(x)
        keys = jax.random.split(key, n_samples)
        predf = jax.vmap(jax.vmap(self._stochastic_mlp, in_axes=(0, None)), in_axes=(None, 0))(
            x, keys
        )
        return predf  # [S, N, O]


class MCDropoutLeNet(StochasticNet):
    """
    A LeNet architecture with Monte Carlo Dropout for uncertainty estimation.
    """

    conv1: eqx.nn.Conv2d
    pool1: eqx.nn.AvgPool2d
    conv2: eqx.nn.Conv2d
    pool2: eqx.nn.AvgPool2d
    fc1: eqx.nn.Linear
    fc2: eqx.nn.Linear
    fc3: eqx.nn.Linear
    drop: eqx.nn.Dropout

    def __init__(self, *, key: jnp.ndarray, dropout_rate: float = 0.5) -> None:
        conv1_key, conv2_key, fc1_key, fc2_key, fc3_key = jax.random.split(key, 5)
        self.conv1 = eqx.nn.Conv2d(1, 6, kernel_size=5, padding=2, key=conv1_key)
        self.pool1 = eqx.nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv2 = eqx.nn.Conv2d(6, 16, kernel_size=5, padding=0, key=conv2_key)
        self.pool2 = eqx.nn.AvgPool2d(kernel_size=2, stride=2)
        self.fc1 = eqx.nn.Linear(16 * 5 * 5, 120, key=fc1_key)
        self.fc2 = eqx.nn.Linear(120, 84, key=fc2_key)
        self.fc3 = eqx.nn.Linear(84, 10, key=fc3_key)
        self.drop = eqx.nn.Dropout(p=dropout_rate)

    def _conv_backbone(self, x: jnp.ndarray) -> jnp.ndarray:
        x = self.pool1(jax.nn.silu(self.conv1(x)))
        x = self.pool2(jax.nn.silu(self.conv2(x)))
        return x.flatten()

    def _dropout_mlp(self, x: jnp.ndarray, key: jnp.ndarray) -> jnp.ndarray:
        x = jax.nn.silu(self.fc1(x))
        x = jax.nn.silu(self.fc2(x))
        x = self.drop(x, inference=False, key=key)
        x = self.fc3(x)
        return x

    def __call__(self, x: jnp.ndarray, key: jnp.ndarray) -> jnp.ndarray:
        x = self._conv_backbone(x)
        x = self._dropout_mlp(x, key)
        return x

    def predict_f_samples(
        self, x: jnp.ndarray, n_samples: int, *, key: jnp.ndarray
    ) -> jnp.ndarray:
        x = jax.vmap(self._conv_backbone)(x)
        keys = jax.random.split(key, n_samples)
        predf = jax.vmap(jax.vmap(self._dropout_mlp, in_axes=(0, None)), in_axes=(None, 0))(
            x, keys
        )
        return predf  # [S, N, O]
