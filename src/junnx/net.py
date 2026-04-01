import abc
from typing import Callable, Mapping, Sequence

import equinox as eqx
import jax
from jax import numpy as jnp

from junnx.layers import DenseStochasticLayer

_ACTIVATIONS: Mapping[str, Callable[[jnp.ndarray], jnp.ndarray]] = {
    "silu": jax.nn.silu,
    "swish": jax.nn.swish,
    "relu": jax.nn.relu,
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
        predf = jax.vmap(lambda k: jax.vmap(lambda _x: self(_x, key=k))(x))(keys)
        return predf  # [S, N, O]

    def tractable_f_mean_cov(
        self, x: jnp.ndarray, key: jnp.ndarray
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """
        Returns the tractable approximation (see Rudner et al. 2024, Section 3.1) of mean and
        covariance of the 'distribution over functions' represented by the `StochasticNet` at
        input locations `x`.
        """
        raise NotImplementedError


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

    def _call_wout_last_layer(self, x: jnp.ndarray, key: jnp.ndarray) -> jnp.ndarray:
        for layer in self.layers[:-1]:
            layer_key, key = jax.random.split(key, 2)
            x = layer(x, layer_key)
            x = _ACTIVATIONS[self.activation](x)
        return x

    def tractable_f_mean_cov(
        self, x: jnp.ndarray, key: jnp.ndarray
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        # x: [N, D]
        x = jax.vmap(self._call_wout_last_layer, in_axes=(0, None))(x, key)  # [N, H]

        w_mean = self.layers[-1].w_mean  # [O, H]
        w_var = self.layers[-1].w_var  # [O, H]
        b = self.layers[-1].bias  # [O,] or None

        mean = w_mean @ x.T  # [O, N]
        if b is not None:
            mean = mean + b[:, None]  # [O, N]
        cov = jnp.einsum("nh,oh,mh->onm", x, w_var, x)  # [O, N, N]
        return mean, cov
