import equinox as eqx
import jax
import jax.numpy as jnp

_TRUNC_2STD_NORM = 0.87962566103423978


class DenseStochasticLayer(eqx.Module):
    """Just your regular densely-connected layer, but with stochastic weights."""

    w_mean: jnp.ndarray
    """The mean of the weights."""
    w_log_var: jnp.ndarray
    """The natural log of the variance of the weights."""
    bias: jnp.ndarray | None
    """The bias term, if `use_bias` is True, else None."""

    n_in: int = eqx.field(static=True)
    """The number of input features."""
    n_out: int = eqx.field(static=True)
    """The number of output features."""
    use_bias: bool = eqx.field(static=True)
    """Whether to use a bias term. Defaults to True."""

    def __init__(
        self, n_in: int, n_out: int, use_bias: bool = True, *, key: jnp.ndarray
    ) -> None:
        wm_key, b_key = jax.random.split(key, 2)

        self.n_in = n_in
        self.n_out = n_out
        self.use_bias = use_bias

        # Glorot normal initialization
        std_init = jnp.sqrt(2.0 / (n_in + n_out))
        w_shape = (n_out, n_in)
        scale = std_init / jnp.sqrt(2)
        # use truncated normal to avoid large weight values that can destabilize training
        self.w_mean = (
            scale * jax.random.truncated_normal(wm_key, -2.0, 2.0, w_shape) / _TRUNC_2STD_NORM
        )
        self.w_log_var = jnp.full(w_shape, 2 * jnp.log(scale))

        self.bias = jnp.zeros((n_out,)) if use_bias else None

    @property
    def w_std(self) -> jnp.ndarray:
        """The standard deviation of the weights."""
        return jnp.exp(0.5 * self.w_log_var)

    @property
    def w_var(self) -> jnp.ndarray:
        """The variance of the weights."""
        return jnp.exp(self.w_log_var)

    def __call__(self, x: jnp.ndarray, key: jnp.ndarray) -> jnp.ndarray:
        eta = (
            jax.random.truncated_normal(key, -2.0, 2.0, self.w_log_var.shape)
            / _TRUNC_2STD_NORM
        )
        weights = self.w_mean + eta * self.w_std

        x = weights @ x
        if self.use_bias:
            x = x + self.bias
        return x
