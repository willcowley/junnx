import jax.numpy as jnp


def sample_mean_cov(predf: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    # predf: [S, N, O]
    predf = jnp.transpose(predf, (2, 0, 1))  # [O, S, N]

    mean = jnp.mean(predf, axis=-2)  # [O, N]
    diffs = predf - mean[..., None, :]  # [O, S, N]
    cov = jnp.einsum("osm,osn->omn", diffs, diffs) / (predf.shape[-2] - 1)  # [O, N, N]
    cov = 0.5 * (cov + jnp.swapaxes(cov, -1, -2))  # enforce symmetry
    return mean, cov  # [O, N], [O, N, N]
