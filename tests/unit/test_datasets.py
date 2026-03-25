from typing import Set

import jax.numpy as jnp
import jax.random as jaxr
import numpy.testing as npt
import pytest

from junnx.datasets import DataLoader, TensorDataset


def _get_dummy_data(n_data: int) -> tuple[jnp.ndarray, jnp.ndarray]:
    key = jaxr.PRNGKey(0)
    key_x, key_y = jaxr.split(key, 2)
    dummy_x = jaxr.normal(key_x, shape=(n_data, 3))
    dummy_y = jaxr.normal(key_y, shape=(n_data, 1))
    return dummy_x, dummy_y


def test_tensor_dataset() -> None:

    dummy_x, dummy_y = _get_dummy_data(32)
    ds = TensorDataset(dummy_x, dummy_y)

    npt.assert_allclose(ds.x, dummy_x)
    npt.assert_allclose(ds.y, dummy_y)

    assert len(ds) == 32


def test_tensor_dataset_raises() -> None:

    dummy_x, dummy_y = _get_dummy_data(16)

    with pytest.raises(ValueError):
        TensorDataset(dummy_x, dummy_y[..., :-1, :])


def test_data_loader() -> None:
    n_data = 32
    dummy_x, dummy_y = _get_dummy_data(n_data)
    ds = TensorDataset(dummy_x, dummy_y)

    batch_size = 4
    key_dl = jaxr.PRNGKey(20260301)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, key=key_dl)

    n_batches = 0
    for i, (x_batch, y_batch) in enumerate(dl):
        npt.assert_allclose(x_batch, dummy_x[i * batch_size : (i + 1) * batch_size])
        npt.assert_allclose(y_batch, dummy_y[i * batch_size : (i + 1) * batch_size])
        n_batches += 1

    assert n_batches == n_data // batch_size  # should have seen all batches


def test_data_loader_shuffle() -> None:
    key = jaxr.PRNGKey(0)
    key_x, key_y, key_dl = jaxr.split(key, 3)
    n_data = 32
    dummy_x = jnp.arange(n_data * 2).reshape(n_data, 2)
    dummy_y = jnp.arange(n_data).reshape(n_data, 1)
    ds = TensorDataset(dummy_x, dummy_y)

    batch_size = 8
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, key=key_dl)

    seen_idxs: Set[int] = set()
    batch_idxss = []
    for i, (x_batch, y_batch) in enumerate(dl):
        batch_idxs = set(y_batch[..., 0].tolist())
        batch_idxss.append(batch_idxs)
        assert len(batch_idxs) == batch_size
        assert not batch_idxs.intersection(seen_idxs)  # no idxs should be repeated
        seen_idxs.update(batch_idxs)
        npt.assert_allclose(x_batch[:, 0], y_batch[:, 0] * 2)  # x and y should correspond
        npt.assert_allclose(x_batch[:, 1] - 1, y_batch[:, 0] * 2)

    assert len(seen_idxs) == n_data  # all idxs seen over epoch

    for i, (x_batch, y_batch) in enumerate(dl):
        batch_idxs = set(y_batch[..., 0].tolist())
        assert batch_idxs != batch_idxss[i]  # different batches each epoch
