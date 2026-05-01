import jax
import pytest

from junnx.datasets import DataLoader, TensorDataset


class DummyTestDataset(TensorDataset):

    def __init__(self, split: str) -> None:
        if split == "train":
            seed = 0
            n = 64
            dx = 0.0
        elif split == "val":
            seed = 1
            n = 16
            dx = 0.0
        elif split == "ood":
            seed = 2
            n = 16
            dx = 1.1
        else:
            raise ValueError(f"Invalid split: {split}")
        key_x, key_y = jax.random.split(jax.random.PRNGKey(seed), 2)
        x = jax.random.uniform(key_x, (n, 1)) + dx
        y = 2 * x + 0.5 + jax.random.normal(key_y, (n, 1)) * 0.1
        super().__init__(x, y)


@pytest.fixture(scope="session")
def dummy_datasets() -> tuple[TensorDataset, TensorDataset, TensorDataset]:
    return DummyTestDataset("train"), DummyTestDataset("val"), DummyTestDataset("ood")


@pytest.fixture(scope="session")
def dummy_dls(dummy_datasets) -> tuple[DataLoader, DataLoader, DataLoader]:
    return tuple(  # type: ignore[return-value]
        DataLoader(
            ds,
            batch_size=16,
            shuffle=True if i == 0 else False,  # only shuffle train set
            key=jax.random.PRNGKey(0),
        )
        for i, ds in enumerate(dummy_datasets)
    )
