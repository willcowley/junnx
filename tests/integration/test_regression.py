import jax
import optax
import tensorflow_probability.substrates.jax as tfp

from junnx.datasets import DataLoader, TensorDataset
from junnx.likelihoods import GaussianLikelihood
from junnx.metrics import MSE, NLL
from junnx.net import DenseStochasticNet
from junnx.priors import Matern52Prior
from junnx.samplers import UniformSampler
from junnx.train import TrainingModel
from junnx.trainer import Trainer
from junnx.variational import GaussianVariationalDistribution


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


def test_regression() -> None:

    key = jax.random.PRNGKey(42)
    key_m, key_dl, key_train = jax.random.split(key, 3)

    ds = DummyTestDataset(split="train")
    val_ds = DummyTestDataset(split="val")
    ood_ds = DummyTestDataset(split="ood")
    key_dl, key_val, key_ood = jax.random.split(key_dl, 3)
    dl = DataLoader(ds, batch_size=32, shuffle=True, key=key_dl)
    val_dl = DataLoader(val_ds, batch_size=8, shuffle=False, key=key_val)
    ood_dl = DataLoader(ood_ds, batch_size=8, shuffle=False, key=key_ood)

    # create model
    model = TrainingModel(
        net=DenseStochasticNet(
            n_in=1,
            n_out=1,
            n_hidden=8,
            depth=2,
            use_bias=True,
            key=key_m,
        ),
        likelihood=GaussianLikelihood(
            scale_init=(1.0,),
            bijector=tfp.bijectors.Softplus(),
        ),
        prior=Matern52Prior(lengthscales=(0.4,)),
        sampler=UniformSampler(n_dim=1, n_samples=32, low=(-6.0,), high=(6.0,)),
        variational_dist=GaussianVariationalDistribution(),
    )

    opt = optax.adam(1e-3)

    metrics = {"MSE": MSE, "NLL": NLL}

    trainer = Trainer(
        n_samples_nll=4,
        n_samples_kl=16,
        n_epochs=10,
        opt=opt,
        metrics=metrics,
    )

    _ = trainer.train(model, dl, val_dl, key=key_train)
    _ = trainer.best_model

    _ = Trainer.eval(
        model=trainer.best_model,
        dl=ood_dl,
        metrics=metrics,
        key=key_train,
    )
