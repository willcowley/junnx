import jax
import optax
import tensorflow_probability.substrates.jax as tfp

from junnx.datasets import DataLoader, TensorDataset
from junnx.likelihoods import GaussianLikelihood
from junnx.net import DenseStochasticNet
from junnx.priors import Matern52Prior
from junnx.samplers import UniformSampler
from junnx.train import TrainingModel
from junnx.trainer import Trainer
from junnx.variational import GaussianVariationalDistribution


class TestDataset(TensorDataset):

    def __init__(self) -> None:
        key_x, key_y = jax.random.split(jax.random.PRNGKey(0), 2)
        x = jax.random.uniform(key_x, (64, 1)) * 2 - 1
        y = 2 * x + 0.5 + jax.random.normal(key_y, (64, 1)) * 0.1
        super().__init__(x, y)


def test_regression() -> None:

    key = jax.random.PRNGKey(42)
    key_m, key_dl, key_train = jax.random.split(key, 3)

    ds = TestDataset()
    dl = DataLoader(ds, batch_size=32, shuffle=True, key=key_dl)

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

    trainer = Trainer(
        n_samples_nll=4,
        n_samples_kl=16,
        n_epochs=10,
        n_data=len(ds),
        opt=opt,
    )

    _ = trainer.train(model, dl, key=key_train)
    _ = trainer.best_model
