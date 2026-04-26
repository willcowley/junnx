import tempfile

import jax.numpy as jnp
import jax.random as jaxr
import numpy.testing as npt
import optax
import pytest
import tensorflow_probability.substrates.jax.bijectors as tfpb
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from tensorboardX import SummaryWriter

from junnx.datasets import DataLoader
from junnx.likelihoods import GaussianLikelihood
from junnx.loss_fns import TractableFSVILoss
from junnx.metrics import MSE, RMSE
from junnx.model import TrainingModel
from junnx.net import DenseStochasticNet
from junnx.priors import Matern52Prior
from junnx.samplers import UniformSampler
from junnx.trainer import Trainer
from junnx.variational import GaussianVariationalDistribution


def model_fn(*, key: jnp.ndarray) -> TrainingModel:
    return TrainingModel(
        net=DenseStochasticNet(n_in=1, n_out=1, n_hidden=4, depth=2, key=key),
        likelihood=GaussianLikelihood(scale_init=(1.0,), bijector=tfpb.Softplus()),
        prior=Matern52Prior(lengthscales=(1.0,)),
        variational_dist=GaussianVariationalDistribution(),
    )


def test_trainer_train(dummy_dls: tuple[DataLoader, DataLoader, DataLoader]) -> None:
    key = jaxr.PRNGKey(0)
    key_init, key_train = jaxr.split(key, 2)
    model = model_fn(key=key_init)

    opt = optax.adam(1e-3)

    loss_fn = TractableFSVILoss(n_samples_nll=1)

    sampler = UniformSampler(n_dim=1, n_samples=4, low=(-1.0,), high=(1.0,))
    dl, val_dl, _ = dummy_dls

    metrics = {"MSE": MSE}

    with tempfile.TemporaryDirectory() as tmpdir:
        logger = SummaryWriter(tmpdir + "/logs")
        trainer = Trainer(
            n_epochs=2,
            loss_fn=loss_fn,
            opt=opt,
            logger=logger,
            metrics=metrics,
        )

        m = trainer.train(model, dl, sampler=sampler, val_dl=val_dl, key=key_train)

        trainer.logger.close()
        events = EventAccumulator(tmpdir + "/logs")
        events.Reload()

    # check logs are written
    loss = events.Scalars("loss/train")
    assert len(loss) == 2

    val_loss = events.Scalars("loss/val")
    assert len(val_loss) == 2

    val_mse = events.Scalars("metric/val_MSE")
    assert len(val_mse) == 2

    # check model weights updated
    def _get_w_mean(_m: TrainingModel, i: int = 0) -> jnp.ndarray:
        net = _m.net
        assert isinstance(net, DenseStochasticNet)  # for mypy
        layers = net.layers
        return layers[0].w_mean

    with pytest.raises(AssertionError):
        npt.assert_allclose(_get_w_mean(model), _get_w_mean(m))

    # check best model available
    best_model = trainer.best_model
    assert best_model is not None

    # check best opt state available
    opt_state = trainer.best_opt_state
    assert opt_state is not None


def test_trainer_eval(dummy_dls: tuple[DataLoader, DataLoader, DataLoader]) -> None:

    key = jaxr.PRNGKey(0)
    key_init, key_eval = jaxr.split(key, 2)
    model = model_fn(key=key_init)

    *_, ood_dl = dummy_dls

    metrics = {"MSE": MSE, "RMSE": RMSE}

    result = Trainer.eval(model, ood_dl, metrics, key=key_eval)

    assert len(result) == len(metrics)

    mse = result["MSE"]
    assert mse.shape == ()  # scalar

    rmse = result["RMSE"]
    assert rmse.shape == ()  # scalar
