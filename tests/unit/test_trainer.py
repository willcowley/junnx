import optax
import pytest

from junnx.loss_fns import SampleFSVILoss
from junnx.trainer import Trainer


def test_trainer_raises_best_model() -> None:

    loss_fn = SampleFSVILoss(4, 4)
    opt = optax.adam(1e-3)
    trainer = Trainer(10, opt, loss_fn)

    with pytest.raises(ValueError):
        trainer.best_model


def test_trainer_raises_best_opt_state() -> None:
    loss_fn = SampleFSVILoss(4, 4)
    opt = optax.adam(1e-3)
    trainer = Trainer(10, opt, loss_fn)

    with pytest.raises(ValueError):
        trainer.best_opt_state
