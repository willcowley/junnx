import hydra
import jax.numpy as jnp
import jax.random as jaxr
from dotenv import load_dotenv
from omegaconf import DictConfig
from utils import instantiate_assert_type

from junnx.datasets import DataLoader, TensorDataset
from junnx.likelihoods import Likelihood
from junnx.metrics import EntropyAUROC
from junnx.net import StochasticNet
from junnx.priors import Prior
from junnx.samplers import Sampler
from junnx.train import TrainingModel
from junnx.trainer import Trainer
from junnx.variational import VariationalDistribution


@hydra.main(version_base="1.3", config_path="../config", config_name="mnist_ood")
def _main(cfg: DictConfig) -> None:

    ds = instantiate_assert_type(cfg.ds, TensorDataset)
    val_ds = instantiate_assert_type(cfg.val_ds, TensorDataset)
    # context_ds = instantiate_assert_type(cfg.context_ds, TensorDataset)
    ood_ds = instantiate_assert_type(cfg.ood_ds, TensorDataset)

    detect_ood_ds = TensorDataset(
        x=jnp.concatenate([ood_ds.x, val_ds.x], axis=0),
        y=jnp.concatenate([jnp.ones_like(ood_ds.y), jnp.zeros_like(val_ds.y)], axis=0),
    )

    key_dl = instantiate_assert_type(cfg.key_dl, jnp.ndarray)

    batch_size = cfg.batch_size
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, key=key_dl)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, key=key_dl)
    ood_dl = DataLoader(detect_ood_ds, batch_size=batch_size, shuffle=False, key=key_dl)

    net = instantiate_assert_type(cfg.net, StochasticNet)
    likelihood = instantiate_assert_type(cfg.likelihood, Likelihood)
    prior = instantiate_assert_type(cfg.prior, Prior)
    sampler = instantiate_assert_type(cfg.sampler, Sampler)
    variational_dist = instantiate_assert_type(cfg.variational, VariationalDistribution)

    model = TrainingModel(
        net=net,
        likelihood=likelihood,
        prior=prior,
        variational_dist=variational_dist,
    )

    trainer = instantiate_assert_type(cfg.trainer, Trainer)

    key_train = instantiate_assert_type(cfg.key_train, jnp.ndarray)
    _ = trainer.train(model, dl, sampler, val_dl, key=key_train)
    model = trainer.best_model

    ood_metrics = Trainer.eval(
        model,
        ood_dl,
        {"EntropyAUROC": EntropyAUROC},
        n_samples=128,
        key=jaxr.PRNGKey(20260409),
    )

    logger = trainer.logger
    assert logger is not None
    for k, v in ood_metrics.items():
        logger.add_scalar(f"ood/{k}", v, trainer.n_epochs - 1)


if __name__ == "__main__":
    load_dotenv("experiments/scripts/.env")
    _main()
