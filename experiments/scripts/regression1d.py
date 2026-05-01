from pathlib import Path

import hydra
import jax.numpy as jnp
import jax.random as jaxr
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from omegaconf import DictConfig
from utils import instantiate_assert_type

from junnx.datasets import DataLoader, TensorDataset
from junnx.likelihoods import Likelihood
from junnx.model import TrainingModel
from junnx.net import StochasticNet
from junnx.priors import Prior
from junnx.samplers import Sampler
from junnx.trainer import Trainer
from junnx.variational import VariationalDistribution


def _plot_model(
    model: TrainingModel,
    ds: TensorDataset,
    i: int,
    outdir: Path,
    n_samples: int = 32,
    *,
    key: jnp.ndarray,
) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    ax.scatter(ds.x, ds.y, s=6, facecolor="#E15759", edgecolor="k", zorder=1)
    xx = jnp.linspace(-6.0, 6.0, 401)[:, None]

    pred_f_samples = model.net.predict_f_samples(xx, n_samples, key=key)  # [S, N, O]
    pred_f_mean = jnp.mean(pred_f_samples, axis=0)  # [N, O]
    pred_f_std = jnp.std(pred_f_samples, axis=0)  # [N, O]
    ymean, yvar = model.predict_y_mean_var(pred_f_samples)  # [N, O], [N, O]
    ystd = jnp.sqrt(yvar)  # [N, O]

    ax.plot(xx, pred_f_mean, c="k", lw=0.8)
    ax.fill_between(
        xx[:, 0],
        pred_f_mean[:, 0] + 1.95 * pred_f_std[:, 0],
        pred_f_mean[:, 0] - 1.95 * pred_f_std[:, 0],
        color="#4E79A7",
        alpha=0.5,
        lw=0.0,
    )
    ax.plot(xx, ymean, c="#59A14F", lw=0.8, ls="--")
    ax.plot(xx, ymean + 1.95 * ystd, c="#59A14F", lw=0.66, ls="--")
    ax.plot(xx, ymean - 1.95 * ystd, c="#59A14F", lw=0.66, ls="--")
    for ii in range(4):
        ax.plot(xx, pred_f_samples[ii, :, 0], c="#4E79A7", lw=0.66, alpha=0.66)
    ax.set_ylim(-2.8, 2.8)
    fig.savefig(outdir / f"junnx_regression1d_{i:04d}.svg", dpi=400)


@hydra.main(version_base="1.3", config_path="../config", config_name="regression1d")
def _main(cfg: DictConfig) -> None:
    ds = instantiate_assert_type(cfg.ds, TensorDataset)

    key_dl = instantiate_assert_type(cfg.key_dl, jnp.ndarray)

    batch_size = cfg.batch_size
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, key=key_dl)

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

    key_train = instantiate_assert_type(cfg.key_train, jnp.ndarray)
    trainer = instantiate_assert_type(cfg.trainer, Trainer)

    _ = trainer.train(model, dl, sampler, key=key_train)
    model = trainer.best_model
    key_eval = jaxr.PRNGKey(seed=20260305)
    _plot_model(
        model=model,
        ds=ds,
        i=trainer.n_epochs - 1,
        n_samples=1_024,
        outdir=Path(trainer.logger.logdir),
        key=key_eval,
    )


if __name__ == "__main__":
    load_dotenv("experiments/scripts/.env")
    _main()
