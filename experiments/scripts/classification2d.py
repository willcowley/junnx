from pathlib import Path

import hydra
import jax
import jax.numpy as jnp
import jax.random as jaxr
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from mpl_toolkits.axes_grid1 import make_axes_locatable
from omegaconf import DictConfig
from utils import instantiate_assert_type

from junnx.datasets import DataLoader, TensorDataset
from junnx.likelihoods import CategoricalLikelihood, Likelihood
from junnx.net import StochasticNet
from junnx.priors import Prior
from junnx.samplers import Sampler
from junnx.train import TrainingModel
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
    n_grid = 201
    x_grid = jnp.linspace(-4, 4, n_grid)  # [N,]
    x0, x1 = jnp.meshgrid(x_grid, x_grid)
    x = jnp.concat([x1.reshape(-1, 1), x0.reshape(-1, 1)], axis=-1)  # [N*N, 2]

    keys = jaxr.split(key, n_samples)
    predf = jax.lax.map(
        lambda _k: model.net.predict_f_samples(x, 1, key=_k), keys, batch_size=64
    )  # [S, 1, N*N, O]
    predf = predf[..., 0, :, :]  # [S, N*N, O]
    likelihood = model.likelihood
    assert isinstance(likelihood, CategoricalLikelihood)
    predp = likelihood.probs(predf)[..., :1]  # [S, N*N, 1],  P(class 0)

    p_mean = predp.mean(axis=0).reshape(n_grid, n_grid).T
    p_var = predp.var(axis=0).reshape(n_grid, n_grid).T

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    imshow_kwargs = {"extent": [-4, 4, -4, 4], "origin": "lower", "vmin": 0.0}
    append_ax_kwargs = {"position": "right", "size": "5%", "pad": 0.05}
    labels = [r"$\mathbb{E}\,[p(y|\mathcal{D})]$", r"$\mathbb{Var}\,[p(y|\mathcal{D})]$"]
    for ax, z, vmax, label in zip(axes, [p_mean, p_var], [1.0, 0.2], labels):
        im = ax.imshow(z, vmax=vmax, **imshow_kwargs)
        cax = make_axes_locatable(ax).append_axes(**append_ax_kwargs)
        fig.colorbar(im, cax=cax, orientation="vertical", label=label)
        ax.scatter(*ds.x.T, c=ds.y, s=6, edgecolor="w", linewidths=0.5)
        ax.set_yticks(ax.get_xticks())
    fig.subplots_adjust(left=0.05, top=0.8, bottom=0.2, right=0.9)
    fig.savefig(outdir / f"junnx_classification2d_{i:04d}.svg", dpi=400)


@hydra.main(version_base="1.3", config_path="../config", config_name="classification2d")
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
