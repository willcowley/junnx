# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 2D Classification with Stochastic Neural Networks

# %%

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax
from mpl_toolkits.axes_grid1 import make_axes_locatable

from junnx.data import MakeMoonsDataset
from junnx.datasets import DataLoader
from junnx.likelihoods import CategoricalLikelihood
from junnx.net import DenseStochasticNet
from junnx.priors import DirichletPrior
from junnx.samplers import UniformSampler
from junnx.train import TrainingModel
from junnx.trainer import Trainer
from junnx.variational import DirichletVariationalDistribution

# %%
ds = MakeMoonsDataset(n_samples=256, noise=0.2, seed=42)

fig, ax = plt.subplots(1, 1, figsize=(4, 4))
ax.scatter(ds.x[:, 0], ds.x[:, 1], c=ds.y[:, 0], s=6, edgecolor="k", zorder=1)

# fig.savefig("/tmp/classification2d.png", dpi=300)


def _plot_model(model: TrainingModel, i: int, n_samples: int = 32) -> None:
    n_grid = 201
    x_grid = jnp.linspace(-4, 4, n_grid)  # [N,]
    x0, x1 = jnp.meshgrid(x_grid, x_grid)
    x = jnp.concat([x1.reshape(-1, 1), x0.reshape(-1, 1)], axis=-1)  # [N*N, 2]

    keys = jax.random.split(jax.random.PRNGKey(0), n_samples)
    predf = jax.lax.map(
        lambda _k: model.net.predict_f_samples(x, 1, key=_k), keys, batch_size=64
    )  # [S, 1, N*N, O]
    predf = predf[..., 0, :, :]  # [S, N*N, O]
    # predf = model.predict_f_samples(x, n_samples, key=jax.random.PRNGKey(0))  # [S, N*N, O]
    likelihood = model.likelihood
    assert isinstance(likelihood, CategoricalLikelihood)
    predp = likelihood.probs(predf)[..., :1]  # [S, N*N, 1],  P(class 0)

    p_mean = predp.mean(axis=0).reshape(n_grid, n_grid).T
    p_var = predp.var(axis=0).reshape(n_grid, n_grid).T

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    imshow_kwargs = {"extent": [-4, 4, -4, 4], "origin": "lower", "vmin": 0.0}
    append_ax_kwargs = {"position": "right", "size": "5%", "pad": 0.05}
    labels = [r"$\mathbb{E}\,[p(y|\mathcal{D})]$", r"$\mathbb{Var}\,[p(y|\mathcal{D})]$"]
    for ax, z, vmax, label in zip(axes, [p_mean, p_var], [1.0, 0.125], labels):
        im = ax.imshow(z, vmax=vmax, **imshow_kwargs)
        cax = make_axes_locatable(ax).append_axes(**append_ax_kwargs)
        fig.colorbar(im, cax=cax, orientation="vertical", label=label)
        ax.scatter(*ds.x.T, c=ds.y, s=6, edgecolor="w", linewidths=0.5)
        ax.set_yticks(ax.get_xticks())
    fig.subplots_adjust(left=0.05, top=0.8, bottom=0.2, right=0.9)
    fig.savefig(f"/tmp/junnx_classification2d_{i:04d}.svg", dpi=400)
    # plt.close(fig)


# %%

# create model
key = jax.random.PRNGKey(42)
key_m, key_dl = jax.random.split(key, 2)
model = TrainingModel(
    net=DenseStochasticNet(
        n_in=2,
        n_out=2,
        n_hidden=32,
        depth=2,
        use_bias=True,
        key=key_m,
    ),
    likelihood=CategoricalLikelihood(),
    prior=DirichletPrior(concentration=jnp.array([0.5, 0.5])),  # Jeffrey's prior
    sampler=UniformSampler(n_dim=2, n_samples=32, low=(-4.0, -4.0), high=(4.0, 4.0)),
    variational_dist=DirichletVariationalDistribution(),
)

dl = DataLoader(ds, batch_size=32, shuffle=True, key=key)

opt = optax.adam(1e-3)
trainer = Trainer(
    n_samples_nll=16,
    n_samples_kl=64,
    n_epochs=2_000,
    n_data=len(ds),
    opt=opt,
)
_ = trainer.train(model, dl, key=key)
m = trainer.best_model
_plot_model(m, 1999, 1_024)
