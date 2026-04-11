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
# # 1D Regression with Stochastic Neural Networks


# %%
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax
import tensorflow_probability.substrates.jax as tfp

from junnx.data import SnelsonDataset
from junnx.datasets import DataLoader
from junnx.likelihoods import GaussianLikelihood
from junnx.net import DenseStochasticNet
from junnx.priors import Matern52Prior
from junnx.samplers import UniformSampler
from junnx.train import TrainingModel
from junnx.trainer import Trainer
from junnx.variational import GaussianVariationalDistribution

# %%
# load data
ds = SnelsonDataset()

fig, ax = plt.subplots(1, 1, figsize=(10, 6))
ax.scatter(ds.x, ds.y, s=6, facecolor="#E15759", edgecolor="k", zorder=1)
ax.set_ylim(-2.8, 2.8)


# %%

# create model

key = jax.random.PRNGKey(42)
key_m, key_dl = jax.random.split(key, 2)
model = TrainingModel(
    net=DenseStochasticNet(
        n_in=1,
        n_out=1,
        n_hidden=32,
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

dl = DataLoader(ds, batch_size=32, shuffle=True, key=key)

opt = optax.adam(1e-3)


def _plot_model(model: TrainingModel, i: int, n_samples: int = 32) -> None:
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
    fig.savefig(f"/tmp/junnx_regression1d_{i:04d}.svg", dpi=400)
    # plt.close(fig)


trainer = Trainer(
    n_samples_nll=16,
    n_samples_kl=64,
    n_epochs=4_000,
    opt=opt,
)
_ = trainer.train(model, dl, key=key)
m = trainer.best_model
_plot_model(m, 3999, 1_024)
