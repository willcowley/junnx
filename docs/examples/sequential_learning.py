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
# # Sequential Learning with Stochastic Neural Networks

# %%

import copy
from typing import Optional, Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax
import tensorflow_probability.substrates.jax as tfp

from junnx.datasets import DataLoader, TensorDataset
from junnx.likelihoods import GaussianLikelihood
from junnx.net import DenseStochasticNet
from junnx.priors import Matern52Prior, SampleStochasticNetPrior
from junnx.samplers import UniformSampler
from junnx.train import TrainingModel
from junnx.trainer import Trainer
from junnx.variational import GaussianVariationalDistribution

# %%


def _f(x: jnp.ndarray) -> jnp.ndarray:
    return jnp.sqrt(x) * jnp.sin(2 * jnp.pi * x)


# %%
# Task 1
n_task = 32
key = jax.random.PRNGKey(20260215)
noise_std = 0.1

key1, key2, key3 = jax.random.split(key, 3)

x1 = jax.random.uniform(key1, (n_task, 1)) * 0.4 + 0.05
y1 = _f(x1)
key1, _ = jax.random.split(key1)
e1 = jax.random.normal(key1, y1.shape) * noise_std
y1 += e1
x1 -= 0.75

x2 = jax.random.uniform(key2, (n_task, 1)) * 0.4 + 0.05 + 1.0
y2 = _f(x2)
key2, _ = jax.random.split(key2)
e2 = jax.random.normal(key2, y2.shape) * noise_std
y2 += e2
x2 -= 0.75

x3 = jax.random.uniform(key3, (n_task, 1)) * 0.4 + 0.05 + 0.5
y3 = _f(x3)
key3, _ = jax.random.split(key3)
e3 = jax.random.normal(key3, y3.shape) * noise_std
y3 += e3
x3 -= 0.75

fig, ax = plt.subplots(1, 1, figsize=(10, 6))
ax.scatter(x1, y1, s=6, facecolor="#E15759", edgecolor="k", zorder=1, label="Task 1")
ax.scatter(x2, y2, s=6, facecolor="#4E79A7", edgecolor="k", zorder=1, label="Task 2")
ax.scatter(x3, y3, s=6, facecolor="#59A14F", edgecolor="k", zorder=1, label="Task 3")
ax.legend()
fig.savefig("/tmp/sequential_learning.png", dpi=300)

# %%

ds1 = TensorDataset(x1, y1)
ds2 = TensorDataset(x2, y2)

key_dl = jax.random.PRNGKey(42)
batch_size = 16
dl1 = DataLoader(ds1, batch_size=batch_size, shuffle=True, key=key_dl)
dl2 = DataLoader(ds2, batch_size=batch_size, shuffle=True, key=key_dl)


def _plot_model(
    model: TrainingModel,
    ds: TensorDataset,
    i: int,
    n_samples: int = 32,
    dss: Optional[Sequence[TensorDataset]] = None,
) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    ax.scatter(ds.x, ds.y, s=6, facecolor="#E15759", edgecolor="k", zorder=1)
    for _ds in dss or []:
        ax.scatter(_ds.x, _ds.y, s=6, facecolor="#B8B8B8", edgecolor="k", zorder=1)
    xx = jnp.linspace(-1.0, 1.0, 401)[:, None]

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
    fig.savefig(f"/tmp/junnx_seq_learning_{i:04d}.svg", dpi=400)
    # plt.close(fig)


key_m = jax.random.PRNGKey(0)
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
    sampler=UniformSampler(n_dim=1, n_samples=32, low=(-1.0,), high=(1.0,)),
    variational_dist=GaussianVariationalDistribution(),
)

opt = optax.adam(1e-3)

trainer1 = Trainer(
    n_samples_nll=16,
    n_samples_kl=64,
    n_epochs=2_000,
    n_data=len(ds1),
    opt=opt,
)
_ = trainer1.train(model, dl1, key=key)
m = trainer1.best_model
_plot_model(m, ds1, 1999, 1_024)

trainable, static = m.partition()

prior_net = SampleStochasticNetPrior(
    net=copy.deepcopy(m.net), variational_dist=GaussianVariationalDistribution()
)
static = eqx.tree_at(lambda _m: _m.prior, static, prior_net)
m = eqx.combine(trainable, static)

trainer2 = Trainer(
    n_samples_nll=16,
    n_samples_kl=64,
    n_epochs=4_000,
    n_data=len(ds2),
    opt=opt,
    opt_state=trainer1.best_opt_state,
)

_, key = jax.random.split(key)
_ = trainer2.train(m, dl2, key=key)
best_model = trainer2.best_model
_plot_model(best_model, ds2, 5999, 1_024, dss=(ds1,))
