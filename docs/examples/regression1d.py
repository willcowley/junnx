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

# JUNNX is a simple, lightweight and modular repository for exploring approximate Bayesian Inference with neural
# networks. In this example, we will show you how to create and train a JUNNX model. In doing so we will
# replicate Figure 1a of [Rudner et al.](https://arxiv.org/abs/2312.17199v1).

# %%
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax
import tensorflow_probability.substrates.jax as tfp

from junnx.data import SnelsonDataset
from junnx.datasets import DataLoader
from junnx.likelihoods import GaussianLikelihood
from junnx.loss_fns import SampleFSVILoss
from junnx.model import TrainingModel
from junnx.net import DenseStochasticNet
from junnx.priors import Matern52Prior
from junnx.samplers import UniformSampler
from junnx.trainer import Trainer
from junnx.variational import GaussianVariationalDistribution

# %% [markdown]

# First we can load and inspect the training dataset. We load the dataset presented in Figure 1. of
# [Snelson & Ghahramani](https://papers.nips.cc/paper_files/paper/2005/hash/4491777b1aa8b5b32c2e8666dbe1a495-Abstract.html),
# however, we remove data between $-1<x<0$ to investigate how the model behaves in interpolation regions (i.e. between
# training data points) vs. out-of-distribution (OOD), away from any training data.

# %%
# load data
ds = SnelsonDataset()

fig, ax = plt.subplots(1, 1, figsize=(10, 6))
ax.scatter(ds.x, ds.y, s=6, facecolor="#E15759", edgecolor="k", zorder=1)
ax.set_ylim(-2.8, 2.8)


# %% [markdown]

# Now we create our `TrainingModel`, this is a JUNNX container for all the objects necessary for training JUNNX model.
# The components are:
# - `net`: A neural network that exhibits some form of stochasticity and thus represents samples from some distribution
# over functions.
# - `likelihood`: The object that determines the likelihood of observing some data given the values of the `net` latent
# functions.
# - `prior`: Represents our prior beliefs about the distribution over functions represented by the `net`.
# - `variational_dist`: The object that computes an approximate analytical distribution to the one represented by `net`
# The `net` and `likelihood` terms are used to compute the "data" i.e. negative log likelihood (NLL) term in our loss
# function(s), whereas the `prior` and `variational_dist` are used to compute the KL divergence regularisation term.

# %%

key = jax.random.PRNGKey(42)
key_m, key_dl = jax.random.split(key, 2)
model = TrainingModel(
    # small MLP architecture
    net=DenseStochasticNet(
        n_in=1,
        n_out=1,
        n_hidden=128,
        depth=3,
        use_bias=True,
        key=key_m,
    ),
    # Gaussian noise seems a reasonable modelling assumption
    likelihood=GaussianLikelihood(
        scale_init=(1.0,),
        bijector=tfp.bijectors.Softplus(),  # must map R->R+
    ),
    # A Matern52Prior with this lengthscale seems reasonable
    prior=Matern52Prior(lengthscales=(0.4,)),
    # Our prior returns a Gaussian distribution, so a Gaussian variational distribution allows for an analytic KL
    # divergence calculation
    variational_dist=GaussianVariationalDistribution(),
)

# %%
# The DataLoader provides some convenience for producing batches of our data
dl = DataLoader(ds, batch_size=32, shuffle=True, key=key)

# %% [markdown]
# The sampler provides context points at which the KL term in the loss is evaluted. For some problems where the data
# represents a low-dimensional manifold of a high-dimensional space (e.g. images) it is sometimes beneficial to sample
# from the training data, or some perturbed version of the training data. However, for regression problems such as this
# it's possible to simply randomly sample from the space - covering the entire range over which we require predictions
# to be made.  This is what the UniformSampler below does.

# %%
sampler = UniformSampler(n_dim=1, n_samples=32, low=(-6.0,), high=(6.0,))

# %% [markdown]

# The `SampleFSVILoss` compute the ELBO using MC samples for approximating both data (NLL) and KL terms.

# %%
loss_fn = SampleFSVILoss(n_samples_nll=16, n_samples_kl=64)

# %%


def _plot_model(model: TrainingModel, i: int, n_samples: int = 32) -> None:
    # helper for plotting models
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    ax.scatter(ds.x, ds.y, s=6, facecolor="#E15759", edgecolor="k", zorder=1)
    xx = jnp.linspace(-6.0, 6.0, 601)[:, None]

    pred_f_samples = model.net.predict_f_samples(xx, n_samples, key=key)  # [S, N, O]
    pred_f_mean = jnp.mean(pred_f_samples, axis=0)  # [N, O]
    pred_f_std = jnp.std(pred_f_samples, axis=0)  # [N, O]
    ymean, yvar = model.predict_y_mean_var(pred_f_samples)  # [N, O], [N, O]
    ystd = jnp.sqrt(yvar)  # [N, O]

    ax.plot(xx, pred_f_mean, c="k", lw=0.8)
    ax.fill_between(
        xx[:, 0],
        pred_f_mean[:, 0] + 1.95 * pred_f_std[:, 0],  # epistemic uncertainty
        pred_f_mean[:, 0] - 1.95 * pred_f_std[:, 0],
        color="#4E79A7",
        alpha=0.5,
        lw=0.0,
    )
    ax.plot(xx, ymean, c="#59A14F", lw=0.8, ls="--")
    ax.plot(
        xx, ymean + 1.95 * ystd, c="#59A14F", lw=0.66, ls="--"
    )  # epistemic + aleatoric uncertainty
    ax.plot(xx, ymean - 1.95 * ystd, c="#59A14F", lw=0.66, ls="--")
    for ii in range(4):
        ax.plot(xx, pred_f_samples[ii, :, 0], c="#4E79A7", lw=0.66, alpha=0.66)
    ax.set_ylim(-2.8, 2.8)


# %% [markdown]

# Finally, we put all of these elements together in our `Trainer` and inspect the final model.

# %%
opt = optax.adam(1e-3)
trainer = Trainer(
    loss_fn=loss_fn,
    n_epochs=4_000,
    opt=opt,
)
_ = trainer.train(model, dl, sampler, key=key)

# %%
m = trainer.best_model
_plot_model(m, 3999, 1_024)
