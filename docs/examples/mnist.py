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
# # MNIST Example with Stochastic CNN

# %%

import jax
import jax.numpy as jnp
import optax

from junnx.data import MNISTDataset
from junnx.datasets import DataLoader
from junnx.likelihoods import CategoricalLikelihood
from junnx.net import StochasticLeNet
from junnx.priors import DirichletPrior
from junnx.samplers import DataSampler
from junnx.trainer import Trainer, TrainingModel
from junnx.variational import DirichletVariationalDistribution

ds = MNISTDataset()

key = jax.random.PRNGKey(42)
key_dl, key_m = jax.random.split(key, 2)
dl = DataLoader(ds, batch_size=32, shuffle=True, key=key_dl)

opt = optax.adam(1e-3)

model = TrainingModel(
    net=StochasticLeNet(key=key_m),
    likelihood=CategoricalLikelihood(),
    prior=DirichletPrior(concentration=jnp.asarray([0.5] * 10)),
    sampler=DataSampler(data=ds, n_samples=32),
    variational_dist=DirichletVariationalDistribution(),
)

trainer = Trainer(
    n_samples_nll=4,
    n_samples_kl=16,
    n_epochs=10,
    n_data=len(ds),
    opt=opt,
)
_ = trainer.train(model, dl, key=key)
m = trainer.best_model
