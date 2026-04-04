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

from datetime import datetime

import jax
import jax.numpy as jnp
import optax
from tensorboardX import SummaryWriter

from junnx.data import MNISTDataset
from junnx.data.mnist import MNISTCorruptedDataset
from junnx.datasets import DataLoader
from junnx.likelihoods import CategoricalLikelihood
from junnx.metrics import ECE, Accuracy, Brier
from junnx.net import StochasticLeNet
from junnx.priors import DirichletPrior
from junnx.samplers import DataSampler
from junnx.train import TrainingModel
from junnx.trainer import Trainer
from junnx.variational import DirichletVariationalDistribution

ds = MNISTDataset(split="train")
val_ds = MNISTDataset(split="test")
context_ds = MNISTCorruptedDataset(split="test", corruption="impulse_noise")
ood_ds = MNISTCorruptedDataset(split="test", corruption="glass_blur")

key = jax.random.PRNGKey(42)
key_dl, key_m = jax.random.split(key, 2)
dl = DataLoader(ds, batch_size=32, shuffle=True, key=key_dl)
val_dl = DataLoader(val_ds, batch_size=32, shuffle=False, key=key_dl)
ood_dl = DataLoader(ood_ds, batch_size=32, shuffle=False, key=key_dl)

opt = optax.adam(1e-3)

model = TrainingModel(
    net=StochasticLeNet(key=key_m),
    likelihood=CategoricalLikelihood(),
    prior=DirichletPrior(concentration=jnp.asarray([0.5] * 10)),
    sampler=DataSampler(data=context_ds, n_samples=32),
    variational_dist=DirichletVariationalDistribution(),
)

metrics = {"ACC": Accuracy, "ECE": ECE, "Brier": Brier}

log_dir = f"/tmp/fsvi_mnist_example/{datetime.now().strftime('%Y%m%d-%H%M%S')}"
logger = SummaryWriter(log_dir=log_dir)

trainer = Trainer(
    n_samples_nll=4,
    n_samples_kl=8,
    n_epochs=20,
    n_data=len(ds),
    opt=opt,
    metrics=metrics,  # type: ignore[arg-type]
    logger=logger,
)
_ = trainer.train(model, dl, val_dl, ood_dl, key=key)
m = trainer.best_model
