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

from junnx.data import EMNISTDataset, FashionMNISTDataset, MNISTDataset
from junnx.datasets import DataLoader, TensorDataset
from junnx.likelihoods import CategoricalLikelihood
from junnx.loss_fns import NLLLoss, SampleFSVILoss
from junnx.metrics import ECE, Accuracy, Brier, EntropyAUROC
from junnx.net import MCDropoutLeNet, StochasticLeNet
from junnx.priors import DirichletPrior
from junnx.samplers import DataSampler
from junnx.model import TrainingModel
from junnx.trainer import Trainer
from junnx.variational import DirichletVariationalDistribution

ds = MNISTDataset(split="train")
val_ds = MNISTDataset(split="test")
context_ds = EMNISTDataset(split="test")
fashion_mnist_ds = FashionMNISTDataset(split="test")

ood_detection_ds = TensorDataset(
    x=jnp.concatenate([fashion_mnist_ds.x, val_ds.x], axis=0),
    y=jnp.concatenate([jnp.ones_like(fashion_mnist_ds.y), jnp.zeros_like(val_ds.y)], axis=0),
)

key = jax.random.PRNGKey(42)
key_dl, key_m = jax.random.split(key, 2)
dl = DataLoader(ds, batch_size=128, shuffle=True, key=key_dl)
val_dl = DataLoader(val_ds, batch_size=128, shuffle=False, key=key_dl)
ood_dl = DataLoader(ood_detection_ds, batch_size=128, shuffle=False, key=key_dl)

schedule = optax.cosine_decay_schedule(
    init_value=2e-3, decay_steps=dl.batches_per_epoch * 30, alpha=0.05
)
opt = optax.sgd(schedule, momentum=0.9)
sampler = DataSampler(data=context_ds, n_samples=128)

mcdropout = False

model = TrainingModel(
    net=StochasticLeNet(key=key_m) if not mcdropout else MCDropoutLeNet(key=key_m),
    likelihood=CategoricalLikelihood(),
    prior=DirichletPrior(concentration=jnp.asarray([0.5] * 10)),
    variational_dist=DirichletVariationalDistribution(),
)

metrics = {"ACC": Accuracy, "ECE": ECE, "Brier": Brier}

log_dir = f"/tmp/fsvi_mnist_example/{datetime.now().strftime('%Y%m%d-%H%M%S')}"
logger = SummaryWriter(log_dir=log_dir)

loss_fn = (
    SampleFSVILoss(n_samples_nll=4, n_samples_kl=16)
    if not mcdropout
    else NLLLoss(n_samples_nll=4)
)

trainer = Trainer(
    loss_fn=loss_fn,
    n_epochs=30,
    opt=opt,
    metrics=metrics,  # type: ignore[arg-type]
    logger=logger,
)
_ = trainer.train(model, dl, sampler, val_dl, key=key)
model = trainer.best_model

ood_metrics = Trainer.eval(
    model,
    ood_dl,
    metrics={"EntropyAUROC": EntropyAUROC},
    n_samples=128,
    key=jax.random.PRNGKey(20260407),
)
for k, v in ood_metrics.items():
    logger.add_scalar(f"metric/ood_{k}", v, 29)
