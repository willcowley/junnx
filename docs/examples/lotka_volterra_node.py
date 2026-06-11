# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Function-space variational inference with neural ODEs
#
# In this notebook we combine function space inference with a neural ODE to solve the Lotka-Volterra (predator-prey) system.
#
# We being with using this [diffrax example](https://docs.kidger.site/diffrax/examples/coupled_odes/) to generate some training data.

import jax

# %%
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
from diffrax import ODETerm, SaveAt, Tsit5, diffeqsolve


# %%
def vector_field(t, y, args):
    prey, predator = y
    α, β, γ, δ = args
    d_prey = α * prey - β * prey * predator
    d_predator = -γ * predator + δ * prey * predator
    d_y = d_prey, d_predator
    return d_y


# %%
ts = jnp.linspace(0, 100, 1_001)
sol = diffeqsolve(
    terms=ODETerm(vector_field),
    solver=Tsit5(),
    t0=ts[0],
    t1=ts[-1],
    dt0=0.1,
    y0=(10.0, 10.0),
    args=(0.1, 0.02, 0.4, 0.02),
    saveat=SaveAt(ts=ts),
)

# %%
fig, ax = plt.subplots(1, 1, figsize=(4, 4))
(l1,) = ax.plot(ts, sol.ys[0], label="prey")
(l2,) = ax.plot(ts, sol.ys[1], label="predator")

# %%
fig, ax = plt.subplots(1, 1, figsize=(4, 4))
ax.plot(sol.ys[0], sol.ys[1])

# %%
key = jr.PRNGKey(seed=42)
perm_idxs = jr.permutation(key, jnp.arange(len(ts))[:400])
n_data = 128  # 16
prey_idxs = perm_idxs[:n_data]
pred_idxs = perm_idxs[n_data : 2 * n_data]

# %%
nans = jnp.repeat(jnp.nan, n_data)
ys = jnp.stack(
    [
        jnp.concat([sol.ys[0][prey_idxs], nans]),
        jnp.concat([nans, sol.ys[1][pred_idxs]]),
    ],
    axis=-1,
)
ys = jnp.stack(
    [
        jnp.concat([sol.ys[0][prey_idxs], sol.ys[0][pred_idxs]]),
        jnp.concat([sol.ys[1][prey_idxs], sol.ys[1][pred_idxs]]),
    ],
    axis=-1,
)
xs = jnp.concat([ts[prey_idxs], ts[pred_idxs]])
sort_idxs = jnp.argsort(xs)
xs = xs[sort_idxs]
ys = ys[sort_idxs]
ymask = ~jnp.isnan(ys)
ymask = None

import tensorflow_probability.substrates.jax as tfp

# %%
from junnx.datasets import (
    DataLoader,
    StandardizeTransformFn,
    TensorDataset,
    TransformedTensorDataset,
)

x_transform = tfp.bijectors.Scale(scale=jnp.array(1e-2))
y_transform = StandardizeTransformFn(axis=None, batch_axis=None).fit(ys)
_ds = TensorDataset(xs, ys, ymask)
ds = TransformedTensorDataset(_ds, x_transform, y_transform)

dl = DataLoader(
    ds, batch_size=n_data * 2, shuffle=False, key=jr.PRNGKey(0)
)  # TODO t must be monotonic in batch?

# %%
fig, ax = plt.subplots(1, 1, figsize=(4, 4))
(l1,) = ax.plot(ts, sol.ys[0], label="prey")
(l2,) = ax.plot(ts, sol.ys[1], label="predator")
ax.scatter(ts[prey_idxs], sol.ys[0][prey_idxs], facecolor=l1.get_color(), edgecolor="k", s=10)
ax.scatter(ts[pred_idxs], sol.ys[1][pred_idxs], facecolor=l2.get_color(), edgecolor="k", s=10)

from typing import Sequence

import equinox as eqx
from diffrax import PIDController, RecursiveCheckpointAdjoint
import diffrax
# %%
from junnx.layers import DenseStochasticLayer
from junnx.net import _ACTIVATIONS, StochasticNet, TractableStochasticNet


class TractableMLP(TractableStochasticNet):

    layers: Sequence[eqx.nn.Linear]
    _last_layer: DenseStochasticLayer

    activation: str = eqx.field(static=True)

    def __init__(
        self,
        n_in,
        n_out,
        n_hidden=32,
        depth: int = 2,
        use_bias: bool = True,
        *,
        key: jnp.ndarray,
    ) -> None:

        layers = []
        for i in range(depth):
            key_layer, key = jr.split(key)
            if i == 0:
                layer = eqx.nn.Linear(n_in, n_hidden, use_bias, key=key_layer)
            else:
                layer = eqx.nn.Linear(n_hidden, n_hidden, use_bias, key=key_layer)
            layers.append(layer)
        key_layer, key = jr.split(key)
        self.layers = layers
        self._last_layer = DenseStochasticLayer(n_hidden, n_out, use_bias, key=key_layer)
        self.activation = "silu"

    def _call_wout_last_layer(self, x: jnp.ndarray, key: jnp.ndarray) -> jnp.ndarray:
        for layer in self.layers:
            x = layer(x)
            x = _ACTIVATIONS[self.activation](x)
        return x

    @property
    def last_layer(self) -> DenseStochasticLayer:
        return self._last_layer


class StochasticVectorField(eqx.Module):
    ln_scale: jnp.ndarray
    mlp: TractableMLP

    def __init__(self, n_states: int, n_hidden: int, depth: int, *, key) -> None:
        self.ln_scale = jnp.ones(shape=n_states) * jnp.log(20.0)
        self.mlp = TractableMLP(n_states, n_states, n_hidden, depth, key=key)

    def __call__(self, t, y, args):
        (key,) = args
        # return self.mlp(y, key)
        return jnp.exp(self.ln_scale) * jnp.tanh(self.mlp(y, key))


class StochasticNODE(StochasticNet):
    vector_field: StochasticVectorField
    y0: jnp.ndarray

    def __init__(self, n_states: int, n_hidden: int, depth: int, *, key) -> None:
        self.vector_field = StochasticVectorField(n_states, n_hidden, depth, key=key)
        self.y0 = jnp.zeros(shape=(n_states,)) #- 0.19580203

    def __call__(self, x, key) -> jnp.ndarray:
        # x is actually time here (in increasing monotonic order)
        sol = diffeqsolve(
            terms=ODETerm(self.vector_field),
            solver=diffrax.Bosh3(),
            t0=0.0,
            t1=x.max(),
            dt0=1e-2,
            y0=self.y0,
            # args=(key,),
            # args=(jr.PRNGKey(42),),
            stepsize_controller=diffrax.PIDController(rtol=1e-3, atol=1e-6),
            saveat=SaveAt(ts=x),
            max_steps=1_024,
        )
        return sol.ys

    def predict_f_samples(
        self, x: jnp.ndarray, n_samples: int, *, key: jnp.ndarray
    ) -> jnp.ndarray:
        # x: [N,]
        keys = jax.random.split(key, n_samples)
        idx = jnp.argsort(x)  # NB: x is time here!
        predf = jax.vmap(self, in_axes=(None, 0))(x[idx], keys)
        ridx = jnp.argsort(idx)
        return predf[:, ridx]  # [S, N, O]


from tensorflow_probability.substrates import jax as tfp

# %%
from junnx.likelihoods import GaussianLikelihood


class _NormalLikelihood(GaussianLikelihood):

    def __call__(self, x: jnp.ndarray) -> tfp.distributions.Normal:
        # x: [S, N, O]
        scale = jax.lax.stop_gradient(self.scale)
        return tfp.distributions.Normal(loc=x, scale=scale)


# %%
from junnx.model import TrainingModel
from junnx.priors import Matern52Prior
from junnx.variational import GaussianVariationalDistribution

model = TrainingModel(
    net=StochasticNODE(2, 128, 3, key=jr.PRNGKey(42)),
    likelihood=_NormalLikelihood(
        scale_init=(jnp.sqrt(0.5), jnp.sqrt(0.5)),
        bijector=tfp.bijectors.Softplus(),
    ),
    # likelihood=GaussianLikelihood(
    #     scale_init=(1.0, 1.0,),
    #     bijector=tfp.bijectors.Softplus(),
    # ),
    prior=Matern52Prior(lengthscales=(0.5, 0.5)),  # not used w/ NLLLoss
    variational_dist=GaussianVariationalDistribution(),  # not used w/ NLLLoss
)

# %%
from junnx.loss_fns import NLLLoss

loss_fn = NLLLoss(n_samples_nll=1)

# %%
import optax

opt = optax.adam(5e-4)

from datetime import datetime

from tensorboardX import SummaryWriter

# %%
from junnx.trainer import Trainer

logdir = "/tmp/lv_node/" + datetime.now().strftime("%Y%m%d%H%M%S")
logger = SummaryWriter(logdir)
trainer = Trainer(
    n_epochs=400,
    opt=opt,
    loss_fn=loss_fn,
    logger=logger,
)
trainer.train(model, dl, key=jr.PRNGKey(42))

# %%
best_model = trainer.best_model
ts_ = jnp.linspace(0, 1.0, 101)
ypred = best_model.net(ts_, key=jr.PRNGKey(42))
fig, ax = plt.subplots(1, 1, figsize=(4, 4))
ax.scatter(
    ts[prey_idxs] * 1e-2,
    y_transform(sol.ys[0][prey_idxs]),
    facecolor=l1.get_color(),
    edgecolor="k",
    s=10,
)
ax.scatter(
    ts[pred_idxs] * 1e-2,
    y_transform(sol.ys[1][pred_idxs]),
    facecolor=l2.get_color(),
    edgecolor="k",
    s=10,
)
ax.plot(ts_, ypred[:, 0])
ax.plot(ts_, ypred[:, 1])
fig.savefig("/tmp/lv_node.png")
# %%
