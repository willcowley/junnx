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
# # Function-space variational inference with Hamiltonian neural ODEs
#
# In this notebook we combine function space inference with a neural ODE

# %%

from datetime import datetime
from typing import Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import optax
import tensorflow_probability.substrates.jax as tfp
from diffrax import ODETerm, PIDController, SaveAt, Tsit5, diffeqsolve
from tensorboardX import SummaryWriter

from junnx.datasets import DataLoader, TensorDataset
from junnx.layers import DenseStochasticLayer
from junnx.likelihoods import GaussianLikelihood
from junnx.loss_fns import TractableFSVILoss
from junnx.model import TrainingModel
from junnx.net import _ACTIVATIONS, StochasticNet, TractableStochasticNet
from junnx.priors import RBFPrior
from junnx.samplers import UniformSampler
from junnx.trainer import Trainer
from junnx.variational import GaussianVariationalDistribution

# %%


def shm_vector_field(t, y, args):
    x, xdot = y
    (omega,) = args
    xddot = -(omega**2) * x
    return jnp.array([xdot, xddot])


# %%

y0 = jnp.array([1.0, 0.0])

t_grid = jnp.linspace(0, jnp.pi, 101)
sol = diffeqsolve(
    terms=ODETerm(shm_vector_field),
    solver=Tsit5(),
    t0=t_grid[0],
    t1=t_grid[-1],
    dt0=1e-2,
    y0=y0,
    args=(1.0,),
    saveat=SaveAt(ts=t_grid),
)

xs, xdots = sol.ys.T

# %%
fig, ax = plt.subplots(1, 1, figsize=(4, 4))
ax.plot(t_grid, xs**2 + xdots**2)

# %%

fig, ax = plt.subplots(1, 1, figsize=(4, 4))
ax.scatter(xs, xdots, c=t_grid, cmap="viridis")
x_lim = ax.get_xlim()
y_lim = ax.get_ylim()
fig.savefig("/tmp/shm_phase.png", dpi=200)

N_data = 32
idxs = jr.permutation(key=jr.PRNGKey(seed=3), x=len(t_grid))
fig, ax = plt.subplots(1, 1, figsize=(4, 4))
key_data = jr.PRNGKey(42)
noise = 1e-1 * jr.normal(key_data, shape=(N_data, 2), dtype=xs.dtype)
ax.scatter(
    xs[idxs][:N_data] + noise[:, 0],
    xdots[idxs][:N_data] + noise[:, 1],
    c=t_grid[idxs][:N_data],
    cmap="viridis",
)
ax.set_xlim(*x_lim)
ax.set_ylim(*y_lim)
fig.savefig("/tmp/shm_phase_data.png", dpi=200)

fig, ax = plt.subplots(1, 1, figsize=(4, 4))
ax.plot(sol.ts, xs)
ax.plot(sol.ts, xdots)
fig.savefig("/tmp/shm_t.png", dpi=200)

# %%


class TractableMLP(TractableStochasticNet):

    layers: Sequence[eqx.nn.Linear]
    _last_layer: DenseStochasticLayer

    activation: str = eqx.field(static=True)

    def __init__(
        self,
        n_in: int,
        n_out: int,
        n_hidden: int = 32,
        depth: int = 3,
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
        key_layer, key_rbf = jr.split(key)
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


# %%


class HamiltonianVectorField(eqx.Module):

    hamiltonian_net: TractableMLP

    def __call__(self, t, y, args) -> jnp.ndarray:
        # y: [2,]
        (key,) = args

        def h_fn(z: jnp.ndarray) -> jnp.ndarray:
            return self.hamiltonian_net(z, key)[0]  # scalar

        grad_h = jax.grad(h_fn)(y)  # [2,]

        J = jnp.array([[0.0, 1.0], [-1.0, 0.0]])  # [2, 2]

        return J @ grad_h  # [2,]


# %%


class StochasticHamiltonianNODE(StochasticNet):
    vector_field: HamiltonianVectorField
    y0: jnp.ndarray

    def __init__(self, n_states: int, n_hidden: int, depth: int, *, key) -> None:
        self.vector_field = HamiltonianVectorField(
            hamiltonian_net=TractableMLP(n_states, 1, n_hidden, depth, use_bias=True, key=key)
        )
        self.y0 = jnp.array([1.0, 0.0])  # initial conditions

    def __call__(self, x, key) -> jnp.ndarray:
        # x is actually time here (in increasing monotonic order)
        sol = diffeqsolve(
            terms=ODETerm(self.vector_field),
            solver=Tsit5(),
            t0=0.0,
            t1=x[-1],
            dt0=1e-2,
            y0=jax.lax.stop_gradient(self.y0),
            args=(key,),
            stepsize_controller=PIDController(rtol=1e-3, atol=1e-6),
            saveat=SaveAt(ts=x),
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


# %%


ds = TensorDataset(
    x=t_grid[idxs][:N_data],
    y=jnp.stack([xs[idxs][:N_data], xdots[idxs][:N_data]], axis=-1) + noise,
)

# %%

model = TrainingModel(
    net=StochasticHamiltonianNODE(n_states=2, n_hidden=64, depth=3, key=jr.PRNGKey(20260704)),
    likelihood=GaussianLikelihood(
        scale_init=(1.0, 1.0),
        bijector=tfp.bijectors.Softplus(),
    ),
    prior=RBFPrior(lengthscales=(1.0, 1.0)),
    variational_dist=GaussianVariationalDistribution(),
)

# %%


class TractableHamiltonianFSVILoss(TractableFSVILoss):

    def get_tractable_net(self, model: TrainingModel) -> TractableStochasticNet:
        net = model.net
        assert isinstance(net, StochasticHamiltonianNODE)
        return net.vector_field.hamiltonian_net


# %%


_N_SAMPLES = 8
loss_fn_fsvi = TractableHamiltonianFSVILoss(n_samples_nll=_N_SAMPLES)

sampler = UniformSampler(n_dim=2, n_samples=64, low=(-2.0, -2.0), high=(2.0, 2.0))
dl = DataLoader(ds, batch_size=32, shuffle=False, key=jr.PRNGKey(20260703))

opt = optax.adam(1e-3)

# %%
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
logdir = "/home/wcowley/junnx_out/shm_node/"
logger = SummaryWriter(logdir + timestamp)
trainer = Trainer(
    n_epochs=1_500,
    opt=opt,
    loss_fn=loss_fn_fsvi,
    logger=logger,
)
trainer.train(model, dl, sampler, key=jr.PRNGKey(20260704))
m = trainer.best_model

# %%
# plot hamiltonian estimates

n_grid = 101
x_grid = jnp.linspace(-2, 2, n_grid)
xdot_grid = jnp.linspace(-2, 2, n_grid)
x0, x1 = jnp.meshgrid(x_grid, xdot_grid)
z = jnp.concat([x0.reshape(-1, 1), x1.reshape(-1, 1)], axis=-1)  # [N*N, 2]

m_hamiltonian = m.net.vector_field.hamiltonian_net

key_pred = jr.PRNGKey(20260705)
hs = m_hamiltonian.predict_f_samples(z, 32, key=key_pred)


# %%
fig, axes = plt.subplots(1, 3, figsize=(12 + 1.5, 4))
ax0, ax1, ax2 = axes

ground_truth = 0.5 * (z[:, :1] ** 2 + z[:, 1:] ** 2).reshape(n_grid, n_grid)
mean = hs.mean(axis=0).reshape(n_grid, n_grid)
var = hs.var(axis=0).reshape(n_grid, n_grid)
imshow_kwargs = {"extent": [-2, 2, -2, 2], "origin": "lower"}
ax0.imshow(ground_truth, **imshow_kwargs)
ax1.imshow(mean, **imshow_kwargs)
ax2.imshow(var, **imshow_kwargs, vmin=0.25, vmax=1.0)
for _ax in (ax0, ax1, ax2):
    _ax.scatter(*ds.y.T, s=6, edgecolor="w", linewidths=0.5, color="#4E79A7")
fig.savefig(f"{logdir}/shm_node_hamiltonian_{timestamp}.png", dpi=200)


# %%
dummy_t = jnp.zeros(shape=())

n_grid = 21
x_grid = jnp.linspace(-2, 2, n_grid)
xdot_grid = jnp.linspace(-2, 2, n_grid)
x0, x1 = jnp.meshgrid(x_grid, xdot_grid)
z = jnp.concat([x0.reshape(-1, 1), x1.reshape(-1, 1)], axis=-1)  # [N*N, 2]

d_ground_truth = shm_vector_field(dummy_t, (z[:, 0], z[:, 1]), (1.0,))

ks = jr.split(jr.PRNGKey(0), 64)
dz = jax.vmap(
    jax.vmap(lambda _z, _k: m.net.vector_field(dummy_t, _z, args=(_k,)), in_axes=(0, None)),
    in_axes=(None, 0),
)(z, ks)

# %%
fig, axes = plt.subplots(1, 2, figsize=(4 * 2 + 0.5, 4))
ax0, ax1 = axes
ax0.quiver(*z.T, *d_ground_truth)
ax1.quiver(*z.T, *jnp.mean(dz, axis=0).T)
for _ax in (ax0, ax1):
    _ax.scatter(*ds.y.T, s=8, edgecolor="w", linewidths=0.5, color="#4E79A7")


# %%
n_pred_cycle = 4
t_pred = jnp.linspace(0, 2 * jnp.pi * n_pred_cycle, 100 * n_pred_cycle + 1)
key_pred = jr.PRNGKey(1)

f_samples = m.net.predict_f_samples(t_pred, n_samples=8, key=key_pred)

# %%

fig, axes = plt.subplots(3, 2, figsize=(4 * 2 + 0.5, 4 * 3 + 0.5))
for ax, f in zip(axes.ravel(), f_samples):
    ax.scatter(ds.y[:, 0], ds.y[:, 1], s=4)
    ax.plot(f[:, 0], f[:, 1], lw=0.5)
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)

fig.savefig(f"{logdir}/shm_node_results_{timestamp}.png", dpi=200)

# %%
