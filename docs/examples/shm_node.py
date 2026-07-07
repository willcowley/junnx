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
# In this notebook we combine function space inference with a Hamiltonian neural ODE, that is a network for which the
# Hamiltonian is defined by a stochastic network (i.e. a distribution over functions $H(z, \theta)$, where
# $z=[x, \dot{x}]$) on which we place an RBF function prior. The vector field is then computed according to
# $$f(z, \theta) = J \begin{bmatrix}\frac{\partial{H(z)}}{\partial x}\\ \frac{\partial{H(z)}}{\partial \dot{x}}\end{bmatrix} $$
# where
# $$
# J = \begin{bmatrix} 0 & 1 \\ -1 & 0 \end{bmatrix}\mathrm{.}
# $$
#
# We use the diffrax library to solve the ODE $$ \frac{\mathrm{d}z}{\mathrm{d}t} = f(z, \theta) $$ and compare this to
# sparse noisy data generated from a simple harmonic oscillator. We show that the variance of the resulting vector field
# is well constrained near observed data, but increases out of distribution (OOD).

# %%

from datetime import datetime
from typing import Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.patches
import matplotlib.pyplot as plt
import optax
import tensorflow_probability.substrates.jax as tfp
from diffrax import ODETerm, PIDController, SaveAt, Tsit5, diffeqsolve
from mpl_toolkits.axes_grid1 import make_axes_locatable
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

# %% [markdown]
# First, we generate some observational data. We take a small number of data points from the first 3/4 of a simple
# harmonic oscillator system orbit, and add some observational noise.


# %%
def shm_vector_field(t, y, args):
    x, xdot = y
    (omega,) = args
    xddot = -(omega**2) * x
    return jnp.array([xdot, xddot])


# %%

y0 = jnp.array([1.0, 0.0])

t_grid = jnp.linspace(0, 2 * jnp.pi * 3 / 4, 301)
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


# %% [markdown]
# Now, we create a sparse noisy dataset from the `diffrax` solution

# %%
key_data = jr.PRNGKey(42)
N_data = 32
idxs = jr.permutation(key=jr.PRNGKey(seed=3), x=len(t_grid))
noise = 1e-1 * jr.normal(key_data, shape=(N_data, 2), dtype=xs.dtype)
xs_data = xs[idxs][:N_data] + noise[:, 0]
xdots_data = xdots[idxs][:N_data] + noise[:, 1]
ts_data = t_grid[idxs][:N_data]

# %%
ds = TensorDataset(
    x=ts_data,
    y=jnp.stack([xs_data, xdots_data], axis=-1),
)


# %%


def _fmt_axes(_ax: plt.Axes) -> None:
    _ax.set_yticks(_ax.get_xticks())
    _x_lim = _ax.get_xlim()
    _ax.set_ylim(*_x_lim)
    _ax.set_aspect("equal")
    _ax.set_xlabel(r"$x$")
    _ax.set_ylabel(r"$\dot{x}$")


fig, axes = plt.subplots(1, 2, figsize=(4 * 2 + 0.8, 4 * 0.9))
plt.subplots_adjust(wspace=0.3)
ax_phase, ax_time = axes
# phase-space visualisation
im = ax_phase.scatter(xs_data, xdots_data, c=ts_data, cmap="viridis", label="data")
ax_phase.add_patch(
    matplotlib.patches.Circle((0, 0), radius=1, label="true orbit", fill=False, lw=0.8)
)
_fmt_axes(ax_phase)
x_lim = ax_phase.get_xlim()
ax_phase.legend(loc=0)

append_ax_kwargs = {"position": "right", "size": "5%", "pad": 0.05}
cax = make_axes_locatable(ax_phase).append_axes(**append_ax_kwargs)
fig.colorbar(im, cax=cax, orientation="vertical", label="$t$")

# time visualisation
(l1,) = ax_time.plot(sol.ts, xs, lw=0.8, label=r"$x$")
ax_time.scatter(ts_data, xs_data, c=l1.get_color(), s=4)
(l2,) = ax_time.plot(sol.ts, xdots, lw=0.8, label=r"$\dot{x}$")
ax_time.scatter(ts_data, xdots_data, c=l2.get_color(), s=4)
ax_time.set_xlabel("$t$")
ax_time.set_ylim(*x_lim)
_ = ax_time.legend(loc=0)


# %% [markdown]
# Now we define some `junnx` objects that we will use later on.
# - `TractableMLP`, a simple MLP that has a stochastic final layer.
# - `HamiltonianVectorField`, an object that computes the vector field from a realisation of the Hamiltonian function.
# Note that this enforces that each vector field realisation conserves some constant of motion (though the form of the
# Hamiltonian is not assumed i.e. we do not assume $H(x, \dot{x})=\frac{1}{2}(x^{2} + \dot{x}^{2})$).
# - `StochasticHamiltonianNODE`, an object that integrates a realisation from the `HamiltonianVectorField`
# from $t_{0}=0.0$ to some times $t$ corresponding to the time of observation of our dynamical system.
# - `TractableHamiltonianFSVILoss`, an FSVI loss object that applies the KL divergence term to the Hamiltonian function


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
        key_layer, _ = jr.split(key)
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

    def predict_f_samples(
        self, x: jnp.ndarray, n_samples: int, *, key: jnp.ndarray
    ) -> jnp.ndarray:
        x = jax.vmap(self._call_wout_last_layer, in_axes=(0, None))(x, key)  # key not used
        keys = jr.split(key, n_samples)
        return jax.vmap(jax.vmap(self._last_layer, in_axes=(0, None)), in_axes=(None, 0))(
            x, keys
        )


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

    def __init__(
        self, n_states: int, n_hidden: int, depth: int, use_bias: bool, *, key
    ) -> None:
        self.vector_field = HamiltonianVectorField(
            hamiltonian_net=TractableMLP(
                n_states, 1, n_hidden, depth, use_bias=use_bias, key=key
            )
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
class TractableHamiltonianFSVILoss(TractableFSVILoss):

    def get_tractable_net(self, model: TrainingModel) -> TractableStochasticNet:
        net = model.net
        assert isinstance(net, StochasticHamiltonianNODE)
        return net.vector_field.hamiltonian_net


# %%
sqrt2o2 = jnp.sqrt(2.0).item() / 2.0
model = TrainingModel(
    net=StochasticHamiltonianNODE(
        n_states=2, n_hidden=128, depth=3, use_bias=True, key=jr.PRNGKey(20260704)
    ),
    likelihood=GaussianLikelihood(
        scale_init=(1.0, 1.0),
        bijector=tfp.bijectors.Softplus(),
    ),
    prior=RBFPrior(lengthscales=(sqrt2o2, sqrt2o2)),
    variational_dist=GaussianVariationalDistribution(),
)

# %%
Z_MAX = 2.0

# %%
loss_fn_fsvi = TractableHamiltonianFSVILoss(n_samples_nll=8)
sampler = UniformSampler(
    n_dim=2, n_samples=64, low=(-Z_MAX * 2, -Z_MAX * 2), high=(Z_MAX * 2, Z_MAX * 2)
)
dl = DataLoader(ds, batch_size=32, shuffle=False, key=jr.PRNGKey(20260703))

opt = optax.adam(1e-3)

# %%
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
logdir = "/home/wcowley/junnx_out/shm_node/"
logger = SummaryWriter(logdir + timestamp)
trainer = Trainer(
    n_epochs=2_000,
    opt=opt,
    loss_fn=loss_fn_fsvi,
    logger=logger,
)
trainer.train(model, dl, sampler, key=jr.PRNGKey(20260704))
m = trainer.best_model


# %%
def get_zgrid(n_grid: int) -> jnp.ndarray:
    xgrid = jnp.linspace(-Z_MAX, Z_MAX, n_grid)
    x0, x1 = jnp.meshgrid(xgrid, xgrid)
    return jnp.concat([x0.reshape(-1, 1), x1.reshape(-1, 1)], axis=-1)  # [N*N, 2]


n_grid = 101
z = get_zgrid(n_grid)


# %%
dummy_t = jnp.zeros(shape=())

n_gridq = 11
zq = get_zgrid(n_gridq)

d_ground_truth = shm_vector_field(dummy_t, (zq[:, 0], zq[:, 1]), (1.0,))

ks = jr.split(jr.PRNGKey(20260705), 256)
dz_fn = jax.jit(
    jax.vmap(
        jax.vmap(
            lambda _z, _k: m.net.vector_field(dummy_t, _z, args=(_k,)), in_axes=(0, None)  # type: ignore[attr-defined]
        ),
        in_axes=(None, 0),
    )
)
dz = dz_fn(zq, ks)

# %% [markdown]
# Now, we'll compare the model predictions for our vector field to the ground truth (and overlay our training data for
# comparison). Note that we don't compare the Hamiltonian function directly, as only its gradient is constrained by our
# data. We also compare the variance of our vector field, noting that it is low near to the observed data, but increases
# significantly OOD. This is exactly the result we expect with our FSVI framework!

# %%
# vector field (mean) comparison
fig, axes = plt.subplots(1, 2, figsize=(4 * 2 + 0.5, 4))
ax0, ax1 = axes
quiver_kwargs = {"scale": 12, "scale_units": "inches", "headwidth": 3, "headlength": 4}
ax0.quiver(*zq.T, *d_ground_truth, **quiver_kwargs)
ax0.set_title("Ground Truth")
ax1.quiver(*zq.T, *jnp.mean(dz, axis=0).T, **quiver_kwargs)
ax1.set_title("FSVI")
for _ax in (ax0, ax1):
    _ax.scatter(*ds.y.T, s=8, facecolor="#E15759", edgecolor="k", zorder=1)
    _fmt_axes(_ax)

# var plot
fig, ax = plt.subplots(1, 1, figsize=(4, 4))
dz_ = dz_fn(z, ks)
dz_var = jnp.var(dz_, axis=0).sum(axis=-1)
im = ax.imshow(
    dz_var.reshape(n_grid, n_grid),
    extent=(-Z_MAX, Z_MAX, -Z_MAX, Z_MAX),
    origin="lower",
    vmin=0.0,
    vmax=3.0,
)
ax.scatter(*ds.y.T, s=8, facecolor="#E15759", edgecolor="w", zorder=1)  # type: ignore[misc]
_fmt_axes(ax)
cax = make_axes_locatable(ax).append_axes(**append_ax_kwargs)
fig.colorbar(
    im, cax=cax, orientation="vertical", label=r"$\mathrm{tr}\,\mathrm{Cov}[f(x,\dot{x})]$"
)


# %% [markdown]
# Finally, we can inspect individual trajectories predicted by the model. Note how we are making predictions over time
# window much greater than our original training data.

# %%
n_pred_cycle = 8
t_pred = jnp.linspace(0, 2 * jnp.pi * n_pred_cycle, 100 * n_pred_cycle + 1)
key_pred = jr.PRNGKey(20260710)

f_samples = m.net.predict_f_samples(t_pred, n_samples=4, key=key_pred)


# %%
def plot_trajectories(trajectories: jnp.ndarray) -> plt.Figure:
    # trajectories: [S, N, 2]
    fig, axes = plt.subplots(2, 2, figsize=(4 * 2 + 0.5, 4 * 2 + 0.5))
    for ax, f in zip(axes.ravel(), trajectories):
        ax.scatter(*ds.y.T, s=8, facecolor="#E15759", edgecolor="k", zorder=1)
        ax.plot(f[:, 0], f[:, 1], lw=0.5)
        ax.set_xlim(-4, 4)
        _fmt_axes(ax)
    return fig


fig = plot_trajectories(f_samples)

# %% [markdown]
# We can also change our initial conditions. These initial conditions are further from our training data, note the
# impact this has on the individual trajectories!

# %%
m_new_y0 = eqx.tree_at(lambda _m: _m.net.y0, m, jnp.array([sqrt2o2, sqrt2o2]))
f_samples_new_y0 = m_new_y0.net.predict_f_samples(t_pred, n_samples=6, key=key_pred)
fig = plot_trajectories(f_samples_new_y0)

# %%
