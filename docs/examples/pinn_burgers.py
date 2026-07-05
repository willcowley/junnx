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
# # Function-space variational inference with Physics-Informed Neural Networks
#
# In this notebook we use function-space variational inference in conjunction with a physics-informed neural network to
# estimate the initial conditions of a physical system. We will use Burgers equation in 1D. We will compare our initial
# condition estimates to those of a model trained without a physics-based loss.
#
# We will assume knowledge of:
# - The boundary conditions, \\\(u(1, t) == u(-1, t) == 0.0\\\) (and will build this directly into our model and prior)
# - The form of burgers equation \\\(\frac{\partial u}{\partial t} + u\frac{\partial u}{\partial x} = \nu \frac{\partial^{2} u}{\partial x^{2}}\\\)
# - The value of the viscosity parameter, \\\(\nu\\\)
#
# We will generate some sparse and noisy observational data at later times and use them to infer the initial conditions.

# %%


import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import optax
from tensorflow_probability.substrates import jax as tfp

from junnx.datasets import DataLoader, TensorDataset
from junnx.layers import DenseStochasticLayer
from junnx.likelihoods import GaussianLikelihood
from junnx.loss_fns import TractableFSVILoss
from junnx.model import TrainingModel
from junnx.net import TractableStochasticNet
from junnx.priors import RBFPrior
from junnx.samplers import UniformSampler
from junnx.trainer import Trainer
from junnx.variational import GaussianVariationalDistribution

# %% [markdown]
#
# First, we will generate some training data using the excellent [diffrax](https://docs.kidger.site/diffrax/) library.
# We will spatially discretise \\\(x\in[-1, 1]\\\) to reduce the Burger PDE into a system of ODEs.

# %%


def vector_field(t: jnp.ndarray, u: jnp.ndarray, args: tuple[jnp.ndarray, ...]) -> jnp.ndarray:

    nu, dx = args
    u = jnp.concatenate(
        [
            jnp.zeros(shape=(1,)),  # u(-1, t) = 0.0 (boundary conditions)
            u,
            jnp.zeros(shape=(1,)),  # u(1, t) = 0.0 (boundary conditions)
        ]
    )

    u_l = u[:-2]
    u_m = u[1:-1]
    u_r = u[2:]

    u_x = (u_r - u_l) / (2.0 * dx)
    u_xx = (u_r - 2.0 * u_m + u_l) / dx**2

    return -u_m * u_x + nu * u_xx


# %%

_NU = 1e-2 / jnp.pi


def solve_burgers(y0: jnp.ndarray, xs: jnp.ndarray, ts: jnp.ndarray) -> jnp.ndarray:
    # y0: [Nx - 2,]
    # xs: [Nx,]
    # ts: [Nt,]
    δt = 1e-4

    (nts,) = ts.shape
    sol = diffrax.diffeqsolve(
        terms=diffrax.ODETerm(vector_field),  # type: ignore[arg-type]
        solver=diffrax.Tsit5(),
        t0=ts[0],
        dt0=δt,
        t1=ts[-1],
        y0=y0,  # initial conditions
        args=(_NU, xs[1] - xs[0]),
        saveat=diffrax.SaveAt(ts=ts),
        max_steps=None,
    )

    us = jnp.concatenate(
        [
            jnp.zeros(shape=(nts, 1)),  # u(-1, t) = 0.0 (boundary conditions)
            sol.ys,
            jnp.zeros(shape=(nts, 1)),  # u(1, t) = 0.0 (boundary conditions)
        ],
        axis=-1,
    )
    return us.T  # [Nx, Nt]


x_grid = jnp.linspace(-1.0, 1.0, 2_001)
t_grid = jnp.linspace(0.0, 1.0, 101)
ic = -jnp.sin(x_grid[1:-1] * jnp.pi)
us = solve_burgers(y0=ic, xs=x_grid, ts=t_grid)


# %% [markdown]
#
# We'll inspect our solution below, you can compare to Fig. 1 of the original physics-informed paper
# ([Raissi et al. 2017](https://arxiv.org/abs/1711.10561)).
#

# %%


def plot_u() -> tuple[plt.Figure, plt.Axes]:
    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_axes((0.175, 0.175, 0.65, 0.65))
    cax = fig.add_axes((0.835, 0.175, 0.025, 0.65))
    im = ax.imshow(
        us,
        origin="lower",
        extent=(0.0, 1.0, -1.0, 1.0),
        aspect=0.5,
        cmap="inferno",
        vmin=-1.0,
        vmax=1.0,
    )
    ax.set_xlabel("t")
    ax.set_ylabel("x")
    plt.colorbar(im, ax=ax, cax=cax, label="u")
    return fig, ax


plot_u()

# %% [markdown]
#
# We'll take a sparse slice of this data (and add some noise!) to train our model

# %%

x_slice = slice(700, 1_200, 50)
t_slice = slice(40, None, 10)
xs = x_grid[x_slice]
ts = t_grid[t_slice]

x = jnp.stack([jnp.repeat(xs, len(ts)), jnp.tile(ts, len(xs))], axis=-1)

y = us[x_slice, t_slice].reshape(-1, 1)

eps = 2e-2 * jr.normal(key=jr.PRNGKey(seed=20260527), shape=y.shape)

ds = TensorDataset(x, y + eps)

fig, ax = plot_u()
ax.scatter(*x[:, ::-1].T, c=y, cmap="inferno", s=4, edgecolors="w")  # type: ignore[misc]
plt.savefig("/tmp/burgers_diffrax.png", dpi=200)

# %% [markdown]
#
# Now we'll start extending the `junnx` framework for this problem. First, we'll introduce a random Fourier feature layer
# to ameliorate spectral bias by allowing for higher frequency components.

# %%


class RBFKernelRFFLayer(eqx.Module):

    log_lengthscale: jnp.ndarray
    omega: jnp.ndarray
    bias: jnp.ndarray

    n_in: int = eqx.field(static=True)
    n_out: int = eqx.field(static=True)

    def __init__(self, n_in: int, n_out: int, *, key: jnp.ndarray) -> None:
        self.log_lengthscale = jnp.zeros(n_in)
        key_omega, key_bias = jr.split(key, 2)
        self.omega = jr.normal(key_omega, (n_out, n_in))
        self.bias = jr.uniform(key_bias, (n_out,)) * 2 * jnp.pi
        self.n_in = n_in
        self.n_out = n_out

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        w = jax.lax.stop_gradient(self.omega)
        b = jax.lax.stop_gradient(self.bias)
        ells = jnp.exp(self.log_lengthscale)
        theta = w @ (x / ells) + b
        return jnp.sqrt(2 / self.n_out) * jnp.cos(theta)


# %% [markdown]
#
# Now we'll incorporate this into a simple tractable FSVI network architecture. Note how the boundary conditions are
# enforced by `boundary_condition_fn`. This means they are predicted correctly by design.


# %%
def boundary_condition_fn(x: jnp.ndarray) -> jnp.ndarray:
    return 1 - x**4


# %%


class BurgerNet(TractableStochasticNet):

    enc: eqx.nn.Linear
    rbf: RBFKernelRFFLayer
    fc1: eqx.nn.Linear
    fc2: eqx.nn.Linear
    dec: DenseStochasticLayer

    n_hidden: int = eqx.field(static=True)

    def __init__(self, n_hidden: int = 32, *, key: jnp.ndarray) -> None:
        key_enc, key_rbf, key_fc1, key_fc2, key_dec = jr.split(key, 5)
        n_in = 2
        self.enc = eqx.nn.Linear(n_in, n_hidden, key=key_enc)
        self.rbf = RBFKernelRFFLayer(n_in, n_hidden, key=key_rbf)
        self.fc1 = eqx.nn.Linear(2 * n_hidden, n_hidden, key=key_fc1)
        self.fc2 = eqx.nn.Linear(n_hidden, n_hidden, key=key_fc2)
        self.dec = DenseStochasticLayer(n_hidden, 1, key=key_dec)
        self.n_hidden = n_hidden

    def _call_wout_last_layer(self, x: jnp.ndarray, key: jnp.ndarray) -> jnp.ndarray:
        x_enc = jax.nn.silu(self.enc(x))
        x_rbf = self.rbf(x)
        x = jnp.concatenate([x_enc, x_rbf], axis=-1)
        x = jax.nn.silu(self.fc1(x))
        x = jax.nn.silu(self.fc2(x))
        return x

    def _boundary_conditions(self, x: jnp.ndarray) -> jnp.ndarray:
        # x: [N, 2]
        _x, _ = jnp.split(x, [1], axis=-1)  # [N, 1]
        return boundary_condition_fn(_x)

    def tractable_f_mean_cov(
        self, x: jnp.ndarray, key: jnp.ndarray
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        # x: [N, 2]
        mean, cov = super().tractable_f_mean_cov(x, key)
        bcs = self._boundary_conditions(x)
        return mean * bcs.T, (bcs @ bcs.T) * cov

    def __call__(self, x: jnp.ndarray, key: jnp.ndarray) -> jnp.ndarray:
        bcs = self._boundary_conditions(x)
        x = super().__call__(x, key)  # [N, 1]
        return bcs * x

    @property
    def last_layer(self) -> DenseStochasticLayer:
        return self.dec

    def predict_f_samples(
        self, x: jnp.ndarray, n_samples: int, *, key: jnp.ndarray
    ) -> jnp.ndarray:
        key, key_deterministic = jr.split(key, 2)
        bcs = self._boundary_conditions(x)  # [N, 1]
        x = jax.vmap(lambda _x: self._call_wout_last_layer(_x, key_deterministic))(
            x
        )  # only last_layer is stochastic
        keys = jax.random.split(key, n_samples)
        predf = jax.vmap(jax.vmap(self.last_layer, in_axes=(0, None)), in_axes=(None, 0))(
            x, keys
        )
        return predf * bcs  # [S, N, O]


# %% [markdown]
#
# Now we'll introduce a custom prior. Note that this is only over the spatial dimension as it is only applied over the
# initial conditions (when \\\(t=0.0\\\)).

# %%


class BurgerPrior(RBFPrior):

    def __init__(self):
        super().__init__(lengthscales=(0.2,), variance=0.5)

    def wx(self, x: jnp.ndarray) -> jnp.ndarray:
        return boundary_condition_fn(x)  # enforces boundary conditions

    def kernel(self, x: jnp.ndarray, x2: jnp.ndarray | None = None) -> jnp.ndarray:
        _x, _ = jnp.split(x, [1], axis=-1)
        kxx = super().kernel(_x)
        wx = self.wx(_x)
        return (wx @ wx.T) * kxx


# %% [markdown]
#
# Finally, we'll extend the `TractableFSVILoss` to incorporate a physics-informed term that penalises the network from
# deviating from the evolution governed by Burgers equation. Note how we use `jacfwd` and `vmap` to apply this
# efficiently over our batch

# %%


class BurgersLoss(TractableFSVILoss):

    use_physics_loss: bool = eqx.field(static=True, default=True)

    def __call__(
        self,
        m: TrainingModel,
        x: jnp.ndarray,
        y: jnp.ndarray,
        context_x: jnp.ndarray | None,
        n_batches_per_epoch: int,
        *,
        key: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        assert context_x is not None
        t0 = jnp.zeros_like(context_x)
        key_fsvi, key_phys = jr.split(key, 2)
        context_xt0 = jnp.concatenate([context_x, t0], axis=-1)
        tractable_fsvi_loss, predf = super().__call__(
            m, x, y, context_xt0, n_batches_per_epoch, key=key_fsvi
        )
        if not self.use_physics_loss:
            return tractable_fsvi_loss, predf

        def u(x: jnp.ndarray, t: jnp.ndarray, key: jnp.ndarray) -> jnp.ndarray:
            xt = jnp.concatenate([x, t], axis=-1)
            return m.net(xt, key)

        ut_fn = jax.jacfwd(u, argnums=1)
        ux_fn = jax.jacfwd(u, argnums=0)
        uxx_fn = jax.jacfwd(ux_fn, argnums=0)

        def _burgers_physics_loss(xt: jnp.ndarray, key: jnp.ndarray) -> jnp.ndarray:
            # u_t + u * u_x - nu * u_xx = 0.0
            x, t = jnp.split(xt, 2, axis=-1)
            return ut_fn(x, t, key) + u(x, t, key) * ux_fn(x, t, key) - _NU * uxx_fn(x, t, key)

        keys_phys = jr.split(key_phys, self.n_samples_nll + 1)
        collocation_x = jr.uniform(
            key=keys_phys[0],
            shape=x.shape,
            minval=jnp.array([-1, 0]),
            maxval=jnp.array([1, 1]),
        )
        physics_loss = jax.vmap(
            jax.vmap(_burgers_physics_loss, in_axes=(0, None)), in_axes=(None, 0)
        )(collocation_x, keys_phys[1:])
        physics_loss = jnp.mean(jnp.square(physics_loss))
        return tractable_fsvi_loss + physics_loss, predf


# %% [markdown]
#
# Now we'll put everything together into the `junnx` `TrainingModel`. For more information on what each component
# represents see the `1D Regression with Stochastic Neural Networks` example.

# %%
key = jr.PRNGKey(0)
key_net, key_dl, key_train = jr.split(key, 3)

sampler = UniformSampler(n_dim=1, n_samples=32, low=(-1,), high=(1,))  # x at t=0.0


def init_model() -> TrainingModel:
    return TrainingModel(
        net=BurgerNet(key=key_net),
        likelihood=GaussianLikelihood(
            scale_init=(1.0,),
            bijector=tfp.bijectors.Softplus(),
        ),
        prior=BurgerPrior(),
        variational_dist=GaussianVariationalDistribution(),
    )


dl = DataLoader(ds, batch_size=16, shuffle=True, key=key_dl)

# %% [markdown]
#
# Now everything's ready to train some models! First, we'll train a model without the physics-informed term and inspect
# its predictions for the initial conditions.

# %%
N_EPOCHS = 5_000
trainer = Trainer(
    loss_fn=BurgersLoss(n_samples_nll=4, use_physics_loss=False),
    n_epochs=N_EPOCHS,
    opt=optax.adam(1e-3),
)

# %%

_ = trainer.train(init_model(), dl, sampler, key=key_train)
m_no_physics = trainer.best_model
# %%
key_plot = jr.PRNGKey(seed=20250603)


def _plot_model(
    ax: plt.Axes, m: TrainingModel, n_samples: int, xx: jnp.ndarray, x_plot_idx: int
) -> None:
    pred_f_samples = m.net.predict_f_samples(xx, n_samples, key=key_plot)  # [S, N, O]
    pred_f_mean = jnp.mean(pred_f_samples, axis=0)  # [N, O]
    pred_f_std = jnp.std(pred_f_samples, axis=0)  # [N, O]
    ymean, yvar = m.predict_y_mean_var(pred_f_samples)  # [N, O], [N, O]
    ystd = jnp.sqrt(yvar)  # [N, O]
    ax.plot(xx[:, x_plot_idx], pred_f_mean, c="k", lw=0.8)
    ax.fill_between(
        xx[:, x_plot_idx],
        pred_f_mean[:, 0] + 1.95 * pred_f_std[:, 0],  # epistemic uncertainty
        pred_f_mean[:, 0] - 1.95 * pred_f_std[:, 0],
        color="#4E79A7",
        alpha=0.5,
        lw=0.0,
    )
    ax.plot(xx[:, x_plot_idx], ymean, c="#59A14F", lw=0.8, ls="--")
    ax.plot(
        xx[:, x_plot_idx], ymean + 1.95 * ystd, c="#59A14F", lw=0.66, ls="--"
    )  # epistemic + aleatoric uncertainty
    ax.plot(xx[:, x_plot_idx], ymean - 1.95 * ystd, c="#59A14F", lw=0.66, ls="--")
    for ii in range(4):
        ax.plot(xx[:, x_plot_idx], pred_f_samples[ii, :, 0], c="#4E79A7", lw=0.66, alpha=0.66)


def plot_model_ic(m: TrainingModel, n_samples: int = 256) -> None:
    x_plot = jnp.linspace(-1, 1, 101)
    t0 = jnp.zeros_like(x_plot)
    xx = jnp.stack([x_plot, t0], axis=-1)
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    _plot_model(ax, m, n_samples, xx, 0)
    ax.plot(x_plot, -jnp.sin(x_plot * jnp.pi), c="#E15759", ls="-")
    ax.plot(x_plot, 1.95 * boundary_condition_fn(x_plot) / jnp.sqrt(2), ls=":", c="k")  # prior
    ax.plot(
        x_plot, -1.95 * boundary_condition_fn(x_plot) / jnp.sqrt(2), ls=":", c="k"
    )  # prior
    ax.set_ylim(-1.8, 1.8)
    ax.set_xlabel("x")
    ax.set_ylabel("u(x, t=0)")
    fig.savefig("/tmp/burgers_ic.png", dpi=200)


# %%
plot_model_ic(m_no_physics)

# %% [markdown]
#
# This provides reasonable uncertainty estimates, matching our prior well, but it is not a great prediction of the true
# initial conditions.
#
# Let's try training a model with the physics informed loss and see what difference that makes

# %%

trainer = Trainer(
    loss_fn=BurgersLoss(n_samples_nll=4, use_physics_loss=True),
    n_epochs=N_EPOCHS,
    opt=optax.adam(1e-3),
)

# %%

dl.key = key_dl  # reset the dl state
_ = trainer.train(init_model(), dl, sampler, key=key_train)
m_physics = trainer.best_model

# %%
plot_model_ic(m_physics)

# %% [markdown]
#
# This shows a much more meaningful estimate of our true initial conditions, showing how the evolution enforced by our
# physics based loss informs our neural network predictions!
#
# Let's see how well calibrated our uncertainty estimates are. We'll sample from the model's initial conditions and
# solve the equation with `diffrax`, comparing it to our training data. We can compare the resulting trajectories for
# our models trained with and without a physics-informed loss.

# %%
t0s = jnp.zeros_like(x_grid)
x_ic = jnp.stack([x_grid, t0s], axis=-1)


def get_uss(m: TrainingModel) -> jnp.ndarray:
    ics = m.net.predict_f_samples(x_ic, n_samples=4, key=jr.PRNGKey(20260601))
    uss = jax.vmap(solve_burgers, in_axes=(0, None, None))(ics[:, 1:-1, 0], x_grid, t_grid)
    uss_xsliced = uss[:, x_slice]
    return uss_xsliced


# %%
uss_phys = get_uss(m_physics)
uss_no_phys = get_uss(m_no_physics)


# %%
def plot_model_t(
    m: TrainingModel,
    uss: jnp.ndarray,
    pos_idx: int,
    ax: plt.Axes,
    n_samples: int = 256,
) -> None:
    nt = 7
    x_here = jnp.stack([jnp.repeat(x_grid[x_slice][pos_idx], len(t_grid)), t_grid], axis=-1)
    _plot_model(ax, m, n_samples, x_here, 1)
    ax.scatter(
        ds.x[nt * pos_idx : nt * (pos_idx + 1), 1:],
        ds.y[nt * pos_idx : nt * (pos_idx + 1)],
        facecolor="#E15759",
        s=6,
        edgecolor="k",
    )
    ax.plot(t_grid, us[x_slice][pos_idx], c="#E15759")

    for j in range(len(uss)):
        ax.plot(t_grid, uss[j, pos_idx], c="#E15759", lw=0.5, ls=":")
    ax.set_xlabel("t")
    ax.set_ylabel(f"u(x={ds.x[nt * pos_idx, 0]: 4.2f}, t)")


# %%
for m, uss, title in [
    (m_physics, uss_phys, "w/ physics loss"),
    (m_no_physics, uss_no_phys, "w/out physics loss"),
]:
    fig, axes = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)
    fig.suptitle(title)
    ax0, ax1 = axes
    plot_model_t(m, uss, 0, ax0)
    plot_model_t(m, uss, 9, ax1)

# %% [markdown]
# The physics-informed model uncertainty is not perfect. The simulated trajectories (dotted red lines) generated from
# the model's predicted initial conditions are slightly more dispersed than the training data at later times. However,
# the predictions are far more meaningful and physically realistic than the model trained without a physics-informed
# loss.

# %%
