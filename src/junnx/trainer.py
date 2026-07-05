from typing import Mapping, Optional, Type

import equinox as eqx
import jax
import optax
from jax import numpy as jnp
from tensorboardX import SummaryWriter
from tqdm import tqdm

from junnx.datasets import DataLoader
from junnx.loss_fns import LossFn
from junnx.metrics import Metric
from junnx.model import TrainingModel
from junnx.samplers import Sampler


def _loss_fn(
    trainable: TrainingModel,
    static: TrainingModel,
    x: jnp.ndarray,
    y: jnp.ndarray,
    context_x: jnp.ndarray | None,
    loss_fn: LossFn,
    n_batches_per_epoch: int,
    *,
    key: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    m = eqx.combine(trainable, static)
    return loss_fn(
        m,
        x,
        y,
        context_x,
        n_batches_per_epoch,
        key=key,
    )


@eqx.filter_jit
def train_step(
    trainable: TrainingModel,
    static: TrainingModel,
    x: jnp.ndarray,
    y: jnp.ndarray,
    context_x: jnp.ndarray | None,
    loss_fn: LossFn,
    n_batches_per_epoch: int,
    opt: optax.GradientTransformation,
    opt_state: optax.OptState,
    *,
    key: jnp.ndarray,
) -> tuple[jnp.ndarray, TrainingModel, optax.OptState]:
    """
    Performs a single function-space variational inference training step given a batch of data.
    Computes the scalar loss and updates the trainable parameters accordingly using the
    provided optimiser and optimiser state.

     Args:
        trainable: The trainable part(s) of the model that gradients are to be computed with
            respect to.
        static: The static part(s) of the model that is not updated during training.
        x: The input data batch.
        y: The target data batch.
        loss_fn: The loss function that is to be used.
        n_batches_per_epoch: The number of data batches per epoch, used to scale the KL
            divergence.
        opt: The optax optimiser to use for updating the trainable parameters.
        opt_state: The current state of the optimiser.
        key: JAX PRNG key to seed sampling (keyword-only argument).

    Returns:
        A tuple containing:
        - The scalar per-batch loss.
        - The updated trainable part(s) of the model.
        - The updated optimiser state.
    """
    (loss, _), grads = eqx.filter_value_and_grad(_loss_fn, has_aux=True)(
        trainable,
        static,
        x,
        y,
        context_x,
        loss_fn,
        n_batches_per_epoch,
        key=key,
    )
    update, opt_state = opt.update(grads, opt_state)
    trainable = eqx.apply_updates(trainable, update)
    return loss, trainable, opt_state


@eqx.filter_jit
def val_step(
    trainable: TrainingModel,
    static: TrainingModel,
    x: jnp.ndarray,
    y: jnp.ndarray,
    context_x: jnp.ndarray | None,
    loss_fn: LossFn,
    n_batches_per_epoch: int,
    *,
    key: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    return _loss_fn(
        trainable,
        static,
        x,
        y,
        context_x,
        loss_fn,
        n_batches_per_epoch,
        key=key,
    )


class Trainer:

    def __init__(
        self,
        n_epochs: int,
        opt: optax.GradientTransformation,
        loss_fn: LossFn,
        opt_state: Optional[optax.OptState] = None,
        metrics: Optional[Mapping[str, Type[Metric]]] = None,
        logger: Optional[SummaryWriter] = None,
        start_epoch: int = 0,
    ) -> None:
        """
        Orchestrate model training.

        Args:
            loss_fn: The loss function to use.
            n_epochs: Number of epochs to train for.
            opt: Optax optimizer to use for training.
            opt_state: Optional initial state for the optimizer. If not provided, the optimizer
                will be initialized with the parameters of the first model passed to `train()`.
            metrics: The metrics to use. Are calculated on the validation step. Defaults to `None`.
            logger: Tensorboard logger to use. Defaults to `None`.
        """
        self.loss_fn = loss_fn
        self.n_epochs = n_epochs
        self.opt = opt
        self._opt_state = opt_state
        self._best_model: Optional[TrainingModel] = None
        self._best_opt_state: Optional[optax.OptState] = None
        self._best_loss = jnp.asarray(jnp.inf)
        self._metrics = metrics if metrics is not None else {}
        self._logger = logger
        self._start_epoch = start_epoch

    @property
    def best_model(self) -> TrainingModel:
        if self._best_model is None:
            raise ValueError("No model has been trained yet.")
        return self._best_model

    @property
    def best_opt_state(self) -> optax.OptState:
        if self._best_opt_state is None:
            raise ValueError("No model has been trained yet.")
        return self._best_opt_state

    @best_opt_state.setter
    def best_opt_state(self, opt_state: optax.OptState) -> None:
        self._best_opt_state = opt_state

    @property
    def logger(self) -> Optional[SummaryWriter]:
        return self._logger

    def train(
        self,
        model: TrainingModel,
        dl: DataLoader,
        sampler: Sampler | None = None,
        val_dl: DataLoader | None = None,
        *,
        key: jnp.ndarray,
    ) -> TrainingModel:
        """
        Train the given model using the provided data loader and random key.

        Args:
            model: TrainingModel to train.
            dl: DataLoader providing batches of training data.
            key: JAX random key for training (keyword-only argument).

        Returns:
            The trained model `TrainingModel`.
        """
        trainable, static = model.partition()
        if self._opt_state is None:
            self._opt_state = self.opt.init(trainable)

        for _ in (
            pbar := tqdm(
                range(self._start_epoch, self._start_epoch + self.n_epochs),
                desc="Training",
                position=0,
                leave=False,
                initial=self._start_epoch,
            )
        ):
            epoch_loss = jnp.array(0.0)
            epoch_step_count = 0
            for x, y in dl:
                key, key_step, key_context = jax.random.split(key, 3)
                context_x = sampler(key_context) if sampler is not None else None
                loss, trainable, self._opt_state = train_step(
                    trainable,
                    static,
                    x,
                    y,
                    context_x,
                    self.loss_fn,
                    dl.batches_per_epoch,
                    self.opt,
                    self._opt_state,
                    key=key_step,
                )
                epoch_loss += loss
                epoch_step_count += 1

            epoch_loss /= epoch_step_count
            pbar.set_postfix({"loss": epoch_loss.item()})
            if self._logger is not None:
                self._logger.add_scalar("loss/train", epoch_loss.item(), pbar.n)

            if val_dl is not None:
                val_epoch_loss = jnp.array(0.0)
                epoch_step_count = 0
                val_metrics = {k: _m.empty() for k, _m in self._metrics.items()}
                for val_x, val_y in val_dl:
                    key, key_step, key_context = jax.random.split(key, 3)
                    val_context_x = sampler(key_context) if sampler is not None else None
                    val_loss, val_predf = val_step(
                        trainable,
                        static,
                        val_x,
                        val_y,
                        val_context_x,
                        self.loss_fn,
                        val_dl.batches_per_epoch,
                        key=key_step,
                    )
                    val_ydist = eqx.combine(trainable, static).predict_ydist(val_predf)
                    for k, metric in self._metrics.items():
                        _metric_state = metric.from_ydist(val_ydist, val_y)
                        val_metrics[k] = val_metrics[k].merge(_metric_state)
                    val_epoch_loss += val_loss
                    epoch_step_count += 1
                val_epoch_loss /= epoch_step_count
                val_metric_results = {k: _m.compute() for k, _m in val_metrics.items()}
                if self._logger is not None:
                    self._logger.add_scalar("loss/val", val_epoch_loss.item(), pbar.n)
                    for k, result in val_metric_results.items():
                        self._logger.add_scalar(f"metric/val_{k}", result.item(), pbar.n)
                pbar.set_postfix(
                    {
                        "val_loss": val_epoch_loss.item(),
                        **{f"val_{k}": m.item() for k, m in val_metric_results.items()},
                    }
                )

            _loss = val_epoch_loss if val_dl is not None else epoch_loss
            if _loss < self._best_loss:
                self._best_loss = _loss
                self._best_model = eqx.combine(trainable, static)
                self._best_opt_state = self._opt_state
                if self._logger is not None and val_dl is not None:
                    for k, result in val_metric_results.items():
                        self._logger.add_scalar(f"metric/best_val_{k}", result.item(), pbar.n)

        return eqx.combine(trainable, static)

    @staticmethod
    def eval(
        model: TrainingModel,
        dl: DataLoader,
        metrics: Mapping[str, type[Metric]],
        n_samples: int = 32,
        *,
        key: jnp.ndarray,
    ) -> Mapping[str, jnp.ndarray]:
        _metrics = {k: _m.empty() for k, _m in metrics.items()}
        for x, y in dl:
            key, subkey = jax.random.split(key)
            f_samples = model.net.predict_f_samples(x, n_samples, key=subkey)
            ydist = model.predict_ydist(f_samples)
            for k, metric in metrics.items():
                _metric_state = metric.from_ydist(ydist, y)
                _metrics[k] = _metrics[k].merge(_metric_state)

        return {k: _m.compute() for k, _m in _metrics.items()}
