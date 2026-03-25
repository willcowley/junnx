from typing import Optional

import equinox as eqx
import jax
import optax
from jax import numpy as jnp
from tqdm import tqdm

from junnx.datasets import DataLoader
from junnx.elbo import train_step
from junnx.train import TrainingModel


class Trainer:

    def __init__(
        self,
        n_samples_nll: int,
        n_samples_kl: int,
        n_epochs: int,
        n_data: int,
        opt: optax.GradientTransformation,
        opt_state: Optional[optax.OptState] = None,
    ) -> None:
        """
        Orchestrate model training.

        Args:
            n_samples_nll: Number of model realisations to use when estimating the negative
                log-likelihood term of the ELBO.
            n_samples_kl: Number of model realisations to use when estimating the KL divergence
                term of the ELBO.
            n_epochs: Number of epochs to train for.
            n_data: Number of data points in the training dataset.
            opt: Optax optimizer to use for training.
            opt_state: Optional initial state for the optimizer. If not provided, the optimizer
                will be initialized with the parameters of the first model passed to `train()`.
        """
        self.n_samples_nll = n_samples_nll
        self.n_samples_kl = n_samples_kl
        self.n_epochs = n_epochs
        self.n_data = n_data
        self.opt = opt
        self._opt_state = opt_state
        self._best_model: Optional[TrainingModel] = None
        self._best_opt_state: Optional[optax.OptState] = None
        self._best_loss = jnp.asarray(jnp.inf)

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

    def train(
        self,
        model: TrainingModel,
        dl: DataLoader,
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
        n_batches_per_epoch = self.n_data // dl.batch_size

        for _ in (
            pbar := tqdm(range(self.n_epochs), desc="Training", position=0, leave=False)
        ):
            epoch_loss = jnp.array(0.0)
            epoch_step_count = 0
            for x, y in dl:
                key, subkey = jax.random.split(key)
                loss, trainable, self._opt_state = train_step(
                    trainable,
                    static,
                    x,
                    y,
                    self.n_samples_nll,
                    self.n_samples_kl,
                    n_batches_per_epoch,
                    self.opt,
                    self._opt_state,
                    key=subkey,
                )
                epoch_loss += loss
                epoch_step_count += 1

            epoch_loss /= epoch_step_count
            pbar.set_postfix({"loss": epoch_loss.item()})

            if epoch_loss < self._best_loss:
                self._best_loss = epoch_loss
                self._best_model = eqx.combine(trainable, static)
                self._best_opt_state = self._opt_state

        return eqx.combine(trainable, static)
