import abc
from typing import TypeVar

import equinox as eqx
import jax
import jax.numpy as jnp
import tensorflow_probability.substrates.jax as tfp

M = TypeVar("M", bound="Metric")


def _safe_divide(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    # divide x by y, but return 0 if y is 0 to avoid NaNs
    return jnp.where(y == 0, 0.0, x / y)


class Metric(eqx.Module):

    @classmethod
    @abc.abstractmethod
    def from_samples(cls: type[M], *args, **kwargs) -> M: ...

    @abc.abstractmethod
    def merge(self: M, other: M) -> M: ...

    @abc.abstractmethod
    def compute(self) -> jnp.ndarray: ...

    @classmethod
    @abc.abstractmethod
    def empty(cls: type[M], *args, **kwargs) -> M: ...

    @classmethod
    def from_ydist(
        cls: type[M], ydist: tfp.distributions.Distribution, labels: jnp.ndarray
    ) -> M:
        predictions = ydist.mean()
        return cls.from_samples(predictions=predictions, labels=labels)


class _Average(Metric):

    total: jnp.ndarray
    count: jnp.ndarray

    def compute(self) -> jnp.ndarray:
        return _safe_divide(self.total, self.count)

    def merge(self, other: "_Average") -> "_Average":
        m = eqx.tree_at(lambda m: m.total, self, self.total + other.total)
        m = eqx.tree_at(lambda m: m.count, m, self.count + other.count)
        return m

    @classmethod
    def empty(cls) -> "_Average":
        return cls(total=jnp.array(0.0), count=jnp.array(0))


class MSE(_Average):

    @classmethod
    def from_samples(cls, predictions: jnp.ndarray, labels: jnp.ndarray) -> "MSE":
        return cls(
            total=jnp.sum((predictions - labels) ** 2), count=jnp.asarray(predictions.size)
        )


class RMSE(MSE):

    def compute(self) -> jnp.ndarray:
        return jnp.sqrt(super().compute())


class NLL(_Average):

    @classmethod
    def from_samples(cls, predictions: jnp.ndarray, labels: jnp.ndarray) -> "NLL":
        raise NotImplementedError

    @classmethod
    def from_ydist(cls, ydist: tfp.distributions.Distribution, labels: jnp.ndarray) -> "NLL":
        nll = -ydist.log_prob(labels)
        return cls(total=nll.sum(), count=jnp.asarray(labels.size))


class _ClassificationAverage(_Average):

    @classmethod
    def from_ydist(
        cls, ydist: tfp.distributions.MixtureSameFamily, labels: jnp.ndarray
    ) -> "_ClassificationAverage":
        dist = ydist.components_distribution
        assert isinstance(dist, (tfp.distributions.Categorical, tfp.distributions.Bernoulli))
        predictions = dist.probs_parameter().mean(axis=-2)  # [N, O]
        return cls.from_samples(predictions=predictions, labels=labels)


class Accuracy(_ClassificationAverage):

    @classmethod
    def from_samples(cls, predictions: jnp.ndarray, labels: jnp.ndarray) -> "Accuracy":
        pred_labels = jnp.argmax(predictions, axis=-1)
        correct = pred_labels == labels
        count = jnp.asarray(labels.size, dtype=jnp.int32)
        return cls(total=correct.sum(), count=count.sum())


class ECE(_ClassificationAverage):

    nbins: int = eqx.field(static=True)

    @classmethod
    def from_samples(
        cls, predictions: jnp.ndarray, labels: jnp.ndarray, nbins: int = 10
    ) -> "ECE":
        accuracy = jnp.argmax(predictions, axis=-1) == labels
        confidence = jnp.max(predictions, axis=-1)
        bins = jnp.linspace(0, 1, nbins + 1)
        acc_binned, _ = jnp.histogram(confidence, bins=bins, weights=accuracy)
        conf_binned, _ = jnp.histogram(confidence, bins=bins, weights=confidence)

        return cls(
            total=acc_binned - conf_binned,
            count=jnp.asarray(labels.size, dtype=jnp.int32),
            nbins=nbins,
        )

    def compute(self) -> jnp.ndarray:
        return _safe_divide(jnp.abs(self.total).sum(), self.count)

    @classmethod
    def empty(cls, nbins: int = 10) -> "ECE":
        return cls(total=jnp.zeros(nbins), count=jnp.array(0), nbins=nbins)


class Brier(_ClassificationAverage):

    @classmethod
    def from_samples(cls, predictions: jnp.ndarray, labels: jnp.ndarray) -> "Brier":
        num_classes = predictions.shape[-1]
        one_hot_labels = jax.nn.one_hot(labels, num_classes, axis=-1)
        return cls(
            total=jnp.sum((predictions - one_hot_labels) ** 2),
            count=jnp.asarray(labels.size),
        )
