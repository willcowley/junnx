"""
Implementation of common machine learning metrics. Follows the
[google-metrax](https://metrax.readthedocs.io/en/latest/) interface.
"""

import abc
from typing import TypeVar

import equinox as eqx
import jax
import jax.numpy as jnp
import tensorflow_probability.substrates.jax as tfp

M = TypeVar("M", bound="Metric")
_EPS = 1e-12


def _safe_divide(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    # divide x by y, but return 0 if y is 0 to avoid NaNs
    return jnp.where(y == 0, 0.0, x / y)


class Metric(eqx.Module):
    """Base class for a `Metric`."""

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
    """Mean Squared Error (MSE) metric."""

    @classmethod
    def from_samples(cls, predictions: jnp.ndarray, labels: jnp.ndarray) -> "MSE":
        return cls(
            total=jnp.sum((predictions - labels) ** 2), count=jnp.asarray(predictions.size)
        )


class RMSE(MSE):
    """Root Mean Squared Error (RMSE) metric."""

    def compute(self) -> jnp.ndarray:
        return jnp.sqrt(super().compute())


class NLL(_Average):
    """Negative Log Likelihood (NLL) metric."""

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
    """Accuracy classification metric."""

    @classmethod
    def from_samples(cls, predictions: jnp.ndarray, labels: jnp.ndarray) -> "Accuracy":
        pred_labels = jnp.argmax(predictions, axis=-1)
        correct = pred_labels == labels
        count = jnp.asarray(labels.size, dtype=jnp.int32)
        return cls(total=correct.sum(), count=count.sum())


class ECE(_ClassificationAverage):
    """The expected calibration error (ECE) classification metric."""

    nbins: int = eqx.field(static=True)
    """The number of confidence bins to use."""

    @classmethod
    def from_samples(
        cls, predictions: jnp.ndarray, labels: jnp.ndarray, nbins: int = 10
    ) -> "ECE":
        accuracy = jnp.argmax(predictions, axis=-1) == labels
        confidence = jnp.max(predictions, axis=-1)
        bins = jnp.linspace(0, 1, nbins + 1)
        bin_id = jnp.digitize(confidence, bins) - 1
        acc_binned = jnp.zeros(nbins).at[bin_id].add(accuracy)
        conf_binned = jnp.zeros(nbins).at[bin_id].add(confidence)

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
    """The Brier classification metric."""

    @classmethod
    def from_samples(cls, predictions: jnp.ndarray, labels: jnp.ndarray) -> "Brier":
        num_classes = predictions.shape[-1]
        one_hot_labels = jax.nn.one_hot(labels, num_classes, axis=-1)
        return cls(
            total=jnp.sum((predictions - one_hot_labels) ** 2),
            count=jnp.asarray(labels.size),
        )


class AUROC(Metric):
    """Area under Receiver Operating Curve (AUROC) metric."""

    tp: jnp.ndarray
    tn: jnp.ndarray
    fp: jnp.ndarray
    fn: jnp.ndarray
    n_thresholds: int = eqx.field(static=True)
    max_threshold: float = eqx.field(static=True)

    @classmethod
    def empty(cls, n_thresholds: int = 200, max_threshold: float = 1.0) -> "AUROC":
        return cls(
            tp=jnp.zeros(n_thresholds),
            tn=jnp.zeros(n_thresholds),
            fp=jnp.zeros(n_thresholds),
            fn=jnp.zeros(n_thresholds),
            n_thresholds=n_thresholds,
            max_threshold=max_threshold,
        )

    @classmethod
    def from_samples(
        cls,
        predictions: jnp.ndarray,
        labels: jnp.ndarray,
        n_thresholds: int = 200,
        max_threshold: float = 1.0,
    ) -> "AUROC":
        # predictions: [N,]
        # labels: [N,]
        thresholds = jnp.linspace(0, max_threshold, n_thresholds)  # [T,]
        pred_is_pos = jnp.greater(predictions, thresholds[..., None])  # [T, N]
        pred_is_neg = jnp.logical_not(pred_is_pos)

        label_is_pos = jnp.equal(labels, 1)  # [N,]
        label_is_neg = jnp.equal(labels, 0)  # [N,]

        tp = pred_is_pos * label_is_pos
        tn = pred_is_neg * label_is_neg
        fp = pred_is_pos * label_is_neg
        fn = pred_is_neg * label_is_pos

        return cls(
            tp=tp.sum(axis=-1),
            tn=tn.sum(axis=-1),
            fp=fp.sum(axis=-1),
            fn=fn.sum(axis=-1),
            n_thresholds=n_thresholds,
            max_threshold=max_threshold,
        )

    def merge(self, other: "AUROC") -> "AUROC":
        assert self.n_thresholds == other.n_thresholds

        tp = self.tp + other.tp
        tn = self.tn + other.tn
        fp = self.fp + other.fp
        fn = self.fn + other.fn

        m = eqx.tree_at(lambda m: m.tp, self, tp)
        m = eqx.tree_at(lambda m: m.tn, m, tn)
        m = eqx.tree_at(lambda m: m.fp, m, fp)
        m = eqx.tree_at(lambda m: m.fn, m, fn)

        return m

    def compute(self) -> jnp.ndarray:
        tp_rate = _safe_divide(self.tp, self.tp + self.fn)
        fp_rate = _safe_divide(self.fp, self.fp + self.tn)

        return -jnp.trapezoid(tp_rate, x=fp_rate)


class EntropyAUROC(AUROC):
    """AUROC metric that uses the entropy across multi-class predictions as a binary classifier."""

    @classmethod
    def from_ydist(
        cls,
        ydist: tfp.distributions.MixtureSameFamily,
        labels: jnp.ndarray,
        n_thresholds: int = 200,
    ) -> "EntropyAUROC":
        dist = ydist.components_distribution
        assert isinstance(dist, tfp.distributions.Categorical)  # [N, S, O]
        predictions = dist.probs_parameter().mean(axis=-2)  # [N, O]
        n_classes = predictions.shape[-1]
        max_threshold = jnp.log(n_classes).item()
        entropy = -jnp.sum(predictions * jnp.log(predictions + _EPS), axis=-1)  # [N,]
        return cls.from_samples(  # type: ignore[return-value]
            predictions=entropy,
            labels=labels,
            n_thresholds=n_thresholds,
            max_threshold=max_threshold,
        )
