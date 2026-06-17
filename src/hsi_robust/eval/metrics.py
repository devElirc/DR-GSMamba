"""HSI classification metrics: OA, AA, kappa, per-class, worst-class, CoV, rare-class.

All functions accept NumPy arrays so they can be called from notebooks, scripts,
and the trainer without a torch dependency. Labels are expected in ``{0, ..., K-1}``
(i.e. 0-indexed, matching :mod:`hsi_robust.data.scene_freq.flatten_labeled_pixels`).

Public surface:

* :func:`per_class_accuracy`
* :func:`overall_accuracy`
* :func:`average_accuracy`
* :func:`kappa_score`
* :func:`worst_class_accuracy`
* :func:`coefficient_of_variation`
* :func:`rare_class_accuracy`
* :func:`compute_metrics`  -- aggregator returning all of the above in a dict.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from sklearn.metrics import cohen_kappa_score, confusion_matrix


def per_class_accuracy(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    """Return per-class accuracy as a length-``K`` array (NaN if class absent)."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    with np.errstate(divide="ignore", invalid="ignore"):
        per = cm.diagonal() / cm.sum(axis=1)
    return per


def overall_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    if y_true.size == 0:
        return float("nan")
    return float((y_true == y_pred).mean())


def average_accuracy(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> float:
    """Mean of per-class accuracies, ignoring classes absent from y_true."""
    per = per_class_accuracy(y_true, y_pred, num_classes)
    mask = np.isfinite(per)
    return float(per[mask].mean()) if mask.any() else float("nan")


def kappa_score(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> float:
    """Cohen's kappa using all K classes (no auto-detection)."""
    return float(cohen_kappa_score(y_true, y_pred, labels=list(range(num_classes))))


def worst_class_accuracy(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> float:
    per = per_class_accuracy(y_true, y_pred, num_classes)
    mask = np.isfinite(per)
    return float(per[mask].min()) if mask.any() else float("nan")


def coefficient_of_variation(values: Iterable[float]) -> float:
    """``std / mean`` for a sequence of metric values (used in cross-seed reports)."""
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return float("nan")
    mu = arr.mean()
    if mu == 0:
        return float("nan")
    return float(arr.std(ddof=0) / mu)


def rare_class_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int,
    scene_freq: np.ndarray,
    rare_fraction: float = 0.25,
) -> float:
    """Average per-class accuracy over the rarest ``rare_fraction`` of classes by ``scene_freq``."""
    per = per_class_accuracy(y_true, y_pred, num_classes)
    order = np.argsort(scene_freq)  # ascending: rarest first
    n_rare = max(1, round(rare_fraction * num_classes))
    rare_idx = order[:n_rare]
    rare_per = per[rare_idx]
    mask = np.isfinite(rare_per)
    return float(rare_per[mask].mean()) if mask.any() else float("nan")


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int,
    scene_freq: np.ndarray | None = None,
    rare_fraction: float = 0.25,
) -> dict[str, float | list[float]]:
    """Return the full reliability metric bundle as a JSON-friendly dict."""
    per = per_class_accuracy(y_true, y_pred, num_classes)
    out: dict[str, float | list[float]] = {
        "OA": overall_accuracy(y_true, y_pred),
        "AA": average_accuracy(y_true, y_pred, num_classes),
        "kappa": kappa_score(y_true, y_pred, num_classes),
        "worst_class": worst_class_accuracy(y_true, y_pred, num_classes),
        "per_class": [float(x) if np.isfinite(x) else float("nan") for x in per],
    }
    if scene_freq is not None:
        out["rare_class"] = rare_class_accuracy(
            y_true, y_pred, num_classes, scene_freq, rare_fraction=rare_fraction
        )
    return out
