"""Calibration metrics and reliability diagrams.

Implements ECE with 15 equal-width confidence bins by default (the standard
HSI-paper choice), plus a helper that emits reliability-diagram data (no
plotting -- a downstream notebook handles matplotlib).
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np


class ReliabilityBins(NamedTuple):
    """Per-bin statistics used to draw a reliability diagram."""

    bin_lower: np.ndarray  # (B,)
    bin_upper: np.ndarray  # (B,)
    bin_count: np.ndarray  # (B,) integer
    bin_confidence: np.ndarray  # (B,) mean predicted prob inside bin
    bin_accuracy: np.ndarray  # (B,) empirical accuracy inside bin


def expected_calibration_error(
    probs: np.ndarray,
    y_true: np.ndarray,
    num_bins: int = 15,
) -> tuple[float, ReliabilityBins]:
    """Compute ECE-N + the bin breakdown.

    Parameters
    ----------
    probs:
        ``(N, K)`` predicted class probabilities (rows sum to 1).
    y_true:
        ``(N,)`` integer labels in ``[0, K)``.
    num_bins:
        Number of equal-width confidence bins. Default 15 per HSI convention.

    Returns
    -------
    ece:
        ``sum_b (n_b / N) * |conf_b - acc_b|``.
    bins:
        A :class:`ReliabilityBins` named tuple useful for diagrams.
    """
    probs = np.asarray(probs)
    y_true = np.asarray(y_true).astype(int)
    if probs.ndim != 2:
        raise ValueError(f"probs must be 2-D; got {probs.shape}")
    if y_true.shape != probs.shape[:1]:
        raise ValueError(f"y_true shape {y_true.shape} incompatible with probs {probs.shape}")
    if num_bins <= 1:
        raise ValueError("num_bins must be >= 2")
    n = probs.shape[0]

    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == y_true).astype(float)

    bin_edges = np.linspace(0.0, 1.0, num_bins + 1)
    bin_lower = bin_edges[:-1]
    bin_upper = bin_edges[1:]

    counts = np.zeros(num_bins, dtype=np.int64)
    confidences = np.zeros(num_bins, dtype=np.float64)
    accuracies = np.zeros(num_bins, dtype=np.float64)

    ece = 0.0
    for b in range(num_bins):
        lo, hi = bin_edges[b], bin_edges[b + 1]
        # Last bin is closed on the right to include conf == 1.0.
        if b == num_bins - 1:
            mask = (conf >= lo) & (conf <= hi)
        else:
            mask = (conf >= lo) & (conf < hi)
        n_b = int(mask.sum())
        counts[b] = n_b
        if n_b > 0:
            conf_b = float(conf[mask].mean())
            acc_b = float(correct[mask].mean())
            confidences[b] = conf_b
            accuracies[b] = acc_b
            ece += (n_b / n) * abs(conf_b - acc_b)
        else:
            confidences[b] = float("nan")
            accuracies[b] = float("nan")

    bins = ReliabilityBins(
        bin_lower=bin_lower,
        bin_upper=bin_upper,
        bin_count=counts,
        bin_confidence=confidences,
        bin_accuracy=accuracies,
    )
    return float(ece), bins
