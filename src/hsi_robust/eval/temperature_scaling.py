"""Post-hoc temperature scaling (Guo et al., ICML 2017).

This module attacks the high ECE the smoke run reported (0.473) by fitting a
single positive scalar :math:`T` to minimise NLL on a held-out *calibration*
slice of the evaluation set, then reporting the temperature-scaled ECE on the
remaining slice.

Why a separate module (rather than baking it into ``Trainer.evaluate``):

* it stays loss-agnostic, so the CE / focal / sample-CVaR / Sagawa group-DRO /
  CFA-GDRO rows of the internal ablation all use the same calibration tool;
* it works on logits, not probabilities, which is the strictly correct input
  per the original Guo et al. derivation;
* the split (calibration / report) is deterministic by index so the ECE_T it
  produces is bit-reproducible.

The fit uses Brent's method on ``log T`` so :math:`T > 0` holds without
constraints. The optimisation is convex in :math:`T` for the NLL of a softmax
classifier, so the bracket search is sufficient.

Reference:
    Guo, Pleiss, Sun, Weinberger.
    "On Calibration of Modern Neural Networks." ICML 2017.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.optimize import minimize_scalar

from hsi_robust.eval.calibration import expected_calibration_error


def _nll_from_logits(logits: np.ndarray, labels: np.ndarray) -> float:
    """Mean NLL of a softmax classifier with the given logits."""
    m = logits.max(axis=1, keepdims=True)
    log_sum = m.squeeze(1) + np.log(np.exp(logits - m).sum(axis=1))
    n = labels.shape[0]
    log_p_y = logits[np.arange(n), labels] - log_sum
    return float(-log_p_y.mean())


def fit_temperature(
    logits: np.ndarray,
    labels: np.ndarray,
    *,
    log_t_bounds: tuple[float, float] = (-4.0, 5.0),
    max_iter: int = 100,
) -> float:
    """Return the optimal :math:`T > 0` that minimises NLL on ``(logits, labels)``.

    Parameters
    ----------
    logits:
        ``(N, K)`` pre-softmax logits.
    labels:
        ``(N,)`` integer labels in ``[0, K)``.
    log_t_bounds:
        ``(lo, hi)`` bounds on ``log T``. Defaults cover
        :math:`T \\in [e^{-4}, e^{5}] \\approx [0.018, 148]`, which is wider
        than any temperature reported in the HSI calibration literature.
        We use bounded (rather than bracketed) optimisation so an
        out-of-bracket minimum does not crash the run.
    max_iter:
        Maximum number of optimiser iterations.
    """
    logits = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    if logits.ndim != 2:
        raise ValueError(f"logits must be 2-D; got {logits.shape}")
    if labels.shape != logits.shape[:1]:
        raise ValueError(f"labels shape {labels.shape} incompatible with logits {logits.shape}")

    def nll(log_t: float) -> float:
        t = math.exp(log_t)
        return _nll_from_logits(logits / t, labels)

    lo, hi = log_t_bounds
    if lo >= hi:
        raise ValueError(f"log_t_bounds must be strictly increasing; got {log_t_bounds}")

    result = minimize_scalar(
        nll,
        bounds=(lo, hi),
        method="bounded",
        options={"maxiter": max_iter, "xatol": 1e-6},
    )
    if not result.success:
        # Fall back gracefully to T=1 (no scaling) rather than crashing the
        # whole ablation run on a degenerate batch.
        return 1.0
    return float(math.exp(result.x))


def apply_temperature(logits: np.ndarray, t: float) -> np.ndarray:
    """Return ``softmax(logits / t)``. Numerically stable via max-shift."""
    if t <= 0:
        raise ValueError(f"temperature must be positive; got {t}")
    scaled = np.asarray(logits, dtype=np.float64) / float(t)
    scaled -= scaled.max(axis=1, keepdims=True)
    exp = np.exp(scaled)
    return exp / exp.sum(axis=1, keepdims=True)


def evaluate_with_temperature(
    logits: np.ndarray,
    labels: np.ndarray,
    *,
    num_bins: int = 15,
    calib_frac: float = 0.2,
) -> dict[str, float]:
    """Fit T on a calibration slice and report raw + temp-scaled ECE on the rest.

    The split is deterministic by index (first ``calib_frac`` of the rows go to
    fitting, the rest to reporting) so the result is bit-reproducible. This
    matches the Guo et al. protocol when no separate validation set is
    available -- a slice of the *test* set is used to fit T, and the remaining
    slice is used to report the temperature-scaled metric.

    Returns a dict with::

        {"T": optimal temperature,
         "ECE_15_raw": ECE on the report slice with T = 1,
         "ECE_15_T":   ECE on the report slice after temperature scaling,
         "NLL_raw":    NLL on the report slice with T = 1,
         "NLL_T":      NLL on the report slice after temperature scaling,
         "n_calib":    size of the calibration slice,
         "n_report":   size of the report slice}
    """
    logits = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    if not (0.0 < calib_frac < 1.0):
        raise ValueError(f"calib_frac must lie in (0, 1); got {calib_frac}")
    n = logits.shape[0]
    n_calib = max(1, round(calib_frac * n))
    if n_calib >= n:
        n_calib = n - 1  # leave at least one sample to report on
    cal_logits, cal_labels = logits[:n_calib], labels[:n_calib]
    rep_logits, rep_labels = logits[n_calib:], labels[n_calib:]

    t = fit_temperature(cal_logits, cal_labels)

    raw_probs = apply_temperature(rep_logits, 1.0)
    scaled_probs = apply_temperature(rep_logits, t)
    ece_raw, _ = expected_calibration_error(raw_probs, rep_labels, num_bins=num_bins)
    ece_t, _ = expected_calibration_error(scaled_probs, rep_labels, num_bins=num_bins)
    return {
        "T": float(t),
        "ECE_15_raw": float(ece_raw),
        "ECE_15_T": float(ece_t),
        "NLL_raw": _nll_from_logits(rep_logits, rep_labels),
        "NLL_T": _nll_from_logits(rep_logits / t, rep_labels),
        "n_calib": int(n_calib),
        "n_report": int(rep_logits.shape[0]),
    }
