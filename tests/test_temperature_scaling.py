"""Unit tests for post-hoc temperature scaling (D-14)."""

from __future__ import annotations

import numpy as np
import pytest

from hsi_robust.eval.calibration import expected_calibration_error
from hsi_robust.eval.temperature_scaling import (
    apply_temperature,
    evaluate_with_temperature,
    fit_temperature,
)


def _make_overconfident_logits(seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Construct logits whose softmax is *overconfident* w.r.t. true accuracy.

    The temperature that minimises NLL on this dataset is therefore > 1: the
    classifier should be *cooled down*.
    """
    rng = np.random.default_rng(seed)
    n, k = 400, 4
    labels = rng.integers(0, k, size=n)
    # Logits = 5 * one_hot(true) + small noise -> very peaked softmax,
    # but only 60 % of the predictions match the label so the classifier is
    # over-confident.
    correct = rng.random(n) < 0.6
    target_class = np.where(correct, labels, (labels + 1) % k)
    one_hot = np.eye(k)[target_class] * 5.0
    logits = one_hot + 0.1 * rng.standard_normal((n, k))
    return logits, labels


# --------------------------------------------------------------------------- #
# fit_temperature
# --------------------------------------------------------------------------- #


def test_fit_temperature_returns_positive_scalar() -> None:
    logits, labels = _make_overconfident_logits()
    t = fit_temperature(logits, labels)
    assert isinstance(t, float)
    assert t > 0


def test_fit_temperature_cools_down_overconfident_classifier() -> None:
    logits, labels = _make_overconfident_logits()
    t = fit_temperature(logits, labels)
    # An over-confident classifier needs T > 1 to be properly calibrated.
    assert t > 1.0


def test_fit_temperature_lowers_nll() -> None:
    logits, labels = _make_overconfident_logits()
    t = fit_temperature(logits, labels)

    def nll(scaled: np.ndarray) -> float:
        m = scaled.max(axis=1, keepdims=True)
        log_sum = m.squeeze(1) + np.log(np.exp(scaled - m).sum(axis=1))
        log_p_y = scaled[np.arange(scaled.shape[0]), labels] - log_sum
        return float(-log_p_y.mean())

    nll_raw = nll(logits)
    nll_scaled = nll(logits / t)
    assert nll_scaled <= nll_raw + 1e-9


def test_fit_temperature_validates_inputs() -> None:
    with pytest.raises(ValueError):
        fit_temperature(np.zeros(5), np.zeros(5, dtype=int))  # 1-D logits
    with pytest.raises(ValueError):
        fit_temperature(np.zeros((5, 3)), np.zeros(4, dtype=int))  # shape mismatch


# --------------------------------------------------------------------------- #
# apply_temperature
# --------------------------------------------------------------------------- #


def test_apply_temperature_returns_probabilities() -> None:
    logits, _ = _make_overconfident_logits()
    p = apply_temperature(logits, t=1.5)
    assert p.shape == logits.shape
    np.testing.assert_allclose(p.sum(axis=1), 1.0, atol=1e-8)
    assert (p >= 0).all() and (p <= 1).all()


def test_apply_temperature_higher_t_flattens_distribution() -> None:
    logits, _ = _make_overconfident_logits()
    low = apply_temperature(logits, t=0.5).max(axis=1)
    high = apply_temperature(logits, t=5.0).max(axis=1)
    assert (high <= low + 1e-9).all()


def test_apply_temperature_rejects_non_positive_t() -> None:
    logits, _ = _make_overconfident_logits()
    with pytest.raises(ValueError):
        apply_temperature(logits, t=0.0)
    with pytest.raises(ValueError):
        apply_temperature(logits, t=-1.0)


# --------------------------------------------------------------------------- #
# evaluate_with_temperature
# --------------------------------------------------------------------------- #


def test_evaluate_with_temperature_reduces_ece_on_overconfident_data() -> None:
    logits, labels = _make_overconfident_logits()
    res = evaluate_with_temperature(logits, labels, num_bins=15, calib_frac=0.2)

    assert res["n_calib"] + res["n_report"] == labels.shape[0]
    assert res["T"] > 1.0  # cools down
    # Temperature scaling must not *increase* ECE on the report slice -- if it
    # does, the optimiser was wrong about its convex problem.
    assert res["ECE_15_T"] <= res["ECE_15_raw"] + 1e-9
    # And the NLL must drop after scaling.
    assert res["NLL_T"] <= res["NLL_raw"] + 1e-9


def test_evaluate_with_temperature_matches_raw_ece_at_t_eq_1() -> None:
    """Sanity: raw ECE in the wrapper == direct ECE on the same probs."""
    logits, labels = _make_overconfident_logits()
    res = evaluate_with_temperature(logits, labels, num_bins=15, calib_frac=0.2)
    n_calib = res["n_calib"]
    rep_logits = logits[n_calib:]
    rep_labels = labels[n_calib:]
    raw_probs = apply_temperature(rep_logits, 1.0)
    direct_ece, _ = expected_calibration_error(raw_probs, rep_labels, num_bins=15)
    assert direct_ece == pytest.approx(res["ECE_15_raw"], abs=1e-9)


def test_evaluate_with_temperature_validates_calib_frac() -> None:
    logits, labels = _make_overconfident_logits()
    with pytest.raises(ValueError):
        evaluate_with_temperature(logits, labels, calib_frac=0.0)
    with pytest.raises(ValueError):
        evaluate_with_temperature(logits, labels, calib_frac=1.0)


def test_evaluate_with_temperature_is_deterministic() -> None:
    """The split is by index, so two calls return identical results."""
    logits, labels = _make_overconfident_logits()
    res_a = evaluate_with_temperature(logits, labels)
    res_b = evaluate_with_temperature(logits, labels)
    for key in ("T", "ECE_15_raw", "ECE_15_T", "NLL_raw", "NLL_T", "n_calib", "n_report"):
        assert res_a[key] == pytest.approx(res_b[key], abs=1e-12)
