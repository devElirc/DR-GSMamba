"""Evaluation utilities (Phase 2E).

Public surface:

* :func:`compute_metrics`, :func:`per_class_accuracy`, :func:`overall_accuracy`,
  :func:`average_accuracy`, :func:`kappa_score`, :func:`worst_class_accuracy`,
  :func:`coefficient_of_variation`, :func:`rare_class_accuracy` -- core metrics.
* :func:`expected_calibration_error`, :class:`ReliabilityBins` -- calibration.
* :func:`classification_map`, :func:`error_map`, :func:`vacuity_map`,
  :func:`aleatoric_map` -- qualitative maps.
"""

from __future__ import annotations

from hsi_robust.eval.calibration import ReliabilityBins, expected_calibration_error
from hsi_robust.eval.metrics import (
    average_accuracy,
    coefficient_of_variation,
    compute_metrics,
    kappa_score,
    overall_accuracy,
    per_class_accuracy,
    rare_class_accuracy,
    worst_class_accuracy,
)
from hsi_robust.eval.qualitative import (
    aleatoric_map,
    classification_map,
    error_map,
    vacuity_map,
)
from hsi_robust.eval.temperature_scaling import (
    apply_temperature,
    evaluate_with_temperature,
    fit_temperature,
)

__all__ = [
    "ReliabilityBins",
    "aleatoric_map",
    "apply_temperature",
    "average_accuracy",
    "classification_map",
    "coefficient_of_variation",
    "compute_metrics",
    "error_map",
    "evaluate_with_temperature",
    "expected_calibration_error",
    "fit_temperature",
    "kappa_score",
    "overall_accuracy",
    "per_class_accuracy",
    "rare_class_accuracy",
    "vacuity_map",
    "worst_class_accuracy",
]
