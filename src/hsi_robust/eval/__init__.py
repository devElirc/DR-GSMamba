"""Evaluation, calibration, and qualitative-map utilities.

Public surface (to be populated in Phase 2E and Phase 7):

* ``compute_metrics``         - OA, AA, kappa, macro-F1, per-class, worst-class,
  rare-class, CoV, worst-split.
* ``compute_ece``             - 15-bin expected calibration error.
* ``reliability_diagram``     - matplotlib figure for the appendix.
* ``classification_maps``     - prediction / error / vacuity / aleatoric maps.
"""

from __future__ import annotations

__all__: list[str] = []
