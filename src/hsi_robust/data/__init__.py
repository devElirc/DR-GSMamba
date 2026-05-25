"""Data loading, splitting, and scene-frequency utilities for HSI datasets.

Public surface (to be populated in Phase 2B):

* ``load_hsi_cube``         - raw cube + ground-truth loader (Indian Pines, Pavia U,
  Salinas, Houston 2013).
* ``compute_scene_freq``    - scene-level class frequency :math:`\\pi_k`. This is the
  quantity consumed by :func:`hsi_robust.losses.cfa_gdro.cfa_gdro_loss`, per
  ``docs/math/cfa_gdro.md`` Section 1.
* ``StratifiedFixedPerClassSampler`` - leak-free, seed-controlled sampler producing
  exactly :math:`k` samples per class.
* ``HSIDataset``            - torch.utils.data.Dataset returning
  ``(raw_spectrum, pca_patch, label)`` tuples.
"""

from __future__ import annotations

__all__: list[str] = []
