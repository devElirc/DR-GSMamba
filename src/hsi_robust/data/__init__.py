"""Data loading, splitting, and scene-frequency utilities for HSI datasets.

Phase 2B public surface
-----------------------

* :func:`load_hsi_cube` -- raw cube + ground-truth loader for ``.mat`` / ``.npy``.
* :func:`flatten_labeled_pixels` -- ``(H, W)`` gt -> 0-indexed flat labels + (i, j).
* :func:`compute_scene_freq` -- scene-level class frequency :math:`\\pi_k`
  consumed by :func:`hsi_robust.losses.cfa_gdro.cfa_gdro_loss`
  (see ``docs/math/cfa_gdro.md`` Section 1).
* :func:`stratified_fixed_per_class_split` -- leak-free, seed-controlled
  sampler producing exactly :math:`k` samples per class.
* :class:`PerBandStandardize`, :class:`PCAReducer` -- train-only fit transforms.
* :func:`pad_cube_for_patches`, :func:`extract_patch` -- patch extraction.
* :class:`HSIDataset` -- ``(raw_spectrum, pca_patch, label)`` tuples.
* :func:`build_split`, :func:`build_split_from_arrays`, :class:`SplitArtifacts`
  -- end-to-end orchestrator that wires every component together.
"""

from __future__ import annotations

from hsi_robust.data.hsi_dataset import (
    HSIDataset,
    SplitArtifacts,
    build_split,
    build_split_from_arrays,
)
from hsi_robust.data.io import load_hsi_cube
from hsi_robust.data.sampler import stratified_fixed_per_class_split
from hsi_robust.data.scene_freq import compute_scene_freq, flatten_labeled_pixels
from hsi_robust.data.transforms import (
    PCAReducer,
    PerBandStandardize,
    extract_patch,
    pad_cube_for_patches,
)

__all__ = [
    "HSIDataset",
    "PCAReducer",
    "PerBandStandardize",
    "SplitArtifacts",
    "build_split",
    "build_split_from_arrays",
    "compute_scene_freq",
    "extract_patch",
    "flatten_labeled_pixels",
    "load_hsi_cube",
    "pad_cube_for_patches",
    "stratified_fixed_per_class_split",
]
