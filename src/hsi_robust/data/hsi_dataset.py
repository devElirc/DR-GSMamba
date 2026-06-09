"""PyTorch :class:`Dataset` returning ``(raw_spectrum, pca_patch, label)`` tuples,
plus the :func:`build_split` orchestrator that wires every Phase 2B component
into a single, deterministic call.

The "build_split" function is the public entry point used by Phase 2E training
code and by the Phase 2B exit check.

Data flow (see ``EXPERIMENT_PLAN.md`` and ``docs/math/cfa_gdro.md`` Section 1):

    cube                         (H, W, B) float32, raw radiance
    gt                           (H, W) int64, raw scene labels
        |
        v  flatten_labeled_pixels
    labels_flat (N,), positions (N, 2)   -- 0-indexed labels, (i, j)
        |
        v  stratified_fixed_per_class_split (seed)
    train_idx, test_idx
        |
        v  PerBandStandardize.fit(train spectra), then transform(whole cube)
    cube_std (H, W, B)
        |
        v  PCAReducer.fit(train spectra std), then transform(cube_std)
    cube_pca (H, W, B')
        |
        v  reflection padding
    cube_std_padded, cube_pca_padded
        |
        v  HSIDataset
    (raw_spectrum, pca_patch, label) tuples

Determinism: given the same ``cube``, ``gt``, ``samples_per_class`` and
``seed``, two calls to :func:`build_split` return train labels, positions, and
the first ``HSIDataset`` item bit-identical.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from hsi_robust.data.io import load_hsi_cube
from hsi_robust.data.sampler import stratified_fixed_per_class_split
from hsi_robust.data.scene_freq import compute_scene_freq, flatten_labeled_pixels
from hsi_robust.data.transforms import (
    PCAReducer,
    PerBandStandardize,
    extract_patch,
    pad_cube_for_patches,
)


class HSIDataset(Dataset):
    """Yields ``(raw_spectrum, pca_patch, label)`` tuples for one split.

    The dataset stores pre-padded standardised and PCA-reduced cubes, so
    ``__getitem__`` is just an array slice and a tensor copy -- a constant-time
    operation per item.
    """

    def __init__(
        self,
        *,
        cube_std_padded: np.ndarray,
        cube_pca_padded: np.ndarray,
        positions: np.ndarray,
        labels: np.ndarray,
        patch_size: int,
    ) -> None:
        if cube_std_padded.ndim != 3:
            raise ValueError(f"cube_std_padded must be 3D; got {cube_std_padded.shape}")
        if cube_pca_padded.ndim != 3:
            raise ValueError(f"cube_pca_padded must be 3D; got {cube_pca_padded.shape}")
        if positions.shape[0] != labels.shape[0]:
            raise ValueError(
                f"positions and labels disagree on length: "
                f"{positions.shape[0]} vs {labels.shape[0]}"
            )

        self._cube_std_padded = cube_std_padded
        self._cube_pca_padded = cube_pca_padded
        self._positions = positions.astype(np.int64, copy=False)
        self._labels = labels.astype(np.int64, copy=False)
        self.patch_size = int(patch_size)
        self._half = self.patch_size // 2

    def __len__(self) -> int:
        return int(self._positions.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        i, j = int(self._positions[index, 0]), int(self._positions[index, 1])
        label = int(self._labels[index])

        patch = extract_patch(self._cube_pca_padded, i, j, self.patch_size)
        # Patch is (P, P, B'); convert to (B', P, P) channels-first for CNN consumption.
        patch_tensor = torch.from_numpy(np.ascontiguousarray(patch.transpose(2, 0, 1))).float()

        # Centre pixel of the padded cube == original pixel (i, j).
        spectrum = self._cube_std_padded[i + self._half, j + self._half, :]
        spectrum_tensor = torch.from_numpy(np.ascontiguousarray(spectrum)).float()

        return spectrum_tensor, patch_tensor, label

    @property
    def positions(self) -> np.ndarray:
        return self._positions

    @property
    def labels(self) -> np.ndarray:
        return self._labels


@dataclass
class SplitArtifacts:
    """All artefacts produced by :func:`build_split`.

    The fields beyond ``train_dataset`` / ``test_dataset`` are exposed so that
    unit tests can verify, e.g., that the PCA was fit on train spectra only.
    """

    train_dataset: HSIDataset
    test_dataset: HSIDataset
    scene_freq: torch.Tensor
    num_classes: int
    num_bands: int
    pca_components: int
    patch_size: int
    standardizer: PerBandStandardize
    pca: PCAReducer
    train_positions: np.ndarray
    train_labels: np.ndarray
    test_positions: np.ndarray
    test_labels: np.ndarray
    cube_std_padded: np.ndarray
    cube_pca_padded: np.ndarray

    @property
    def per_class_train_count(self) -> torch.Tensor:
        counts = np.bincount(self.train_labels, minlength=self.num_classes).astype(np.int64)
        return torch.from_numpy(counts)

    @property
    def per_class_test_count(self) -> torch.Tensor:
        counts = np.bincount(self.test_labels, minlength=self.num_classes).astype(np.int64)
        return torch.from_numpy(counts)


def build_split_from_arrays(
    *,
    cube: np.ndarray,
    gt: np.ndarray,
    num_classes: int,
    ignore_label: int,
    patch_size: int,
    pca_components: int,
    samples_per_class: int,
    seed: int,
) -> SplitArtifacts:
    """Deterministic train/test split + transforms on raw arrays.

    This is the low-level function used by both :func:`build_split` (which
    loads from a config) and the unit tests (which inject synthetic arrays).
    """
    if patch_size <= 0 or patch_size % 2 == 0:
        raise ValueError(f"patch_size must be positive and odd; got {patch_size}")
    if pca_components <= 0 or pca_components > cube.shape[2]:
        raise ValueError(
            f"pca_components must be in (0, {cube.shape[2]}]; got {pca_components}"
        )

    labels_flat, positions = flatten_labeled_pixels(
        gt, num_classes=num_classes, ignore_label=ignore_label
    )
    scene_freq = compute_scene_freq(labels_flat, num_classes=num_classes)

    train_idx, test_idx = stratified_fixed_per_class_split(
        labels_flat,
        samples_per_class=samples_per_class,
        seed=seed,
        num_classes=num_classes,
    )

    train_positions = positions[train_idx]
    train_labels = labels_flat[train_idx]
    test_positions = positions[test_idx]
    test_labels = labels_flat[test_idx]

    # Fit standardizer on TRAIN spectra (raw radiance) only.
    train_spectra_raw = cube[train_positions[:, 0], train_positions[:, 1], :]
    standardizer = PerBandStandardize().fit(train_spectra_raw)
    cube_std = standardizer.transform(cube)

    # Fit PCA on TRAIN spectra (standardised) only.
    train_spectra_std = cube_std[train_positions[:, 0], train_positions[:, 1], :]
    pca = PCAReducer(n_components=pca_components).fit(train_spectra_std)
    cube_pca = pca.transform(cube_std)

    # Pad once; the Dataset will slice into these views.
    cube_std_padded = pad_cube_for_patches(cube_std, patch_size)
    cube_pca_padded = pad_cube_for_patches(cube_pca, patch_size)

    train_dataset = HSIDataset(
        cube_std_padded=cube_std_padded,
        cube_pca_padded=cube_pca_padded,
        positions=train_positions,
        labels=train_labels,
        patch_size=patch_size,
    )
    test_dataset = HSIDataset(
        cube_std_padded=cube_std_padded,
        cube_pca_padded=cube_pca_padded,
        positions=test_positions,
        labels=test_labels,
        patch_size=patch_size,
    )

    return SplitArtifacts(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        scene_freq=scene_freq,
        num_classes=num_classes,
        num_bands=int(cube.shape[2]),
        pca_components=int(pca_components),
        patch_size=int(patch_size),
        standardizer=standardizer,
        pca=pca,
        train_positions=train_positions,
        train_labels=train_labels,
        test_positions=test_positions,
        test_labels=test_labels,
        cube_std_padded=cube_std_padded,
        cube_pca_padded=cube_pca_padded,
    )


def build_split(config: dict[str, Any], seed: int) -> SplitArtifacts:
    """Load a dataset and build the train/test split per ``config`` and ``seed``.

    The ``config`` is a dataset config dict as returned by
    :func:`hsi_robust.utils.config.load_yaml`. ``samples_per_class`` can be
    overridden by mutating ``config["samples_per_class"]`` before calling.

    Two invocations with the same ``config`` and ``seed`` produce bit-identical
    train labels / positions / first dataset item -- this is the
    exit criterion for Phase 2B.
    """
    cube, gt = load_hsi_cube(config)
    return build_split_from_arrays(
        cube=cube,
        gt=gt,
        num_classes=int(config["num_classes"]),
        ignore_label=int(config["ignore_label"]),
        patch_size=int(config["patch_size"]),
        pca_components=int(config["pca_components"]),
        samples_per_class=int(config["samples_per_class"]),
        seed=int(seed),
    )
