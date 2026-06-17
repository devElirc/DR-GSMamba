"""Phase 2B data-module tests.

Coverage (per roadmap M2B.7):

    * ``compute_scene_freq`` sums to 1 and matches counts.
    * ``stratified_fixed_per_class_split`` returns exactly ``samples_per_class``
      per class, and is seed-deterministic.
    * ``PerBandStandardize`` and ``PCAReducer`` are fit on train pixels only
      (leak-free).
    * ``HSIDataset.__getitem__`` returns the expected shapes and dtypes.
    * End-to-end determinism: two ``build_split_from_arrays`` calls with the
      same inputs are bit-identical.
    * Real Indian Pines data exit check (skipped if data is unavailable):
      ``build_split`` runs and is reproducible.

Synthetic data is used for unit tests so the suite stays fast and works on
machines without the .mat files present.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pytest
import torch

from hsi_robust.data import (
    PCAReducer,
    PerBandStandardize,
    build_split,
    build_split_from_arrays,
    compute_scene_freq,
    extract_patch,
    flatten_labeled_pixels,
    pad_cube_for_patches,
    stratified_fixed_per_class_split,
)
from hsi_robust.utils import load_yaml

# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent


def _make_synthetic_scene(
    *,
    h: int = 40,
    w: int = 40,
    bands: int = 30,
    num_classes: int = 4,
    class_pixel_targets: tuple[int, ...] = (300, 200, 100, 50),
    seed: int = 123,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a deterministic synthetic ``(cube, gt)`` pair.

    ``gt`` uses the 1-indexed HSI convention (labels in ``{1, ..., K}``;
    ``0`` is the ignore label). Per-class pixel counts are exactly the
    ``class_pixel_targets``; the remaining pixels are filled with ``0``.
    """
    assert len(class_pixel_targets) == num_classes
    assert sum(class_pixel_targets) <= h * w

    rng = np.random.default_rng(seed)
    cube = rng.standard_normal((h, w, bands)).astype(np.float32)

    gt_flat = np.zeros(h * w, dtype=np.int64)
    positions = rng.permutation(h * w)
    cursor = 0
    for k, n in enumerate(class_pixel_targets, start=1):
        gt_flat[positions[cursor : cursor + n]] = k
        cursor += n
    gt = gt_flat.reshape(h, w)
    return cube, gt


# ---------------------------------------------------------------------------
# scene_freq
# ---------------------------------------------------------------------------


def test_flatten_labeled_pixels_remaps_to_zero_indexed() -> None:
    gt = np.array([[0, 1, 2], [3, 0, 1]], dtype=np.int64)
    labels, positions = flatten_labeled_pixels(gt, num_classes=3, ignore_label=0)
    assert labels.tolist() == [0, 1, 2, 0]  # 1-indexed scene -> 0-indexed
    assert positions.tolist() == [[0, 1], [0, 2], [1, 0], [1, 2]]


def test_flatten_labeled_pixels_rejects_unknown_label_range() -> None:
    gt = np.array([[0, 1, 5]], dtype=np.int64)
    with pytest.raises(ValueError, match="do not fit"):
        flatten_labeled_pixels(gt, num_classes=3, ignore_label=0)


def test_compute_scene_freq_sums_to_one_and_matches_counts() -> None:
    labels = np.array([0, 0, 0, 1, 1, 2], dtype=np.int64)
    pi = compute_scene_freq(labels, num_classes=3)
    assert pi.shape == (3,)
    assert torch.allclose(pi.sum(), torch.tensor(1.0, dtype=pi.dtype))
    expected = torch.tensor([3.0 / 6.0, 2.0 / 6.0, 1.0 / 6.0], dtype=pi.dtype)
    torch.testing.assert_close(pi, expected)


def test_compute_scene_freq_raises_on_empty_class() -> None:
    labels = np.array([0, 0, 2], dtype=np.int64)  # class 1 missing
    with pytest.raises(ValueError, match="zero labelled pixels"):
        compute_scene_freq(labels, num_classes=3)


# ---------------------------------------------------------------------------
# sampler
# ---------------------------------------------------------------------------


def test_sampler_returns_exactly_k_per_class() -> None:
    labels = np.concatenate([np.full(100, k, dtype=np.int64) for k in range(4)])  # 100 per class
    train_idx, test_idx = stratified_fixed_per_class_split(
        labels, samples_per_class=5, seed=0, num_classes=4
    )
    assert train_idx.size == 4 * 5
    assert test_idx.size == 4 * 95
    train_counts = np.bincount(labels[train_idx], minlength=4)
    test_counts = np.bincount(labels[test_idx], minlength=4)
    assert train_counts.tolist() == [5, 5, 5, 5]
    assert test_counts.tolist() == [95, 95, 95, 95]


def test_sampler_is_deterministic_with_same_seed() -> None:
    labels = np.concatenate([np.full(50, k) for k in range(3)])
    t1, e1 = stratified_fixed_per_class_split(labels, samples_per_class=5, seed=0, num_classes=3)
    t2, e2 = stratified_fixed_per_class_split(labels, samples_per_class=5, seed=0, num_classes=3)
    np.testing.assert_array_equal(t1, t2)
    np.testing.assert_array_equal(e1, e2)


def test_sampler_different_seed_produces_different_draw() -> None:
    labels = np.concatenate([np.full(50, k) for k in range(3)])
    t1, _ = stratified_fixed_per_class_split(labels, samples_per_class=5, seed=0, num_classes=3)
    t2, _ = stratified_fixed_per_class_split(labels, samples_per_class=5, seed=1, num_classes=3)
    # Per-class counts must still match.
    assert np.bincount(labels[t1], minlength=3).tolist() == [5, 5, 5]
    assert np.bincount(labels[t2], minlength=3).tolist() == [5, 5, 5]
    # But the actual indices drawn must differ.
    assert not np.array_equal(t1, t2)


def test_sampler_warns_when_class_is_starved() -> None:
    # Class 2 has only 3 pixels; with samples_per_class=5, the class is starved
    # and the sampler must warn and emit zero test indices for that class.
    labels = np.concatenate([np.full(20, 0), np.full(20, 1), np.full(3, 2)]).astype(np.int64)
    with pytest.warns(UserWarning, match="cannot be evaluated"):
        train_idx, test_idx = stratified_fixed_per_class_split(
            labels, samples_per_class=5, seed=0, num_classes=3
        )
    train_counts = np.bincount(labels[train_idx], minlength=3)
    test_counts = np.bincount(labels[test_idx], minlength=3)
    assert train_counts.tolist() == [5, 5, 3]
    assert test_counts.tolist() == [15, 15, 0]


# ---------------------------------------------------------------------------
# transforms
# ---------------------------------------------------------------------------


def test_per_band_standardize_fit_then_transform() -> None:
    rng = np.random.default_rng(0)
    train = rng.normal(loc=[1.0, 2.0, 3.0], scale=[0.5, 1.0, 2.0], size=(1000, 3))
    standardizer = PerBandStandardize().fit(train.astype(np.float32))
    transformed = standardizer.transform(train.astype(np.float32))
    np.testing.assert_allclose(transformed.mean(axis=0), 0.0, atol=1e-3)
    np.testing.assert_allclose(transformed.std(axis=0), 1.0, atol=1e-3)


def test_per_band_standardize_transform_before_fit_raises() -> None:
    with pytest.raises(RuntimeError, match="must be fit"):
        PerBandStandardize().transform(np.zeros((1, 3), dtype=np.float32))


def test_pca_reducer_fit_only_uses_provided_spectra() -> None:
    rng = np.random.default_rng(0)
    train = rng.normal(loc=5.0, size=(200, 10)).astype(np.float32)
    test = rng.normal(loc=-5.0, size=(200, 10)).astype(np.float32)

    pca = PCAReducer(n_components=4).fit(train)
    # PCA's mean_ must equal the train mean (within numerical tolerance), not
    # the train+test mean. This is the leak-freeness check.
    np.testing.assert_allclose(pca.mean_, train.mean(axis=0), atol=1e-4)
    combined_mean = np.concatenate([train, test]).mean(axis=0)
    assert not np.allclose(pca.mean_, combined_mean, atol=1.0)


def test_pad_cube_and_extract_patch_roundtrip() -> None:
    cube = np.arange(3 * 4 * 2, dtype=np.float32).reshape(3, 4, 2)
    padded = pad_cube_for_patches(cube, patch_size=3)
    assert padded.shape == (3 + 2, 4 + 2, 2)
    # Corner patch should include reflected boundary; centre pixel == original (0, 0).
    patch = extract_patch(padded, 0, 0, patch_size=3)
    assert patch.shape == (3, 3, 2)
    np.testing.assert_array_equal(patch[1, 1], cube[0, 0])


def test_pad_cube_rejects_even_patch_size() -> None:
    with pytest.raises(ValueError, match="positive and odd"):
        pad_cube_for_patches(np.zeros((3, 3, 1), dtype=np.float32), patch_size=4)


# ---------------------------------------------------------------------------
# HSIDataset + build_split_from_arrays
# ---------------------------------------------------------------------------


def _synthetic_split(seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    return _make_synthetic_scene()


def test_hsi_dataset_shapes_and_dtypes() -> None:
    cube, gt = _synthetic_split()
    artifacts = build_split_from_arrays(
        cube=cube,
        gt=gt,
        num_classes=4,
        ignore_label=0,
        patch_size=9,
        pca_components=6,
        samples_per_class=5,
        seed=0,
    )
    assert len(artifacts.train_dataset) == 4 * 5
    spectrum, patch, label = artifacts.train_dataset[0]
    assert spectrum.dtype == torch.float32
    assert spectrum.shape == (cube.shape[2],)  # raw spectral bands
    assert patch.dtype == torch.float32
    assert patch.shape == (6, 9, 9)  # (B', P, P)
    assert isinstance(label, int)
    assert 0 <= label < 4


def test_build_split_pca_leak_freeness() -> None:
    """PCA's ``mean_`` must equal the train spectra (standardised) mean."""
    cube, gt = _synthetic_split()
    artifacts = build_split_from_arrays(
        cube=cube,
        gt=gt,
        num_classes=4,
        ignore_label=0,
        patch_size=7,
        pca_components=5,
        samples_per_class=10,
        seed=42,
    )
    # Standardiser fit on TRAIN spectra (raw).
    assert artifacts.standardizer.mean_ is not None
    train_spectra_raw = cube[artifacts.train_positions[:, 0], artifacts.train_positions[:, 1], :]
    np.testing.assert_allclose(
        artifacts.standardizer.mean_, train_spectra_raw.mean(axis=0), atol=1e-4
    )
    # PCA fit on TRAIN spectra (standardised).
    cube_std = artifacts.standardizer.transform(cube)
    train_spectra_std = cube_std[
        artifacts.train_positions[:, 0], artifacts.train_positions[:, 1], :
    ]
    np.testing.assert_allclose(artifacts.pca.mean_, train_spectra_std.mean(axis=0), atol=1e-4)


def test_build_split_scene_freq_matches_full_gt() -> None:
    cube, gt = _synthetic_split()
    artifacts = build_split_from_arrays(
        cube=cube,
        gt=gt,
        num_classes=4,
        ignore_label=0,
        patch_size=5,
        pca_components=4,
        samples_per_class=5,
        seed=0,
    )
    assert artifacts.scene_freq.shape == (4,)
    assert torch.allclose(
        artifacts.scene_freq.sum(), torch.tensor(1.0, dtype=artifacts.scene_freq.dtype)
    )
    # Scene counts: (300, 200, 100, 50) per _make_synthetic_scene.
    total = 300 + 200 + 100 + 50
    expected = torch.tensor(
        [300 / total, 200 / total, 100 / total, 50 / total],
        dtype=artifacts.scene_freq.dtype,
    )
    torch.testing.assert_close(artifacts.scene_freq, expected)


def test_build_split_is_deterministic_with_same_seed() -> None:
    cube, gt = _synthetic_split()
    kwargs = dict(
        cube=cube,
        gt=gt,
        num_classes=4,
        ignore_label=0,
        patch_size=5,
        pca_components=4,
        samples_per_class=5,
        seed=0,
    )
    a = build_split_from_arrays(**kwargs)
    b = build_split_from_arrays(**kwargs)
    np.testing.assert_array_equal(a.train_labels, b.train_labels)
    np.testing.assert_array_equal(a.train_positions, b.train_positions)
    np.testing.assert_array_equal(a.test_labels, b.test_labels)
    s1, p1, l1 = a.train_dataset[0]
    s2, p2, l2 = b.train_dataset[0]
    torch.testing.assert_close(s1, s2)
    torch.testing.assert_close(p1, p2)
    assert l1 == l2


# ---------------------------------------------------------------------------
# Optional end-to-end real-data smoke test
# ---------------------------------------------------------------------------


def _indian_pines_available() -> bool:
    cube = REPO_ROOT / "data" / "indian_pines" / "Indian_pines_corrected.mat"
    gt = REPO_ROOT / "data" / "indian_pines" / "Indian_pines_gt.mat"
    return cube.is_file() and gt.is_file()


@pytest.mark.skipif(
    not _indian_pines_available(),
    reason="Indian Pines .mat files not present under data/indian_pines/",
)
def test_build_split_real_indian_pines_deterministic() -> None:
    """Phase 2B exit-criterion regression: 5 samples/class on IP, seed=0,
    two invocations produce bit-identical first items.
    """
    config = load_yaml(REPO_ROOT / "configs" / "datasets" / "indian_pines.yaml")
    config["samples_per_class"] = 5
    # Make data_dir absolute so the test works regardless of cwd.
    config["data_dir"] = str(REPO_ROOT / config["data_dir"])

    # The starvation warning could fire on extreme low-shot regimes; suppress
    # only for THIS call so the real exit-criterion behaviour is exercised.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        a = build_split(config, seed=0)
        b = build_split(config, seed=0)

    assert len(a.train_dataset) == 16 * 5
    assert a.num_classes == 16
    assert a.num_bands == 200
    np.testing.assert_array_equal(a.train_labels, b.train_labels)
    np.testing.assert_array_equal(a.train_positions, b.train_positions)

    s1, p1, l1 = a.train_dataset[0]
    s2, p2, l2 = b.train_dataset[0]
    torch.testing.assert_close(s1, s2)
    torch.testing.assert_close(p1, p2)
    assert l1 == l2

    # Scene frequencies sum to 1; rare-class frequency is small.
    assert torch.allclose(a.scene_freq.sum(), torch.tensor(1.0, dtype=a.scene_freq.dtype))
    pi_min = float(a.scene_freq.min())
    pi_max = float(a.scene_freq.max())
    assert pi_min > 0.0
    assert pi_min < pi_max
