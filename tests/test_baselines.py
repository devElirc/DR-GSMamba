"""Phase-3 unit tests for the baseline suite.

Coverage:

* Every deep baseline produces a logit tensor of the correct shape with a
  finite, differentiable gradient.
* The baseline registry resolves every name in :func:`all_baseline_names`.
* Each per-baseline YAML loads, parses, and round-trips through
  ``from_config`` for the deep baselines.
* Shallow baselines fit on a tiny synthetic dataset and produce predictions
  with the right shape.
* :class:`BaselineTrainer` runs two epochs end-to-end on a synthetic dataset
  and the metrics dict contains OA.
* Determinism: the same seed produces an identical first-batch shuffle order
  in :class:`BaselineTrainer` (matches the main ``Trainer`` policy).
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from torch.utils.data import TensorDataset

from hsi_robust.baselines import (
    CNN3D,
    SSFTT,
    BaselineTrainConfig,
    BaselineTrainer,
    HybridSN,
    MambaHSI,
    NonlocalGCN,
    SpectralFormer,
    all_baseline_names,
    deep_baseline_names,
    is_deep_baseline,
    is_shallow_baseline,
    make_deep_baseline,
    make_shallow_model,
    shallow_baseline_names,
)
from hsi_robust.data.hsi_dataset import HSIDataset

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_BASELINES = REPO_ROOT / "configs" / "baselines"


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _make_hsi_dataset(
    n: int = 16, bands: int = 30, pca: int = 8, patch: int = 9, num_classes: int = 4, seed: int = 0
) -> HSIDataset:
    """Return a tiny HSIDataset for the trainer end-to-end tests."""
    rng = np.random.default_rng(seed)
    cube_std = rng.standard_normal((20, 20, bands)).astype(np.float32)
    cube_pca = rng.standard_normal((20, 20, pca)).astype(np.float32)
    half = patch // 2
    cube_std_padded = np.pad(cube_std, ((half, half), (half, half), (0, 0)), mode="reflect")
    cube_pca_padded = np.pad(cube_pca, ((half, half), (half, half), (0, 0)), mode="reflect")
    positions = rng.integers(0, 20, size=(n, 2)).astype(np.int64)
    labels = rng.integers(0, num_classes, size=(n,)).astype(np.int64)
    return HSIDataset(
        cube_std_padded=cube_std_padded,
        cube_pca_padded=cube_pca_padded,
        positions=positions,
        labels=labels,
        patch_size=patch,
    )


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #


def test_registry_lists_all_baselines() -> None:
    names = set(all_baseline_names())
    expected = {
        "svm",
        "rf",
        "knn",
        "cnn3d",
        "hybridsn",
        "spectralformer",
        "ssftt",
        "nonlocal_gcn",
        "mambahsi",
    }
    assert names == expected
    for shallow in shallow_baseline_names():
        assert is_shallow_baseline(shallow)
        assert not is_deep_baseline(shallow)
    for deep in deep_baseline_names():
        assert is_deep_baseline(deep)
        assert not is_shallow_baseline(deep)


def test_registry_rejects_unknown_deep_baseline() -> None:
    with pytest.raises(ValueError):
        make_deep_baseline(
            "not_a_baseline", {}, num_bands=30, num_pca=8, patch_size=9, num_classes=4
        )


# --------------------------------------------------------------------------- #
# Deep baselines: forward + grad shapes
# --------------------------------------------------------------------------- #


@pytest.fixture
def tiny_inputs() -> dict:
    torch.manual_seed(0)
    return {
        "spectrum": torch.randn(4, 30, requires_grad=True),
        "patch": torch.randn(4, 8, 9, 9, requires_grad=True),
        "num_classes": 4,
    }


def _forward_and_check(model: torch.nn.Module, tiny_inputs: dict) -> None:
    logits = model(tiny_inputs["spectrum"], tiny_inputs["patch"])
    assert logits.shape == (4, tiny_inputs["num_classes"])
    assert torch.isfinite(logits).all()
    logits.sum().backward()
    # Gradients flow into at least one of the inputs.
    assert tiny_inputs["spectrum"].grad is not None or tiny_inputs["patch"].grad is not None


def test_cnn3d_forward(tiny_inputs: dict) -> None:
    model = CNN3D(
        num_pca=8, patch_size=9, num_classes=tiny_inputs["num_classes"], channels=(4, 8, 16)
    )
    _forward_and_check(model, tiny_inputs)


def test_hybridsn_forward(tiny_inputs: dict) -> None:
    # HybridSN needs num_pca >= 7 and a patch large enough for three 3D convs.
    model = HybridSN(num_pca=15, patch_size=9, num_classes=tiny_inputs["num_classes"])
    patch = torch.randn(4, 15, 9, 9, requires_grad=True)
    spec = torch.randn(4, 30)
    logits = model(spec, patch)
    assert logits.shape == (4, tiny_inputs["num_classes"])
    logits.sum().backward()
    assert patch.grad is not None


def test_spectralformer_forward(tiny_inputs: dict) -> None:
    model = SpectralFormer(
        num_bands=30, num_classes=tiny_inputs["num_classes"], dim=32, depth=2, heads=4
    )
    _forward_and_check(model, tiny_inputs)


def test_ssftt_forward(tiny_inputs: dict) -> None:
    model = SSFTT(
        num_pca=8,
        patch_size=9,
        num_classes=tiny_inputs["num_classes"],
        dim=32,
        num_tokens=4,
        depth=1,
    )
    _forward_and_check(model, tiny_inputs)


def test_nonlocal_gcn_forward(tiny_inputs: dict) -> None:
    model = NonlocalGCN(
        num_pca=8, patch_size=9, num_classes=tiny_inputs["num_classes"], dim=32, num_layers=2
    )
    _forward_and_check(model, tiny_inputs)


def test_mambahsi_forward(tiny_inputs: dict) -> None:
    model = MambaHSI(
        num_bands=30,
        num_pca=8,
        patch_size=9,
        num_classes=tiny_inputs["num_classes"],
        spectral_dim=32,
        spatial_dim=32,
        num_layers=1,
    )
    _forward_and_check(model, tiny_inputs)


# --------------------------------------------------------------------------- #
# Per-baseline YAML configs round-trip through from_config
# --------------------------------------------------------------------------- #


def _patched_for_test(name: str, raw_model_cfg: dict, num_pca: int) -> dict:
    """Patch a YAML model block to fit the tiny test tensors.

    HybridSN in particular requires num_pca >= 7 for its (7,3,3) 3D conv;
    we don't override that here -- callers pass num_pca large enough.
    """
    cfg = dict(raw_model_cfg)
    if name in {"cnn3d"}:
        # Shrink to keep parameter count small in the unit test.
        cfg["channels"] = [4, 8, 16]
    if name == "spectralformer":
        cfg["dim"] = 32
        cfg["depth"] = 2
    if name == "ssftt":
        cfg["dim"] = 32
        cfg["depth"] = 1
    if name == "nonlocal_gcn":
        cfg["dim"] = 32
    if name == "mambahsi":
        cfg["spectral_dim"] = 32
        cfg["spatial_dim"] = 32
        cfg["num_layers"] = 1
    return cfg


@pytest.mark.parametrize("name", list(deep_baseline_names()))
def test_per_baseline_yaml_loads_and_builds(name: str) -> None:
    path = CONFIG_BASELINES / f"{name}.yaml"
    assert path.is_file(), f"missing config for baseline '{name}'"
    cfg = yaml.safe_load(path.read_text())
    assert cfg["name"] == name
    assert cfg["family"] == "deep"
    model_cfg = _patched_for_test(name, dict(cfg.get("model") or {}), num_pca=15)
    model = make_deep_baseline(
        name,
        model_cfg,
        num_bands=30,
        num_pca=15 if name == "hybridsn" else 8,
        patch_size=9,
        num_classes=5,
    )
    spec = torch.randn(2, 30)
    patch = torch.randn(2, 15 if name == "hybridsn" else 8, 9, 9)
    logits = model(spec, patch)
    assert logits.shape == (2, 5)


@pytest.mark.parametrize("name", list(shallow_baseline_names()))
def test_per_shallow_yaml_loads(name: str) -> None:
    path = CONFIG_BASELINES / f"{name}.yaml"
    cfg = yaml.safe_load(path.read_text())
    assert cfg["name"] == name
    assert cfg["family"] == "shallow"


# --------------------------------------------------------------------------- #
# Shallow baselines: fit + predict on a tiny synthetic dataset
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("name", list(shallow_baseline_names()))
def test_shallow_baseline_fit_predict(name: str) -> None:
    train = _make_hsi_dataset(n=24, num_classes=3)
    test = _make_hsi_dataset(n=12, num_classes=3, seed=7)
    model = make_shallow_model(name, config=None, seed=0)
    model.fit(train)
    preds, probs = model.predict(test)
    assert preds.shape == (12,)
    assert probs.ndim == 2 and probs.shape[0] == 12
    # Probabilities (when available) sum to 1.
    if name != "knn":
        assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)


# --------------------------------------------------------------------------- #
# BaselineTrainer end-to-end on tiny tensors
# --------------------------------------------------------------------------- #


class _PassThroughDataset(TensorDataset):
    """(spectrum, patch, label) tuples for the BaselineTrainer."""

    def __init__(
        self, n: int, num_bands: int, num_pca: int, patch_size: int, num_classes: int, seed: int = 0
    ) -> None:
        g = torch.Generator().manual_seed(seed)
        labels = torch.randint(0, num_classes, (n,), generator=g)
        spec = torch.randn(n, num_bands, generator=g) + labels.float().unsqueeze(1) * 0.05
        patch = (
            torch.randn(n, num_pca, patch_size, patch_size, generator=g)
            + labels.float().view(-1, 1, 1, 1) * 0.05
        )
        super().__init__(spec, patch, labels)


def test_baseline_trainer_runs_end_to_end() -> None:
    num_classes = 3
    model = CNN3D(num_pca=4, patch_size=5, num_classes=num_classes, channels=(4, 8, 16))
    train_ds = _PassThroughDataset(
        n=24, num_bands=12, num_pca=4, patch_size=5, num_classes=num_classes, seed=1
    )
    val_ds = _PassThroughDataset(
        n=12, num_bands=12, num_pca=4, patch_size=5, num_classes=num_classes, seed=2
    )
    scene_freq = torch.full((num_classes,), 1.0 / num_classes)
    cfg = BaselineTrainConfig(
        epochs=2,
        batch_size=8,
        num_workers=0,
        pin_memory=False,
        scheduler={"name": "cosine", "warmup_epochs": 1, "min_lr": 1e-4},
        save_every=99,
        log_every=99,
    )
    with tempfile.TemporaryDirectory() as tmp:
        tr = BaselineTrainer(
            model=model,
            scene_freq=scene_freq,
            train_dataset=train_ds,
            val_dataset=val_ds,
            config=cfg,
            output_dir=Path(tmp),
            seed=0,
        )
        state = tr.fit()
    assert len(state.history) == 2
    last = state.history[-1]
    assert "val/OA" in last and isinstance(last["val/OA"], float)


def test_baseline_trainer_dataloader_seed_keyed_to_experiment_seed() -> None:
    """Same D-11 contract as the main Trainer: shuffle order depends on seed."""
    num_classes = 3
    train_ds = _PassThroughDataset(
        n=32, num_bands=10, num_pca=4, patch_size=3, num_classes=num_classes, seed=0
    )
    val_ds = _PassThroughDataset(
        n=4, num_bands=10, num_pca=4, patch_size=3, num_classes=num_classes, seed=99
    )
    scene_freq = torch.full((num_classes,), 1.0 / num_classes)
    cfg = BaselineTrainConfig(
        epochs=1,
        batch_size=4,
        num_workers=0,
        pin_memory=False,
        scheduler={"name": "cosine", "warmup_epochs": 0, "min_lr": 1e-4},
        save_every=99,
        val_every=99,
    )

    def _first_batch_labels(seed: int) -> torch.Tensor:
        torch.manual_seed(seed)
        model = CNN3D(num_pca=4, patch_size=3, num_classes=num_classes, channels=(2, 4, 8))
        with tempfile.TemporaryDirectory() as tmp:
            tr = BaselineTrainer(
                model=model,
                scene_freq=scene_freq,
                train_dataset=train_ds,
                val_dataset=val_ds,
                config=cfg,
                output_dir=Path(tmp),
                seed=seed,
            )
            return next(iter(tr.train_loader))[2]

    labels_seed0 = _first_batch_labels(0)
    labels_seed1 = _first_batch_labels(1)
    labels_seed0_again = _first_batch_labels(0)
    assert torch.equal(labels_seed0, labels_seed0_again)
    assert not torch.equal(labels_seed0, labels_seed1)
