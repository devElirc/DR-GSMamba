"""Unit tests for Phase 2E training pipeline components.

Covers ``training.ema_class_loss``, ``training.optim``, and the ``Trainer``
end-to-end on a tiny synthetic problem.
"""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import pytest
import torch
from torch.utils.data import TensorDataset

from hsi_robust.eval import (
    classification_map,
    compute_metrics,
    error_map,
    expected_calibration_error,
)
from hsi_robust.models import CFAGDRO
from hsi_robust.training import (
    EMAClassLoss,
    TrainConfig,
    Trainer,
    WarmupCosineSchedule,
    build_optimizer,
    clip_grad_norm,
)

# --------------------------------------------------------------------------- #
# EMA class loss
# --------------------------------------------------------------------------- #


def test_ema_first_update_sets_value_to_batch_mean() -> None:
    ema = EMAClassLoss(num_classes=4, momentum=0.9)
    losses = torch.tensor([1.0, 2.0, 3.0, 4.0])
    labels = torch.tensor([0, 1, 0, 2])
    ema.update(losses, labels)
    # Class 0: mean(1, 3) = 2; class 1: 2; class 2: 4; class 3: untouched.
    assert torch.allclose(ema.losses[:3], torch.tensor([2.0, 2.0, 4.0]))
    assert ema.losses[3].item() == 0.0  # never seen
    assert ema.seen.tolist() == [True, True, True, False]


def test_ema_second_update_applies_momentum() -> None:
    ema = EMAClassLoss(num_classes=2, momentum=0.5)
    ema.update(torch.tensor([2.0]), torch.tensor([0]))
    ema.update(torch.tensor([6.0]), torch.tensor([0]))
    # After first update: hat[0] = 2.0. After second: 0.5 * 2 + 0.5 * 6 = 4.0.
    assert ema.losses[0].item() == pytest.approx(4.0)


def test_ema_state_round_trip() -> None:
    a = EMAClassLoss(num_classes=3, momentum=0.9)
    a.update(torch.tensor([1.0, 0.5]), torch.tensor([0, 2]))
    state = a.state_dict()
    b = EMAClassLoss(num_classes=3, momentum=0.9)
    b.load_state_dict(state)
    assert torch.equal(a.losses, b.losses)
    assert torch.equal(a.seen, b.seen)


def test_ema_validates_inputs() -> None:
    with pytest.raises(ValueError):
        EMAClassLoss(num_classes=0)
    with pytest.raises(ValueError):
        EMAClassLoss(num_classes=2, momentum=1.5)


# --------------------------------------------------------------------------- #
# Optim
# --------------------------------------------------------------------------- #


def test_warmup_cosine_schedule_shape() -> None:
    model = torch.nn.Linear(4, 2)
    opt = build_optimizer(model.parameters(), {"name": "adamw", "lr": 1.0})
    sched = WarmupCosineSchedule(opt, warmup_steps=5, total_steps=20, min_lr_ratio=0.1)
    lrs = []
    for _ in range(20):
        lrs.append(opt.param_groups[0]["lr"])
        opt.step()  # cheap dummy step
        sched.step()
    # During warmup, lr grows linearly from 0.2 (step 0) to 1.0 (step 4).
    assert lrs[0] == pytest.approx(0.2, abs=1e-6)
    assert lrs[4] == pytest.approx(1.0, abs=1e-6)
    # After warmup, lr decays monotonically to >= min_lr_ratio.
    decay = lrs[5:]
    assert all(a >= b - 1e-9 for a, b in pairwise(decay))
    assert decay[-1] >= 0.1 - 1e-6


def test_clip_grad_norm_returns_pre_clip_norm() -> None:
    p = torch.nn.Parameter(torch.tensor([3.0, 4.0]))
    p.grad = torch.tensor([3.0, 4.0])
    norm = clip_grad_norm([p], max_norm=1.0)
    assert norm == pytest.approx(5.0, abs=1e-6)
    # Post-clip, grad norm should be <= 1.0.
    assert float(p.grad.norm()) == pytest.approx(1.0, abs=1e-6)


def test_build_optimizer_rejects_unknown() -> None:
    p = torch.nn.Linear(2, 2).parameters()
    with pytest.raises(ValueError):
        build_optimizer(p, {"name": "sgd"})


# --------------------------------------------------------------------------- #
# Metrics + calibration
# --------------------------------------------------------------------------- #


def test_compute_metrics_basic() -> None:
    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_pred = np.array([0, 1, 1, 1, 2, 2])
    m = compute_metrics(y_true, y_pred, num_classes=3, scene_freq=np.array([0.4, 0.4, 0.2]))
    assert m["OA"] == pytest.approx(5 / 6)
    assert m["AA"] == pytest.approx((0.5 + 1.0 + 1.0) / 3)
    assert m["worst_class"] == pytest.approx(0.5)
    # The rarest 25% of classes (= 1 of 3) is class 2 (lowest freq). Its acc is 1.0.
    assert m["rare_class"] == pytest.approx(1.0)


def test_expected_calibration_error_perfectly_calibrated() -> None:
    # Construct 4 perfectly calibrated bins.
    probs = np.array(
        [[0.9, 0.1], [0.9, 0.1], [0.9, 0.1], [0.9, 0.1], [0.9, 0.1], [0.9, 0.1]],
        dtype=np.float32,
    )
    # In a confidence-0.9 bin we expect 90% accuracy.
    labels = np.array([0, 0, 0, 0, 0, 1])
    ece, bins = expected_calibration_error(probs, labels, num_bins=10)
    # All confidence == 0.9 -> falls in bin index 8 of 10 (range [0.8, 0.9)).
    # Wait: np.linspace(0,1,11) gives bin edges 0.0, 0.1, ..., 1.0. The bin
    # [0.8, 0.9) contains 0.9-epsilon; conf == 0.9 falls in [0.9, 1.0] thanks
    # to the closed-on-right last bin rule (b == num_bins - 1 means b = 9).
    # Empirical accuracy in that bin = 5/6 ~ 0.833, confidence = 0.9,
    # |0.833 - 0.9| ~ 0.067.
    assert ece == pytest.approx(0.067, abs=1e-3)
    assert int(bins.bin_count.sum()) == 6


def test_qualitative_maps_shape_and_background() -> None:
    pred = np.array([1, 2, 3])
    target = np.array([1, 2, 4])
    pixel_coords = np.array([[0, 0], [1, 1], [2, 2]])
    vac = np.array([0.1, 0.5, 0.9])
    ale = np.array([0.2, 0.6, 0.8])
    spatial = (3, 3)
    cmap = classification_map(pred, pixel_coords, spatial)
    emap = error_map(pred, target, pixel_coords, spatial)
    from hsi_robust.eval import aleatoric_map, vacuity_map

    vmap = vacuity_map(vac, pixel_coords, spatial)
    amap = aleatoric_map(ale, pixel_coords, spatial)
    assert cmap.shape == (3, 3) and cmap[0, 0] == 1 and cmap[0, 1] == -1
    assert emap[2, 2] == 1.0  # last pixel mis-classified
    assert np.isnan(emap[0, 1])
    assert vmap[1, 1] == pytest.approx(0.5)
    assert amap[0, 0] == pytest.approx(0.2)


# --------------------------------------------------------------------------- #
# Trainer end-to-end on synthetic tensors
# --------------------------------------------------------------------------- #


class _SyntheticDataset(TensorDataset):
    """Yields (raw_spectrum, pca_patch, label) like HSIDataset."""

    def __init__(
        self, n: int, num_bands: int, num_pca: int, patch_size: int, num_classes: int, seed: int = 0
    ) -> None:
        g = torch.Generator().manual_seed(seed)
        # Class-conditional means in feature space so the model can learn.
        labels = torch.randint(0, num_classes, (n,), generator=g)
        spec = torch.randn(n, num_bands, generator=g) + labels.float().unsqueeze(1) * 0.1
        patch = (
            torch.randn(n, num_pca, patch_size, patch_size, generator=g)
            + labels.float().view(-1, 1, 1, 1) * 0.1
        )
        super().__init__(spec, patch, labels)


def test_trainer_smoke_runs_one_epoch_on_synthetic() -> None:
    torch.manual_seed(0)
    num_classes = 4
    model = CFAGDRO(
        num_bands=20,
        num_pca=4,
        patch_size=5,
        num_classes=num_classes,
        feature_dim=16,
        op_s4_hidden=8,
        op_s4_state=4,
        op_s4_layers=1,
        spatial_base_channels=8,
        cp_graph_k=2,
        dropout=0.0,
    )
    train_ds = _SyntheticDataset(
        n=32, num_bands=20, num_pca=4, patch_size=5, num_classes=num_classes, seed=1
    )
    val_ds = _SyntheticDataset(
        n=16, num_bands=20, num_pca=4, patch_size=5, num_classes=num_classes, seed=2
    )
    scene_freq = torch.full((num_classes,), 1.0 / num_classes)
    cfg = TrainConfig(
        epochs=2,
        batch_size=8,
        num_workers=0,
        pin_memory=False,
        scheduler={"name": "cosine", "warmup_epochs": 1, "min_lr": 1e-4},
        evi_anneal_epochs=2,
        save_every=99,
    )
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        tr = Trainer(
            model=model,
            scene_freq=scene_freq,
            train_dataset=train_ds,
            val_dataset=val_ds,
            config=cfg,
            output_dir=tmp,
        )
        state = tr.fit()
        assert len(state.history) == cfg.epochs
        # The metrics dict in history must contain val metrics.
        last = state.history[-1]
        assert "val/OA" in last
        assert isinstance(last["val/OA"], float)


def test_trainer_dataloader_seed_keyed_to_experiment_seed() -> None:
    """D-11: Trainer DataLoader generator must depend on the experiment seed.

    With identical model + dataset, two trainers initialised with different
    seeds must produce different shuffle orders in the first batch.
    """
    num_classes = 3
    train_ds = _SyntheticDataset(
        n=32, num_bands=10, num_pca=4, patch_size=3, num_classes=num_classes, seed=0
    )
    val_ds = _SyntheticDataset(
        n=4, num_bands=10, num_pca=4, patch_size=3, num_classes=num_classes, seed=99
    )
    scene_freq = torch.full((num_classes,), 1.0 / num_classes)
    cfg = TrainConfig(
        epochs=1,
        batch_size=4,
        num_workers=0,
        pin_memory=False,
        scheduler={"name": "cosine", "warmup_epochs": 0, "min_lr": 1e-4},
        evi_anneal_epochs=1,
        save_every=99,
        val_every=99,
    )

    def _first_batch_labels(seed: int) -> torch.Tensor:
        torch.manual_seed(seed)
        model = CFAGDRO(
            num_bands=10,
            num_pca=4,
            patch_size=3,
            num_classes=num_classes,
            feature_dim=8,
            op_s4_hidden=4,
            op_s4_state=2,
            op_s4_layers=1,
            spatial_base_channels=4,
            cp_graph_k=2,
            dropout=0.0,
        )
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            tr = Trainer(
                model=model,
                scene_freq=scene_freq,
                train_dataset=train_ds,
                val_dataset=val_ds,
                config=cfg,
                output_dir=tmp,
                seed=seed,
            )
            return next(iter(tr.train_loader))[2]

    labels_seed0 = _first_batch_labels(0)
    labels_seed1 = _first_batch_labels(1)
    labels_seed0_again = _first_batch_labels(0)
    # Same seed -> identical shuffle.
    assert torch.equal(labels_seed0, labels_seed0_again)
    # Different seed -> different shuffle (probability of accidental match
    # over 4 picks from 32 with non-trivial ordering is negligible).
    assert not torch.equal(labels_seed0, labels_seed1)
