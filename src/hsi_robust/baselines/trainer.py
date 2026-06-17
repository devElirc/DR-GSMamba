"""Compact training loop for the deep baselines.

Identical data path and identical optimiser/scheduler stack as
:class:`hsi_robust.training.Trainer` so head-to-head numbers are clean. The
only differences are:

* the loss is **plain mean cross-entropy** (no CFA-GDRO, no EPH, no CP-Graph),
* the model returns logits directly (no Dirichlet posterior),
* the evaluator measures OA / AA / kappa / per-class with the same
  :mod:`hsi_robust.eval.metrics` code so the resulting numbers are
  directly comparable to the main model's.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from hsi_robust.eval.calibration import expected_calibration_error
from hsi_robust.eval.metrics import compute_metrics
from hsi_robust.training.optim import build_optimizer, build_scheduler, clip_grad_norm


@dataclass
class BaselineTrainConfig:
    """Trainer config for plain cross-entropy baselines."""

    epochs: int = 200
    batch_size: int = 64
    num_workers: int = 0
    pin_memory: bool = False

    optimizer: dict[str, Any] = field(
        default_factory=lambda: {
            "name": "adamw",
            "lr": 1e-3,
            "weight_decay": 1e-4,
            "betas": (0.9, 0.999),
        }
    )
    scheduler: dict[str, Any] = field(
        default_factory=lambda: {
            "name": "cosine",
            "warmup_epochs": 5,
            "min_lr": 1e-5,
        }
    )
    grad_clip: float = 1.0

    log_every: int = 50
    val_every: int = 1
    save_every: int = 50

    @classmethod
    def from_yaml_dict(cls, training_cfg: dict[str, Any]) -> BaselineTrainConfig:
        return cls(
            epochs=int(training_cfg.get("epochs", 200)),
            batch_size=int(training_cfg.get("batch_size", 64)),
            num_workers=int(training_cfg.get("num_workers", 0)),
            pin_memory=bool(training_cfg.get("pin_memory", False)),
            optimizer=dict(training_cfg.get("optimizer") or {}),
            scheduler=dict(training_cfg.get("scheduler") or {}),
            grad_clip=float(training_cfg.get("grad_clip", 1.0)),
            log_every=int(training_cfg.get("log_every", 50)),
            val_every=int(training_cfg.get("val_every", 1)),
            save_every=int(training_cfg.get("save_every", 50)),
        )


@dataclass
class BaselineTrainState:
    epoch: int = 0
    global_step: int = 0
    best_oa: float = 0.0
    history: list[dict[str, Any]] = field(default_factory=list)


def _move_batch(batch: tuple, device: torch.device) -> tuple:
    spec, patch, label = batch
    return (
        spec.to(device, non_blocking=False),
        patch.to(device, non_blocking=False),
        label.to(device, non_blocking=False),
    )


class BaselineTrainer:
    """Plain CE trainer for any deep baseline in :mod:`hsi_robust.baselines`."""

    def __init__(
        self,
        *,
        model: nn.Module,
        scene_freq: torch.Tensor,
        train_dataset: torch.utils.data.Dataset,
        val_dataset: torch.utils.data.Dataset,
        config: BaselineTrainConfig,
        output_dir: Path,
        device: torch.device | str = "cpu",
        on_log: Callable[[dict[str, Any]], None] | None = None,
        seed: int = 0,
    ) -> None:
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.scene_freq = scene_freq.to(self.device).float()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.on_log = on_log
        self.seed = int(seed)

        # Same seed-keyed shuffle as the main Trainer (decision D-11).
        self._dl_generator = torch.Generator()
        self._dl_generator.manual_seed(self.seed + 1)

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
            drop_last=False,
            generator=self._dl_generator,
        )
        eval_batch = max(config.batch_size, min(512, 8 * config.batch_size))
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=eval_batch,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
            drop_last=False,
        )

        self.optimizer = build_optimizer(self.model.parameters(), config.optimizer)
        steps_per_epoch = max(1, len(self.train_loader))
        total_steps = max(1, config.epochs * steps_per_epoch)
        self.scheduler = build_scheduler(
            self.optimizer,
            config.scheduler,
            total_steps=total_steps,
            steps_per_epoch=steps_per_epoch,
        )

        try:
            self.num_classes = int(model.num_classes)  # type: ignore[attr-defined]
        except AttributeError as exc:
            raise AttributeError("baseline model must expose `num_classes` attribute") from exc

        self.state = BaselineTrainState()

    # ------------------------------------------------------------------ #
    def fit(self) -> BaselineTrainState:
        for epoch in range(self.config.epochs):
            self.state.epoch = epoch
            epoch_loss = self._train_epoch()
            log_row: dict[str, Any] = {
                "epoch": epoch,
                "lr": float(self.optimizer.param_groups[0]["lr"]),
                "train_loss": epoch_loss,
            }
            if (epoch + 1) % self.config.val_every == 0:
                val_metrics = self.evaluate()
                log_row.update({f"val/{k}": v for k, v in val_metrics.items()})
                if (
                    isinstance(val_metrics.get("OA"), float)
                    and val_metrics["OA"] > self.state.best_oa
                ):
                    self.state.best_oa = val_metrics["OA"]
                    self._save_checkpoint("best.pt", val_metrics=val_metrics)
            self.state.history.append(log_row)
            if self.on_log is not None:
                self.on_log(log_row)
            if (epoch + 1) % self.config.save_every == 0:
                self._save_checkpoint(f"epoch_{epoch + 1:04d}.pt")
        self._save_checkpoint("last.pt")
        self._save_history()
        return self.state

    def _train_epoch(self) -> float:
        self.model.train()
        running = 0.0
        n_batches = 0
        for batch in self.train_loader:
            spec, patch, labels = _move_batch(batch, self.device)
            logits = self.model(spec, patch)
            loss = F.cross_entropy(logits, labels)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            _ = clip_grad_norm(self.model.parameters(), self.config.grad_clip)
            self.optimizer.step()
            self.scheduler.step()
            running += float(loss.detach())
            n_batches += 1
            self.state.global_step += 1
        return running / max(1, n_batches)

    @torch.no_grad()
    def evaluate(self) -> dict[str, Any]:
        self.model.eval()
        all_probs: list[np.ndarray] = []
        all_preds: list[np.ndarray] = []
        all_labels: list[np.ndarray] = []
        for batch in self.val_loader:
            spec, patch, labels = _move_batch(batch, self.device)
            logits = self.model(spec, patch)
            probs = F.softmax(logits, dim=-1).detach().cpu().numpy()
            preds = probs.argmax(axis=1)
            all_probs.append(probs)
            all_preds.append(preds)
            all_labels.append(labels.detach().cpu().numpy())

        probs_arr = (
            np.concatenate(all_probs, axis=0) if all_probs else np.zeros((0, self.num_classes))
        )
        preds_arr = (
            np.concatenate(all_preds, axis=0) if all_preds else np.zeros((0,), dtype=np.int64)
        )
        labels_arr = (
            np.concatenate(all_labels, axis=0) if all_labels else np.zeros((0,), dtype=np.int64)
        )

        metrics = compute_metrics(
            labels_arr,
            preds_arr,
            num_classes=self.num_classes,
            scene_freq=self.scene_freq.detach().cpu().numpy(),
        )
        if probs_arr.shape[0] > 0:
            ece, _ = expected_calibration_error(probs_arr, labels_arr, num_bins=15)
            metrics["ECE_15"] = float(ece)
        return metrics

    # ------------------------------------------------------------------ #
    def _save_checkpoint(self, name: str, *, val_metrics: dict[str, Any] | None = None) -> Path:
        path = self.output_dir / name
        payload = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "state": {
                "epoch": self.state.epoch,
                "global_step": self.state.global_step,
                "best_oa": self.state.best_oa,
            },
            "config": self.config.__dict__,
            "val_metrics": val_metrics,
        }
        torch.save(payload, path)
        return path

    def _save_history(self) -> Path:
        path = self.output_dir / "metrics.json"
        with path.open("w", encoding="utf-8") as fh:
            json.dump(
                {
                    "history": self.state.history,
                    "best_oa": self.state.best_oa,
                    "total_epochs": self.state.epoch + 1,
                    "wall_time": time.time(),
                },
                fh,
                indent=2,
            )
        return path
