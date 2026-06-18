"""End-to-end trainer for the CFA-GDRO model (backbone + EPH + CP-Graph).

The total loss assembled here is exactly Eq. (4) of
``docs/math/cfa_gdro.md`` §3:

    L_total = mean_CE
              + lambda_rob  * L_CFA-GDRO   (operates on per-sample CE losses)
              + lambda_evi  * L_EPH         (Bayes-risk + annealed KL)
              + lambda_graph * L_CP-graph

The ``mean_CE`` anchor is required for stable optimisation: the EPH Bayes-risk
loss has a vanishing gradient at the uniform-prediction saddle (verified
empirically during the Phase 2E smoke debug) and cannot escape it on its own.
The EPH note's Eq. (16) was a transcription drift; this trainer follows the
CFA-GDRO note Eq. (4) which is the source of truth (see decision D-08 in
``roadmap.md``).

CE logits come from the EPH head itself -- ``logits = tau * cos(f, m)`` -- so
the EPH and CE pathways share *exactly* the same network and prototypes, and
the EPH calibration claim still operates on the same predictions the trainer
optimises for OA.

Everything else (model, losses, EMA, optimiser, scheduler, metrics) is
delegated to the modules they belong to.

Public surface:

* :class:`TrainConfig` -- typed config for the trainer.
* :class:`TrainState`  -- lightweight container for the runtime state.
* :class:`Trainer`     -- the trainer itself.
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
from torch.utils.data import DataLoader

from hsi_robust.eval.calibration import expected_calibration_error
from hsi_robust.eval.metrics import compute_metrics
from hsi_robust.losses import cfa_gdro_loss, cp_graph_loss, evidential_loss
from hsi_robust.models import CFAGDRO
from hsi_robust.training.ema_class_loss import EMAClassLoss
from hsi_robust.training.optim import build_optimizer, build_scheduler, clip_grad_norm


@dataclass
class TrainConfig:
    """Trainer configuration. Reflects ``configs/training/default.yaml``."""

    epochs: int = 200
    batch_size: int = 64
    num_workers: int = 0  # CPU-friendly default
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

    # Loss-mixing weights (matching cfa_gdro.md §3 Eq. (4)).
    cfa_alpha: float = 0.3
    cfa_gamma: float = 1.0
    cfa_lambda_rob: float = 0.5
    cfa_ema_momentum: float = 0.9
    evi_anneal_epochs: int = 10
    evi_lambda: float = 1.0  # lambda_evi
    cp_graph_k: int = 8
    cp_graph_tau_g: float = 1.0
    cp_graph_weight: float = 0.1
    cp_graph_stop_grad_target: bool = True

    # Bookkeeping.
    log_every: int = 50
    val_every: int = 1
    save_every: int = 50

    @classmethod
    def from_yaml_dict(cls, training_cfg: dict[str, Any]) -> TrainConfig:
        """Build from a parsed ``configs/training/*.yaml`` dict."""
        loss = training_cfg.get("loss") or {}
        cfa = loss.get("cfa_gdro") or {}
        evi = loss.get("evidential") or {}
        cpg = loss.get("cp_graph") or {}
        return cls(
            epochs=int(training_cfg.get("epochs", 200)),
            batch_size=int(training_cfg.get("batch_size", 64)),
            num_workers=int(training_cfg.get("num_workers", 0)),
            pin_memory=bool(training_cfg.get("pin_memory", False)),
            optimizer=dict(training_cfg.get("optimizer") or {}),
            scheduler=dict(training_cfg.get("scheduler") or {}),
            grad_clip=float(training_cfg.get("grad_clip", 1.0)),
            cfa_alpha=float(cfa.get("alpha", 0.3)),
            cfa_gamma=float(cfa.get("gamma", 1.0)),
            cfa_lambda_rob=float(cfa.get("weight", 0.5)),
            cfa_ema_momentum=float(cfa.get("ema_momentum", 0.9)),
            evi_anneal_epochs=int(evi.get("anneal_epochs", 10)),
            evi_lambda=float(evi.get("weight", 1.0)),
            cp_graph_k=int(cpg.get("k", 8)),
            cp_graph_tau_g=float(cpg.get("tau_g", 1.0)),
            cp_graph_weight=float(cpg.get("weight", 0.1)),
            cp_graph_stop_grad_target=bool(cpg.get("stop_grad_target", True)),
            log_every=int(training_cfg.get("log_every", 50)),
            val_every=int(training_cfg.get("val_every", 1)),
            save_every=int(training_cfg.get("save_every", 50)),
        )


@dataclass
class TrainState:
    """Lightweight runtime state. Updated in-place by the trainer."""

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


class Trainer:
    """Train one ``CFAGDRO`` instance with the full CFA-GDRO + EPH + CP-Graph stack."""

    def __init__(
        self,
        *,
        model: CFAGDRO,
        scene_freq: torch.Tensor,
        train_dataset: torch.utils.data.Dataset,
        val_dataset: torch.utils.data.Dataset,
        config: TrainConfig,
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

        # Train DataLoader generator: keyed by the experiment seed so every
        # seed in the Phase 6 grid sees a different batch shuffle order.
        # ``seed + 1`` keeps the model-init RNG (seeded with ``seed``) and the
        # shuffle RNG on independent streams.
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
        # Eval has no backward, so a larger batch reduces Python overhead by 3-4x
        # compared to the train batch size. We cap at 512 for memory safety.
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
        total_steps = max(1, config.epochs * max(1, len(self.train_loader)))
        self.scheduler = build_scheduler(
            self.optimizer,
            config.scheduler,
            total_steps=total_steps,
            steps_per_epoch=max(1, len(self.train_loader)),
        )

        self.ema = EMAClassLoss(
            num_classes=model.num_classes,
            momentum=config.cfa_ema_momentum,
            device=self.device,
            dtype=torch.float32,
        )

        self.state = TrainState()

    # ------------------------------------------------------------------ #
    # Loss assembly
    # ------------------------------------------------------------------ #
    def _kl_anneal(self, epoch: int) -> float:
        """``lambda_t = min(1, t / T_anneal)`` from EPH math note §4.3."""
        t_anneal = max(1, int(self.config.evi_anneal_epochs))
        return min(1.0, (epoch + 1) / t_anneal)

    def _step(
        self,
        spec: torch.Tensor,
        patch: torch.Tensor,
        labels: torch.Tensor,
        kl_weight: float,
    ) -> dict[str, Any]:
        """One forward + loss assembly per ``cfa_gdro.md`` §3 Eq. (4)."""
        out = self.model(spec, patch)
        alpha = out["alpha"]
        probs = out["probs"]
        fused = out["fused_feat"]
        cos = out["cos"]
        tau = out["temperature"]

        # CE pathway (anchor + CFA-GDRO input). Logits = tau * cos -- same as
        # what EPH consumes upstream so the two views agree on predictions.
        logits = tau * cos
        per_sample_ce = F.cross_entropy(logits, labels, reduction="none")
        mean_ce = per_sample_ce.mean()

        # EPH calibration term (auxiliary to CE).
        _, mean_eph, evi_info = evidential_loss(alpha, labels, kl_weight=kl_weight)

        # CFA-GDRO on per-sample CE losses (EMA-stabilised).
        self.ema.update(per_sample_ce.detach(), labels)
        cfa, cfa_info = cfa_gdro_loss(
            per_sample_ce,
            labels,
            self.scene_freq,
            alpha=self.config.cfa_alpha,
            gamma=self.config.cfa_gamma,
            ema_class_losses=self.ema.losses,
            ema_seen=self.ema.seen,
            ema_momentum=self.config.cfa_ema_momentum,
        )

        # CP-Graph consistency on fused features + EPH predictive probabilities.
        cp_loss, cp_info = cp_graph_loss(
            fused,
            probs,
            k=self.config.cp_graph_k,
            tau_g=self.config.cp_graph_tau_g,
            stop_grad_target=self.config.cp_graph_stop_grad_target,
        )

        total = (
            mean_ce
            + self.config.cfa_lambda_rob * cfa
            + self.config.evi_lambda * mean_eph
            + self.config.cp_graph_weight * cp_loss
        )

        scalars = {
            "loss/total": float(total.detach()),
            "loss/mean_ce": float(mean_ce.detach()),
            "loss/mean_eph": float(mean_eph.detach()),
            "loss/cfa_gdro": float(cfa.detach()),
            "loss/cp_graph": float(cp_loss.detach()),
            "loss/kl": float(evi_info["kl"]),
            "loss/lik": float(evi_info["lik"]),
            "loss/kl_weight": float(kl_weight),
            "cfa/threshold_nu": float(cfa_info["threshold_nu"]),
            "cfa/active_set_size": int(cfa_info["active_set_size"]),
            "cp_graph/weight_entropy": float(cp_info["mean_weight_entropy"]),
            "eph/mean_vacuity": float(evi_info["mean_vacuity"]),
            "eph/temperature": float(tau.detach()),
        }
        return {"total": total, "scalars": scalars}

    # ------------------------------------------------------------------ #
    # Main loop
    # ------------------------------------------------------------------ #
    def fit(self) -> TrainState:
        """Run the training loop for ``epochs`` epochs and return final state."""
        for epoch in range(self.config.epochs):
            self.state.epoch = epoch
            kl_weight = self._kl_anneal(epoch)
            epoch_loss = self._train_epoch(kl_weight)
            log_row: dict[str, Any] = {
                "epoch": epoch,
                "lr": float(self.optimizer.param_groups[0]["lr"]),
                "train_loss": epoch_loss,
                "kl_weight": kl_weight,
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
        # Final dump.
        self._save_checkpoint("last.pt")
        self._save_history()
        return self.state

    def _train_epoch(self, kl_weight: float) -> float:
        self.model.train()
        running = 0.0
        n_batches = 0
        for batch in self.train_loader:
            spec, patch, labels = _move_batch(batch, self.device)
            out = self._step(spec, patch, labels, kl_weight=kl_weight)
            total: torch.Tensor = out["total"]
            self.optimizer.zero_grad(set_to_none=True)
            total.backward()
            _ = clip_grad_norm(self.model.parameters(), self.config.grad_clip)
            self.optimizer.step()
            self.scheduler.step()

            running += float(total.detach())
            n_batches += 1
            self.state.global_step += 1
        return running / max(1, n_batches)

    @torch.no_grad()
    def evaluate(self) -> dict[str, Any]:
        self.model.eval()
        all_probs: list[np.ndarray] = []
        all_preds: list[np.ndarray] = []
        all_labels: list[np.ndarray] = []
        all_vacuity: list[np.ndarray] = []
        all_aleatoric: list[np.ndarray] = []
        for batch in self.val_loader:
            spec, patch, labels = _move_batch(batch, self.device)
            out = self.model(spec, patch)
            probs = out["probs"].detach().cpu().numpy()
            preds = probs.argmax(axis=1)
            all_probs.append(probs)
            all_preds.append(preds)
            all_labels.append(labels.detach().cpu().numpy())
            all_vacuity.append(out["vacuity"].detach().cpu().numpy())
            all_aleatoric.append(out["aleatoric"].detach().cpu().numpy())

        probs_arr = (
            np.concatenate(all_probs, axis=0)
            if all_probs
            else np.zeros((0, self.model.num_classes))
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
            num_classes=self.model.num_classes,
            scene_freq=self.scene_freq.detach().cpu().numpy(),
        )
        if probs_arr.shape[0] > 0:
            ece, _ = expected_calibration_error(probs_arr, labels_arr, num_bins=15)
            metrics["ECE_15"] = float(ece)
        return metrics

    # ------------------------------------------------------------------ #
    # Checkpointing
    # ------------------------------------------------------------------ #
    def _save_checkpoint(self, name: str, *, val_metrics: dict[str, Any] | None = None) -> Path:
        path = self.output_dir / name
        payload = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "ema": self.ema.state_dict(),
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
