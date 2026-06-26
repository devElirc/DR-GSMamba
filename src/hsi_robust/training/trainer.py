"""End-to-end trainer for the CFA-GDRO model (backbone + EPH + CP-Graph).

The total loss assembled here depends on ``TrainConfig.loss_name``:

* ``"cfa_gdro"`` (default, the paper's contribution) follows Eq. (4) of
  ``docs/math/cfa_gdro.md`` §3 exactly::

      L_total = mean_CE
                + lambda_rob   * L_CFA-GDRO   (operates on per-sample CE)
                + lambda_evi   * L_EPH         (Bayes-risk + annealed KL)
                + lambda_graph * L_CP-graph

  The ``mean_CE`` anchor is required for stable optimisation: the EPH
  Bayes-risk loss has a vanishing gradient at the uniform-prediction saddle
  (Phase 2E smoke debug, decision D-08).

* ``"ce"``, ``"focal"``, ``"sample_cvar"``, ``"sagawa_group_dro"`` swap the
  *robust* objective for the named baseline while keeping the **same backbone
  + same EPH-prototype classifier head + same data + same seed grid** (the
  internal Phase-5 M5.2 ablation Kass asked for in his 2026-06-18 feedback,
  D-14). EPH / CP-Graph auxiliaries are disabled in those rows so the
  comparison is apples-to-apples on the loss alone.

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
from hsi_robust.eval.temperature_scaling import evaluate_with_temperature
from hsi_robust.losses import (
    cfa_gdro_loss,
    cp_graph_loss,
    evidential_loss,
    focal_loss,
    sagawa_group_dro_loss,
    sample_cvar_loss,
)
from hsi_robust.models import CFAGDRO
from hsi_robust.training.ema_class_loss import EMAClassLoss
from hsi_robust.training.optim import build_optimizer, build_scheduler, clip_grad_norm

SUPPORTED_LOSSES = ("ce", "focal", "sample_cvar", "sagawa_group_dro", "cfa_gdro")


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

    # Which training objective to use. Drives `_step` branching.
    # One of: "ce" | "focal" | "sample_cvar" | "sagawa_group_dro" | "cfa_gdro".
    loss_name: str = "cfa_gdro"

    # Baseline-loss-specific knobs (only consulted when loss_name selects them).
    focal_gamma: float = 2.0
    cvar_alpha: float = 0.3
    gdro_eta: float = 0.01

    # Loss-mixing weights for the full CFA-GDRO stack (cfa_gdro.md §3 Eq. (4)).
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

    # Calibration. When True, the trainer additionally reports the
    # temperature-scaled ECE on the held-out report slice of the test set
    # (Guo et al., 2017). Costs O(K * N_test) at the end of training.
    temperature_scaling: bool = True
    temperature_calib_frac: float = 0.2

    # Bookkeeping.
    log_every: int = 50
    val_every: int = 1
    save_every: int = 50

    def __post_init__(self) -> None:
        if self.loss_name not in SUPPORTED_LOSSES:
            raise ValueError(
                f"unsupported loss_name {self.loss_name!r}; expected one of {SUPPORTED_LOSSES}"
            )

    @classmethod
    def from_yaml_dict(cls, training_cfg: dict[str, Any]) -> TrainConfig:
        """Build from a parsed ``configs/training/*.yaml`` dict."""
        loss = training_cfg.get("loss") or {}
        cfa = loss.get("cfa_gdro") or {}
        evi = loss.get("evidential") or {}
        cpg = loss.get("cp_graph") or {}
        foc = loss.get("focal") or {}
        cvar = loss.get("sample_cvar") or {}
        gdro = loss.get("sagawa_group_dro") or {}
        ts = training_cfg.get("temperature_scaling") or {}
        return cls(
            epochs=int(training_cfg.get("epochs", 200)),
            batch_size=int(training_cfg.get("batch_size", 64)),
            num_workers=int(training_cfg.get("num_workers", 0)),
            pin_memory=bool(training_cfg.get("pin_memory", False)),
            optimizer=dict(training_cfg.get("optimizer") or {}),
            scheduler=dict(training_cfg.get("scheduler") or {}),
            grad_clip=float(training_cfg.get("grad_clip", 1.0)),
            loss_name=str(loss.get("name", "cfa_gdro")),
            focal_gamma=float(foc.get("gamma", 2.0)),
            cvar_alpha=float(cvar.get("alpha", 0.3)),
            gdro_eta=float(gdro.get("eta", 0.01)),
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
            temperature_scaling=bool(ts.get("enabled", True)),
            temperature_calib_frac=float(ts.get("calib_frac", 0.2)),
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

        # Sagawa group-DRO carries a state vector q (one entry per class) that
        # persists across batches. Initialised lazily on the first step.
        self._gdro_q: torch.Tensor | None = None

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
        """One forward + loss assembly. Branches on ``config.loss_name``.

        Independent of which loss is selected, every branch shares the
        backbone (OP-S4 + spatial CNN + fusion) **and** the EPH-prototype
        classifier head (``logits = tau * cos``). This guarantees the
        internal Phase-5 M5.2 ablation (D-14) really is "same backbone +
        different loss".
        """
        out = self.model(spec, patch)
        alpha = out["alpha"]
        probs = out["probs"]
        fused = out["fused_feat"]
        cos = out["cos"]
        tau = out["temperature"]

        # Shared inputs for all loss branches.
        logits = tau * cos
        per_sample_ce = F.cross_entropy(logits, labels, reduction="none")
        mean_ce = per_sample_ce.mean()

        scalars: dict[str, Any] = {
            "loss/mean_ce": float(mean_ce.detach()),
            "eph/temperature": float(tau.detach()),
        }

        name = self.config.loss_name
        if name == "ce":
            total = mean_ce
        elif name == "focal":
            _, mean_focal = focal_loss(logits, labels, gamma_focal=self.config.focal_gamma)
            total = mean_focal
            scalars["loss/focal"] = float(mean_focal.detach())
        elif name == "sample_cvar":
            cvar_loss_val, cvar_info = sample_cvar_loss(per_sample_ce, alpha=self.config.cvar_alpha)
            total = cvar_loss_val
            scalars["loss/sample_cvar"] = float(cvar_loss_val.detach())
            scalars["cvar/threshold"] = float(cvar_info["threshold"])
            scalars["cvar/num_kept"] = int(cvar_info["num_kept"])
        elif name == "sagawa_group_dro":
            gdro_loss_val, q_new, gdro_info = sagawa_group_dro_loss(
                per_sample_ce,
                labels,
                num_classes=self.model.num_classes,
                q_state=self._gdro_q,
                eta=self.config.gdro_eta,
            )
            self._gdro_q = q_new
            total = gdro_loss_val
            scalars["loss/sagawa_group_dro"] = float(gdro_loss_val.detach())
            scalars["gdro/active_set_size"] = int(gdro_info["active_set_size"])
            scalars["gdro/q_max"] = float(q_new.max().item())
        else:  # cfa_gdro (full stack -- Eq. (4) of cfa_gdro.md §3)
            _, mean_eph, evi_info = evidential_loss(alpha, labels, kl_weight=kl_weight)
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
            scalars.update(
                {
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
                }
            )

        scalars["loss/total"] = float(total.detach())
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
    def evaluate(self, *, return_logits: bool = False) -> dict[str, Any]:
        """Run inference on the val/test set and return metrics.

        When ``return_logits`` is True, the returned dict also includes
        ``_logits`` and ``_labels`` numpy arrays so a caller can fit
        post-hoc temperature scaling (Guo et al., 2017). The logits are
        ``tau * cos`` matching what the model would feed into a softmax for
        prediction.
        """
        self.model.eval()
        all_probs: list[np.ndarray] = []
        all_logits: list[np.ndarray] = []
        all_preds: list[np.ndarray] = []
        all_labels: list[np.ndarray] = []
        all_vacuity: list[np.ndarray] = []
        all_aleatoric: list[np.ndarray] = []
        for batch in self.val_loader:
            spec, patch, labels = _move_batch(batch, self.device)
            out = self.model(spec, patch)
            probs = out["probs"].detach().cpu().numpy()
            logits = (out["temperature"] * out["cos"]).detach().cpu().numpy()
            preds = probs.argmax(axis=1)
            all_probs.append(probs)
            all_logits.append(logits)
            all_preds.append(preds)
            all_labels.append(labels.detach().cpu().numpy())
            all_vacuity.append(out["vacuity"].detach().cpu().numpy())
            all_aleatoric.append(out["aleatoric"].detach().cpu().numpy())

        probs_arr = (
            np.concatenate(all_probs, axis=0)
            if all_probs
            else np.zeros((0, self.model.num_classes))
        )
        logits_arr = (
            np.concatenate(all_logits, axis=0)
            if all_logits
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
        if return_logits:
            metrics["_logits"] = logits_arr
            metrics["_labels"] = labels_arr
        return metrics

    def evaluate_with_calibration(self) -> dict[str, Any]:
        """Final evaluation that also reports the temperature-scaled ECE.

        This is the metric set the internal Phase-5 M5.2 ablation table prints
        (D-14). The temperature is fit on a deterministic
        ``temperature_calib_frac`` slice of the test set; the raw and scaled
        ECE values are both reported on the remaining slice, so the comparison
        does not double-count.
        """
        metrics = self.evaluate(return_logits=True)
        logits = metrics.pop("_logits")
        labels = metrics.pop("_labels")
        if self.config.temperature_scaling and labels.shape[0] >= 2:
            ts = evaluate_with_temperature(
                logits,
                labels,
                num_bins=15,
                calib_frac=self.config.temperature_calib_frac,
            )
            metrics["temperature"] = ts["T"]
            metrics["ECE_15_report"] = ts["ECE_15_raw"]
            metrics["ECE_15_T"] = ts["ECE_15_T"]
            metrics["NLL_report"] = ts["NLL_raw"]
            metrics["NLL_T"] = ts["NLL_T"]
            metrics["temperature_n_calib"] = ts["n_calib"]
            metrics["temperature_n_report"] = ts["n_report"]
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
