"""Optimiser + LR scheduler factories.

* :func:`build_optimizer` -- AdamW with the YAML-defined hyperparameters.
* :class:`WarmupCosineSchedule` -- linear warmup followed by cosine decay to
  ``min_lr_ratio * initial_lr`` over the remaining steps.
* :func:`build_scheduler` -- convenience constructor that reads the trainer YAML.
* :func:`clip_grad_norm` -- thin wrapper around ``torch.nn.utils.clip_grad_norm_``
  that returns the global norm (Python float) for logging.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch.optim.lr_scheduler import LambdaLR


def build_optimizer(
    parameters: Any, opt_cfg: dict[str, Any]
) -> torch.optim.Optimizer:
    """Construct an optimiser from a ``configs/training/*.yaml`` ``optimizer`` block."""
    name = str(opt_cfg.get("name", "adamw")).lower()
    if name != "adamw":
        raise ValueError(f"only 'adamw' is supported in Phase 2E; got {name!r}")
    return torch.optim.AdamW(
        parameters,
        lr=float(opt_cfg.get("lr", 1e-3)),
        weight_decay=float(opt_cfg.get("weight_decay", 1e-4)),
        betas=tuple(opt_cfg.get("betas", (0.9, 0.999))),  # type: ignore[arg-type]
    )


class WarmupCosineSchedule(LambdaLR):
    """Linear warmup for ``warmup_steps`` then half-cosine to ``min_lr_ratio``."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        *,
        warmup_steps: int,
        total_steps: int,
        min_lr_ratio: float = 1e-2,
        last_epoch: int = -1,
    ) -> None:
        if warmup_steps < 0 or total_steps <= 0 or warmup_steps >= total_steps:
            raise ValueError(
                f"need 0 <= warmup_steps < total_steps; got "
                f"warmup={warmup_steps}, total={total_steps}"
            )
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr_ratio = float(min_lr_ratio)
        super().__init__(optimizer, self._lr_lambda, last_epoch=last_epoch)

    def _lr_lambda(self, step: int) -> float:
        if step < self.warmup_steps:
            return float(step + 1) / float(max(1, self.warmup_steps))
        progress = (step - self.warmup_steps) / max(
            1, self.total_steps - self.warmup_steps
        )
        progress = min(max(progress, 0.0), 1.0)
        cos = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.min_lr_ratio + (1.0 - self.min_lr_ratio) * cos


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    sched_cfg: dict[str, Any],
    *,
    total_steps: int,
    steps_per_epoch: int,
) -> WarmupCosineSchedule:
    """Construct a scheduler from a ``configs/training/*.yaml`` ``scheduler`` block."""
    name = str(sched_cfg.get("name", "cosine")).lower()
    if name != "cosine":
        raise ValueError(f"only 'cosine' is supported in Phase 2E; got {name!r}")
    warmup_epochs = int(sched_cfg.get("warmup_epochs", 5))
    warmup_steps = warmup_epochs * steps_per_epoch
    # min_lr is absolute; convert to a *ratio* relative to the optimiser's lr.
    base_lr = optimizer.param_groups[0]["lr"]
    min_lr = float(sched_cfg.get("min_lr", 1e-5))
    min_lr_ratio = max(min_lr / base_lr, 0.0)
    return WarmupCosineSchedule(
        optimizer,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
        min_lr_ratio=min_lr_ratio,
    )


def clip_grad_norm(parameters: Any, max_norm: float) -> float:
    """Clip the global L2 norm; return the *pre-clip* norm as a Python float."""
    total_norm = torch.nn.utils.clip_grad_norm_(parameters, max_norm=max_norm)
    return float(total_norm.detach())
