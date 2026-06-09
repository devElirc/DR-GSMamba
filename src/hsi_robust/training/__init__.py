"""Training utilities (Phase 2E).

Public surface:

* :class:`EMAClassLoss`             -- per-class EMA from cfa_gdro.md §4.4.
* :func:`build_optimizer`           -- AdamW factory.
* :class:`WarmupCosineSchedule`     -- LambdaLR with warmup + cosine decay.
* :func:`build_scheduler`           -- scheduler factory.
* :func:`clip_grad_norm`            -- thin wrapper returning the pre-clip norm.
* :class:`TrainConfig`              -- typed trainer config.
* :class:`TrainState`               -- runtime state container.
* :class:`Trainer`                  -- end-to-end train loop.
"""

from __future__ import annotations

from hsi_robust.training.ema_class_loss import EMAClassLoss
from hsi_robust.training.optim import (
    WarmupCosineSchedule,
    build_optimizer,
    build_scheduler,
    clip_grad_norm,
)
from hsi_robust.training.trainer import TrainConfig, Trainer, TrainState

__all__ = [
    "EMAClassLoss",
    "TrainConfig",
    "TrainState",
    "Trainer",
    "WarmupCosineSchedule",
    "build_optimizer",
    "build_scheduler",
    "clip_grad_norm",
]
