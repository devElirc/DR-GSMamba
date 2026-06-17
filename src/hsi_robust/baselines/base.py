"""Common abstractions for the Phase-3 baseline suite.

Every deep baseline in :mod:`hsi_robust.baselines` is an :class:`nn.Module`
subclass that implements :meth:`BaselineModel.forward` returning **classification
logits** ``(N, K)`` from a ``(raw_spectrum, pca_patch)`` pair -- the exact same
input interface as :class:`hsi_robust.models.DRGSMamba`. This makes the deep
baselines drop-in replacements at the trainer's data boundary so head-to-head
comparison numbers in Phase 6 are not contaminated by data-pipeline drift.

Shallow baselines (SVM, RF, kNN) live in :mod:`hsi_robust.baselines.shallow`
because they consume features as NumPy arrays and have no notion of epochs.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import torch
from torch import nn


@runtime_checkable
class BaselineModel(Protocol):
    """Static protocol every deep baseline must satisfy.

    The protocol is :func:`runtime_checkable` so the registry's
    ``isinstance(model, BaselineModel)`` check works without forcing every
    subclass to inherit a concrete base.
    """

    num_classes: int

    def forward(self, spectrum: torch.Tensor, patch: torch.Tensor) -> torch.Tensor: ...

    def num_parameters(self) -> int: ...


class BaselineBase(nn.Module):
    """Optional concrete base that satisfies :class:`BaselineModel`.

    Subclasses may override :meth:`forward` only; :meth:`num_parameters` is
    inherited.
    """

    def __init__(self, *, num_classes: int) -> None:
        super().__init__()
        if num_classes <= 0:
            raise ValueError("num_classes must be positive")
        self.num_classes = int(num_classes)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, spectrum: torch.Tensor, patch: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


def cfg_int(cfg: dict[str, Any], key: str, default: int) -> int:
    return int(cfg.get(key, default))


def cfg_float(cfg: dict[str, Any], key: str, default: float) -> float:
    return float(cfg.get(key, default))


def cfg_bool(cfg: dict[str, Any], key: str, default: bool) -> bool:
    return bool(cfg.get(key, default))
