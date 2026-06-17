"""Registry mapping baseline names to factories.

Two separate registries because the shallow baselines run a different code
path (no epochs, no torch optimiser) from the deep baselines.

``make_deep_baseline(name, config, ...)`` returns an ``nn.Module``;
``make_shallow_model(name, config, seed)`` returns a sklearn wrapper.
"""

from __future__ import annotations

from typing import Any

from hsi_robust.baselines.base import BaselineBase
from hsi_robust.baselines.cnn3d import CNN3D
from hsi_robust.baselines.hybridsn import HybridSN
from hsi_robust.baselines.mambahsi import MambaHSI
from hsi_robust.baselines.nonlocal_gcn import NonlocalGCN
from hsi_robust.baselines.spectralformer import SpectralFormer
from hsi_robust.baselines.ssftt import SSFTT

_DEEP_REGISTRY: dict[str, type[BaselineBase]] = {
    "cnn3d": CNN3D,
    "hybridsn": HybridSN,
    "spectralformer": SpectralFormer,
    "ssftt": SSFTT,
    "nonlocal_gcn": NonlocalGCN,
    "mambahsi": MambaHSI,
}

_SHALLOW_NAMES: tuple[str, ...] = ("svm", "rf", "knn")


def deep_baseline_names() -> tuple[str, ...]:
    return tuple(_DEEP_REGISTRY)


def shallow_baseline_names() -> tuple[str, ...]:
    return _SHALLOW_NAMES


def all_baseline_names() -> tuple[str, ...]:
    return _SHALLOW_NAMES + tuple(_DEEP_REGISTRY)


def is_shallow_baseline(name: str) -> bool:
    return name in _SHALLOW_NAMES


def is_deep_baseline(name: str) -> bool:
    return name in _DEEP_REGISTRY


def make_deep_baseline(
    name: str,
    config: dict[str, Any],
    *,
    num_bands: int,
    num_pca: int,
    patch_size: int,
    num_classes: int,
) -> BaselineBase:
    if name not in _DEEP_REGISTRY:
        raise ValueError(f"unknown deep baseline '{name}'; choose from {deep_baseline_names()}")
    cls = _DEEP_REGISTRY[name]
    return cls.from_config(  # type: ignore[attr-defined]
        config,
        num_bands=num_bands,
        num_pca=num_pca,
        patch_size=patch_size,
        num_classes=num_classes,
    )
