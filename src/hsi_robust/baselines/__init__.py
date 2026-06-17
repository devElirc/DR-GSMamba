"""Phase 3 baseline implementations.

Public surface:

* :class:`BaselineBase` -- concrete ``nn.Module`` base that all deep
  baselines extend; it satisfies :class:`BaselineModel`.
* :class:`BaselineModel` -- structural protocol.
* :func:`make_deep_baseline` -- registry-driven factory for the six deep
  baselines (CNN3D, HybridSN, SpectralFormer, SSFTT, NonlocalGCN, MambaHSI).
* :func:`make_shallow_model` -- registry-driven factory for SVM / RF / kNN.
* :class:`BaselineTrainer` -- plain CE trainer used by every deep baseline.
* :class:`BaselineTrainConfig` -- config dataclass for the baseline trainer.
* :func:`deep_baseline_names`, :func:`shallow_baseline_names`,
  :func:`all_baseline_names`, :func:`is_deep_baseline`,
  :func:`is_shallow_baseline` -- registry introspection helpers.

The baselines map cleanly onto the six families in
``EXPERIMENT_PLAN.md`` §"Baselines and protocol":

* Shallow:  ``svm`` / ``rf`` / ``knn``
* CNN:      ``cnn3d`` / ``hybridsn``
* Transformer: ``spectralformer`` / ``ssftt``
* Graph:    ``nonlocal_gcn``
* Mamba:    ``mambahsi``

See ``docs/baselines.md`` for the per-baseline reference + reproduction
gap (populated in Phase 5 when the real-data numbers come in).
"""

from __future__ import annotations

from hsi_robust.baselines.base import BaselineBase, BaselineModel
from hsi_robust.baselines.cnn3d import CNN3D
from hsi_robust.baselines.hybridsn import HybridSN
from hsi_robust.baselines.mambahsi import MambaHSI
from hsi_robust.baselines.nonlocal_gcn import NonlocalGCN
from hsi_robust.baselines.registry import (
    all_baseline_names,
    deep_baseline_names,
    is_deep_baseline,
    is_shallow_baseline,
    make_deep_baseline,
    shallow_baseline_names,
)
from hsi_robust.baselines.shallow import ShallowBaselineModel, make_shallow_model
from hsi_robust.baselines.spectralformer import SpectralFormer
from hsi_robust.baselines.ssftt import SSFTT
from hsi_robust.baselines.trainer import (
    BaselineTrainConfig,
    BaselineTrainer,
    BaselineTrainState,
)

__all__ = [
    "CNN3D",
    "SSFTT",
    "BaselineBase",
    "BaselineModel",
    "BaselineTrainConfig",
    "BaselineTrainState",
    "BaselineTrainer",
    "HybridSN",
    "MambaHSI",
    "NonlocalGCN",
    "ShallowBaselineModel",
    "SpectralFormer",
    "all_baseline_names",
    "deep_baseline_names",
    "is_deep_baseline",
    "is_shallow_baseline",
    "make_deep_baseline",
    "make_shallow_model",
    "shallow_baseline_names",
]
