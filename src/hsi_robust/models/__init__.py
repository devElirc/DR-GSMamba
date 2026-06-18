"""Backbone and head modules (Phase 2C).

Public surface:

* :class:`OPS4Block`, :class:`OPS4Encoder`  -- bidirectional spectral SSM.
* :class:`SpatialCNNStem`                   -- compact 2D-CNN on PCA patches.
* :func:`build_cp_graph`, :class:`CPGraphRefinement`
                                              -- in-batch k-NN graph reasoning.
* :class:`FusionMLP`                        -- concat + MLP fusion.
* :class:`EvidentialPrototypeHead`          -- Dirichlet-evidential head.
* :class:`CFAGDRO`                          -- assembled full CFA-GDRO model.
"""

from __future__ import annotations

from hsi_robust.models.cfa_gdro import CFAGDRO
from hsi_robust.models.cp_graph import CPGraphRefinement, build_cp_graph
from hsi_robust.models.evidential_head import EvidentialPrototypeHead
from hsi_robust.models.fusion import FusionMLP
from hsi_robust.models.op_s4 import OPS4Block, OPS4Encoder
from hsi_robust.models.spatial_stem import SpatialCNNStem

__all__ = [
    "CFAGDRO",
    "CPGraphRefinement",
    "EvidentialPrototypeHead",
    "FusionMLP",
    "OPS4Block",
    "OPS4Encoder",
    "SpatialCNNStem",
    "build_cp_graph",
]
