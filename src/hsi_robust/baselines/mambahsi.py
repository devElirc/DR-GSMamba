"""MambaHSI baseline (Huang et al., 2024, TGRS).

Reference
---------
Huang, L., Chen, Y., & He, X. (2024). "Spectral-spatial Mamba for
hyperspectral image classification." *IEEE Transactions on Geoscience and
Remote Sensing*, 62, 1-14.

This re-implementation re-uses the in-house OP-S4 selective-scan block (which
is a faithful Mamba-style state-space layer with band gating, see
``src/hsi_robust/models/op_s4.py``) on the raw spectrum, plus a small CNN on
the PCA patch, fused into a single per-pixel logit vector. We re-use OP-S4
rather than pull a third-party Mamba kernel for two reasons:

* Reproducibility: the trainer already certifies OP-S4 to 1e-5 vs its
  recurrent reference, so the baseline is exactly the same SSM the main
  model uses (the MambaHSI vs DR-GSMamba comparison then isolates the
  loss-level contribution).
* Anonymisation: the Phase 9 submission must compile from this repo alone;
  re-using OP-S4 keeps the dependency surface clean.

Reproduction gap (vs the published MambaHSI) will be recorded in the Phase 5
sanity report.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from hsi_robust.baselines.base import BaselineBase
from hsi_robust.models.op_s4 import OPS4Encoder
from hsi_robust.models.spatial_stem import SpatialCNNStem


class MambaHSI(BaselineBase):
    """Spectral SSM (OP-S4) + spatial CNN + linear classifier."""

    def __init__(
        self,
        *,
        num_bands: int,
        num_pca: int,
        patch_size: int,
        num_classes: int,
        spectral_dim: int = 64,
        spatial_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__(num_classes=num_classes)
        self.op_s4 = OPS4Encoder(
            num_bands=num_bands,
            d_model=spectral_dim,
            d_state=16,
            num_layers=num_layers,
            out_dim=spectral_dim,
            bidirectional=True,
            dropout=dropout,
            use_hippo_init=True,
            use_band_gate=True,
        )
        self.spatial = SpatialCNNStem(
            in_channels=num_pca,
            patch_size=patch_size,
            out_dim=spatial_dim,
            base_channels=max(8, spatial_dim // 4),
            norm_type="gn",
            dropout=0.0,
        )
        fused_dim = spectral_dim + spatial_dim
        self.fuse = nn.Sequential(
            nn.Linear(fused_dim, fused_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(fused_dim, num_classes)

    def forward(self, spectrum: torch.Tensor, patch: torch.Tensor) -> torch.Tensor:
        f_spec = self.op_s4(spectrum)
        f_spat = self.spatial(patch)
        fused = self.fuse(torch.cat([f_spec, f_spat], dim=-1))
        return self.classifier(fused)

    @classmethod
    def from_config(
        cls,
        config: dict[str, Any],
        *,
        num_bands: int,
        num_pca: int,
        patch_size: int,
        num_classes: int,
    ) -> MambaHSI:
        return cls(
            num_bands=num_bands,
            num_pca=num_pca,
            patch_size=patch_size,
            num_classes=num_classes,
            spectral_dim=int(config.get("spectral_dim", 64)),
            spatial_dim=int(config.get("spatial_dim", 64)),
            num_layers=int(config.get("num_layers", 2)),
            dropout=float(config.get("dropout", 0.1)),
        )
