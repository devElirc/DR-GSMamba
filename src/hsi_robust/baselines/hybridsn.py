"""HybridSN baseline (Roy et al., 2020, GRSL).

Reference
---------
Roy, S. K., Krishna, G., Dubey, S. R., & Chaudhuri, B. B. (2020). "HybridSN:
Exploring 3-D--2-D CNN feature hierarchy for hyperspectral image
classification." *IEEE Geoscience and Remote Sensing Letters*, 17(2),
277-281.

The published architecture is:

    Input: (N, 1, C, P, P)   (with PCA reducing C to 30)
      Conv3D(1 -> 8,  kernel=(7,3,3))
      Conv3D(8 -> 16, kernel=(5,3,3))
      Conv3D(16 -> 32, kernel=(3,3,3))
      Reshape to (N, 32 * C', P', P')
      Conv2D(32 * C' -> 64, kernel=3)
      Flatten + Linear(-> 256) -> Dropout -> Linear(-> 128) -> Dropout
      Linear(-> num_classes)

We use ``GroupNorm`` instead of the original BN for the same label-scarce
reason recorded in decision D-09 of ``roadmap.md`` (BatchNorm running stats
drift on 5-spc training). The original publication used ReLU; we keep ReLU to
stay faithful to the head architecture but switch the norm.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from hsi_robust.baselines.base import BaselineBase


def _gn(num_channels: int) -> nn.GroupNorm:
    return nn.GroupNorm(num_groups=min(8, num_channels), num_channels=num_channels)


class HybridSN(BaselineBase):
    """Roy et al. 2020 HybridSN, GroupNorm variant for label-scarce regimes."""

    def __init__(
        self,
        *,
        num_pca: int,
        patch_size: int,
        num_classes: int,
        dropout: float = 0.4,
    ) -> None:
        super().__init__(num_classes=num_classes)
        if patch_size <= 0:
            raise ValueError("patch_size must be positive")
        if num_pca < 7:
            raise ValueError("num_pca must be >= 7 for HybridSN's (7,3,3) 3D-conv")

        self.conv1 = nn.Conv3d(1, 8, kernel_size=(7, 3, 3), padding=0, bias=False)
        self.gn1 = _gn(8)
        self.conv2 = nn.Conv3d(8, 16, kernel_size=(5, 3, 3), padding=0, bias=False)
        self.gn2 = _gn(16)
        self.conv3 = nn.Conv3d(16, 32, kernel_size=(3, 3, 3), padding=0, bias=False)
        self.gn3 = _gn(32)

        # Derive shapes from input sizes so the model adapts to (P, num_pca).
        d1 = num_pca - 7 + 1
        d2 = d1 - 5 + 1
        d3 = d2 - 3 + 1
        p1 = patch_size - 3 + 1
        p2 = p1 - 3 + 1
        p3 = p2 - 3 + 1
        if d3 <= 0 or p3 <= 0:
            raise ValueError(f"input too small for HybridSN: derived (D, P) = ({d3}, {p3})")

        flat_2d_in = 32 * d3
        self.conv2d = nn.Conv2d(flat_2d_in, 64, kernel_size=3, padding=0, bias=False)
        self.gn2d = nn.GroupNorm(num_groups=8, num_channels=64)
        # After 2D conv: spatial shrinks by 2 -> (p3 - 2)^2.
        p4 = p3 - 2
        if p4 <= 0:
            raise ValueError(f"patch too small after 2-D conv: derived P = {p4}")
        fc_in = 64 * p4 * p4
        self.fc1 = nn.Linear(fc_in, 256)
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(256, 128)
        self.dropout2 = nn.Dropout(dropout)
        self.classifier = nn.Linear(128, num_classes)
        self.act = nn.ReLU(inplace=True)

    def forward(self, spectrum: torch.Tensor, patch: torch.Tensor) -> torch.Tensor:
        _ = spectrum
        if patch.ndim != 4:
            raise ValueError(f"patch must be 4-D; got {tuple(patch.shape)}")
        x = patch.unsqueeze(1)
        x = self.act(self.gn1(self.conv1(x)))
        x = self.act(self.gn2(self.conv2(x)))
        x = self.act(self.gn3(self.conv3(x)))
        # Collapse (channel, depth) into one feature axis for the 2D conv stage.
        n, c, d, h, w = x.shape
        x = x.reshape(n, c * d, h, w)
        x = self.act(self.gn2d(self.conv2d(x)))
        x = x.flatten(1)
        x = self.act(self.fc1(x))
        x = self.dropout1(x)
        x = self.act(self.fc2(x))
        x = self.dropout2(x)
        return self.classifier(x)

    @classmethod
    def from_config(
        cls,
        config: dict[str, Any],
        *,
        num_bands: int,
        num_pca: int,
        patch_size: int,
        num_classes: int,
    ) -> HybridSN:
        _ = num_bands
        return cls(
            num_pca=num_pca,
            patch_size=patch_size,
            num_classes=num_classes,
            dropout=float(config.get("dropout", 0.4)),
        )
