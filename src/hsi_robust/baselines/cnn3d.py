"""3D-CNN baseline (Chen et al., 2016, TGRS).

Reference
---------
Chen, Y., Jiang, H., Li, C., Jia, X., & Ghamisi, P. (2016). "Deep feature
extraction and classification of hyperspectral images based on convolutional
neural networks." *IEEE Transactions on Geoscience and Remote Sensing*,
54(10), 6232-6251.

This is a compact re-implementation: three 3D convolution blocks followed by
adaptive pooling and a linear classifier. We deliberately keep it small so it
fits the same parameter budget as the main model -- the comparison should
isolate the loss-level contribution, not the depth advantage.

The model operates on the **PCA patch** ``(N, C_pca, P, P)``; the raw spectrum
input is ignored. We unsqueeze a singleton "depth" channel so PyTorch's
:class:`nn.Conv3d` consumes the spectral axis as time.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from hsi_robust.baselines.base import BaselineBase


def _conv3d_block(in_ch: int, out_ch: int, kernel: tuple[int, int, int]) -> nn.Sequential:
    pad = tuple(k // 2 for k in kernel)
    return nn.Sequential(
        nn.Conv3d(in_ch, out_ch, kernel_size=kernel, padding=pad, bias=False),
        nn.GroupNorm(num_groups=min(8, out_ch), num_channels=out_ch),
        nn.GELU(),
    )


class CNN3D(BaselineBase):
    """Three-block 3D CNN with adaptive pooling head.

    Parameters
    ----------
    num_pca:
        Spectral depth after PCA reduction (used as the depth axis of the
        3D conv tensor).
    patch_size:
        Spatial side length P of the input patch.
    num_classes:
        Output class count.
    channels:
        Width schedule for the three 3D conv blocks (default ``(8, 16, 32)``).
    """

    def __init__(
        self,
        *,
        num_pca: int,
        patch_size: int,
        num_classes: int,
        channels: tuple[int, int, int] = (8, 16, 32),
        dropout: float = 0.4,
    ) -> None:
        super().__init__(num_classes=num_classes)
        if patch_size <= 0:
            raise ValueError("patch_size must be positive")
        c1, c2, c3 = channels
        self.block1 = _conv3d_block(1, c1, kernel=(3, 3, 3))
        self.block2 = _conv3d_block(c1, c2, kernel=(3, 3, 3))
        self.block3 = _conv3d_block(c2, c3, kernel=(3, 3, 3))
        # Global pool over (D, H, W) -> (N, c3).
        self.gap = nn.AdaptiveAvgPool3d(1)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(c3, num_classes)

    def forward(self, spectrum: torch.Tensor, patch: torch.Tensor) -> torch.Tensor:
        # patch: (N, C_pca, P, P). Treat C_pca as spectral depth.
        _ = spectrum  # unused
        if patch.ndim != 4:
            raise ValueError(f"patch must be 4-D; got {tuple(patch.shape)}")
        x = patch.unsqueeze(1)  # (N, 1, C_pca, P, P)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.gap(x).flatten(1)
        x = self.dropout(x)
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
    ) -> CNN3D:
        _ = num_bands  # spectrum unused
        channels_cfg = config.get("channels") or [8, 16, 32]
        channels = (int(channels_cfg[0]), int(channels_cfg[1]), int(channels_cfg[2]))
        return cls(
            num_pca=num_pca,
            patch_size=patch_size,
            num_classes=num_classes,
            channels=channels,
            dropout=float(config.get("dropout", 0.4)),
        )
