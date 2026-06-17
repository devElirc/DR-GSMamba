"""SSFTT baseline (Sun et al., 2022, TGRS).

Reference
---------
Sun, L., Zhao, G., Zheng, Y., & Wu, Z. (2022). "Spectral--spatial feature
tokenization Transformer for hyperspectral image classification." *IEEE
Transactions on Geoscience and Remote Sensing*, 60, 1-14.

Architecture (compact re-implementation):

    PCA patch (N, C_pca, P, P)
        -> 3D conv block (1 -> 8, kernel=(3,3,3))
        -> 2D conv block (-> 64, kernel=(3,3))
        -> flatten spatial -> tokens (N, T, dim)
        -> Gaussian weighted token initialiser (semantic tokens)
        -> Transformer encoder (depth=1, heads=4)
        -> class token classifier

We use ``GroupNorm`` instead of BatchNorm for the same label-scarce reasoning
recorded in decision D-09 of ``roadmap.md``.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from hsi_robust.baselines.base import BaselineBase


class _GaussianTokenizer(nn.Module):
    """Soft assignment of feature-map pixels to ``num_tokens`` semantic groups."""

    def __init__(self, in_dim: int, num_tokens: int) -> None:
        super().__init__()
        self.assign = nn.Linear(in_dim, num_tokens)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, in_dim, H*W) -- channels-as-features, spatial-as-tokens.
        feats = x.transpose(1, 2)  # (N, H*W, in_dim)
        weights = torch.softmax(self.assign(feats), dim=1)  # (N, H*W, T)
        # Aggregate spatial features into T tokens by Gaussian-weighted sum.
        tokens = torch.einsum("nij,nik->nkj", feats, weights)  # (N, T, in_dim)
        return tokens


class SSFTT(BaselineBase):
    """Sun et al. 2022 SSFTT (faithful, GroupNorm variant)."""

    def __init__(
        self,
        *,
        num_pca: int,
        patch_size: int,
        num_classes: int,
        dim: int = 64,
        num_tokens: int = 4,
        depth: int = 1,
        heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__(num_classes=num_classes)
        if dim % heads != 0:
            raise ValueError("dim must be divisible by heads")
        if patch_size <= 0 or num_pca <= 0:
            raise ValueError("patch_size and num_pca must be positive")

        self.conv3d = nn.Conv3d(1, 8, kernel_size=(3, 3, 3), padding=(1, 1, 1), bias=False)
        self.gn3d = nn.GroupNorm(num_groups=min(8, 8), num_channels=8)
        # After 3D conv: shape (N, 8, C_pca, P, P) -- depth/PCA preserved by padding.
        self.conv2d = nn.Conv2d(8 * num_pca, dim, kernel_size=3, padding=1, bias=False)
        self.gn2d = nn.GroupNorm(num_groups=min(8, dim), num_channels=dim)
        self.act = nn.GELU()

        self.tokenizer = _GaussianTokenizer(in_dim=dim, num_tokens=num_tokens)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=dim * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=depth, enable_nested_tensor=False
        )
        self.norm = nn.LayerNorm(dim)
        self.classifier = nn.Linear(dim, num_classes)

    def forward(self, spectrum: torch.Tensor, patch: torch.Tensor) -> torch.Tensor:
        _ = spectrum
        if patch.ndim != 4:
            raise ValueError(f"patch must be 4-D; got {tuple(patch.shape)}")
        x = patch.unsqueeze(1)  # (N, 1, C, P, P)
        x = self.act(self.gn3d(self.conv3d(x)))
        # Collapse (channel, depth) into the 2D conv input.
        n, c, d, h, w = x.shape
        x = x.reshape(n, c * d, h, w)
        x = self.act(self.gn2d(self.conv2d(x)))
        # Flatten spatial -> token grid: (N, dim, H*W).
        n, dim, h, w = x.shape
        tokens = x.reshape(n, dim, h * w)
        sem_tokens = self.tokenizer(tokens)  # (N, T, dim)
        cls = self.cls_token.expand(n, -1, -1)
        sequence = torch.cat([cls, sem_tokens], dim=1)
        sequence = self.transformer(sequence)
        y = self.norm(sequence[:, 0, :])
        return self.classifier(y)

    @classmethod
    def from_config(
        cls,
        config: dict[str, Any],
        *,
        num_bands: int,
        num_pca: int,
        patch_size: int,
        num_classes: int,
    ) -> SSFTT:
        _ = num_bands
        return cls(
            num_pca=num_pca,
            patch_size=patch_size,
            num_classes=num_classes,
            dim=int(config.get("dim", 64)),
            num_tokens=int(config.get("num_tokens", 4)),
            depth=int(config.get("depth", 1)),
            heads=int(config.get("heads", 4)),
            dropout=float(config.get("dropout", 0.1)),
        )
