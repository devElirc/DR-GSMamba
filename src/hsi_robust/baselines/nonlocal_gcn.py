"""Nonlocal-GCN baseline (Wan et al., 2020-style intra-patch graph reasoning).

Reference
---------
Wan, S., Gong, C., Zhong, P., Du, B., Zhang, L., & Yang, J. (2020).
"Multiscale dynamic graph convolutional network for hyperspectral image
classification." *IEEE Transactions on Geoscience and Remote Sensing*,
58(5), 3162-3177.

This compact re-implementation treats every spatial position of the PCA patch
as a graph node, builds a fully-connected, self-attention style affinity
matrix (the "non-local" block), and stacks two GCN-like updates with
softmax-normalised edge weights and a residual identity connection. The
classifier reads the central-pixel node feature -- the standard
``per-pixel-classification`` setup for HSI.

We use ``LayerNorm`` rather than the original BN; in label-scarce HSI the BN
running statistics are unreliable (decision D-09 of ``roadmap.md``).
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from hsi_robust.baselines.base import BaselineBase


class _NonlocalGCNLayer(nn.Module):
    """One non-local graph propagation step with a learnable affinity head."""

    def __init__(self, dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.theta = nn.Linear(dim, dim, bias=False)
        self.phi = nn.Linear(dim, dim, bias=False)
        self.g = nn.Linear(dim, dim, bias=False)
        self.out = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, T, dim)
        q = self.theta(x)
        k = self.phi(x)
        v = self.g(x)
        # Affinity in (N, T, T) -- temperature is sqrt(dim) per attention norm.
        scale = 1.0 / max(1.0, x.shape[-1] ** 0.5)
        attn = torch.softmax(q @ k.transpose(1, 2) * scale, dim=-1)
        propagated = attn @ v
        propagated = self.dropout(self.act(self.out(propagated)))
        return self.norm(x + propagated)


class NonlocalGCN(BaselineBase):
    """Two-layer nonlocal-GCN on flattened PCA-patch tokens."""

    def __init__(
        self,
        *,
        num_pca: int,
        patch_size: int,
        num_classes: int,
        dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__(num_classes=num_classes)
        if patch_size <= 0 or num_pca <= 0:
            raise ValueError("patch_size and num_pca must be positive")
        self.patch_size = int(patch_size)
        self.num_pca = int(num_pca)

        self.token_proj = nn.Linear(num_pca, dim)
        self.layers = nn.ModuleList(
            [_NonlocalGCNLayer(dim=dim, dropout=dropout) for _ in range(num_layers)]
        )
        self.classifier = nn.Linear(dim, num_classes)

    def forward(self, spectrum: torch.Tensor, patch: torch.Tensor) -> torch.Tensor:
        _ = spectrum
        if patch.ndim != 4 or patch.shape[1] != self.num_pca:
            raise ValueError(f"patch must be (N, {self.num_pca}, P, P); got {tuple(patch.shape)}")
        n, c, h, w = patch.shape
        # Reshape (N, C, H, W) -> (N, T=H*W, C).
        tokens = patch.permute(0, 2, 3, 1).reshape(n, h * w, c)
        tokens = self.token_proj(tokens)
        for layer in self.layers:
            tokens = layer(tokens)
        # Central token corresponds to the labelled pixel.
        centre = (h // 2) * w + (w // 2)
        y = tokens[:, centre, :]
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
    ) -> NonlocalGCN:
        _ = num_bands
        return cls(
            num_pca=num_pca,
            patch_size=patch_size,
            num_classes=num_classes,
            dim=int(config.get("dim", 64)),
            num_layers=int(config.get("num_layers", 2)),
            dropout=float(config.get("dropout", 0.1)),
        )
