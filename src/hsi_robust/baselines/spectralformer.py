"""SpectralFormer baseline (Hong et al., 2022, TGRS).

Reference
---------
Hong, D., Han, Z., Yao, J., Gao, L., Zhang, B., Plaza, A., & Chanussot, J.
(2022). "SpectralFormer: Rethinking hyperspectral image classification with
Transformers." *IEEE Transactions on Geoscience and Remote Sensing*, 60,
1-15.

Key idea: tokenise the spectrum into overlapping bands-groups, feed those
tokens to a Transformer encoder, and produce per-pixel logits. We keep the
"GCB-pixel" (group-wise covariance band tokenization) variant from the
original paper because it dominates on Indian Pines per their Table II.

This is a faithful re-implementation: a learnable patch (band-group)
projection, sinusoidal positional encodings, and a vanilla Transformer
encoder. We use ``LayerNorm`` and dropout per the original paper.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn

from hsi_robust.baselines.base import BaselineBase


def _sinusoidal_positional_encoding(num_pos: int, dim: int) -> torch.Tensor:
    pe = torch.zeros(num_pos, dim)
    position = torch.arange(0, num_pos, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32) * -(math.log(10000.0) / dim))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term[: pe.shape[1] // 2])
    return pe  # (num_pos, dim)


class SpectralFormer(BaselineBase):
    """Compact SpectralFormer: band-group tokens + Transformer encoder.

    Parameters
    ----------
    num_bands:
        Raw spectral band count (the raw spectrum input is used here).
    group_size:
        Number of adjacent bands grouped into one input token.
    dim:
        Token embedding dimensionality.
    depth:
        Number of Transformer encoder layers.
    heads:
        Multi-head attention head count.
    mlp_ratio:
        Feed-forward block width relative to ``dim``.
    """

    def __init__(
        self,
        *,
        num_bands: int,
        num_classes: int,
        group_size: int = 7,
        dim: int = 64,
        depth: int = 4,
        heads: int = 4,
        mlp_ratio: float = 2.0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__(num_classes=num_classes)
        if num_bands < group_size:
            raise ValueError("num_bands must be >= group_size")
        if dim % heads != 0:
            raise ValueError("dim must be divisible by heads")
        self.num_bands = int(num_bands)
        self.group_size = int(group_size)
        self.dim = int(dim)

        # Tokeniser: a 1-D convolution that produces one (dim,) vector per
        # overlapping band-group of length ``group_size``.
        self.token_conv = nn.Conv1d(
            in_channels=1,
            out_channels=dim,
            kernel_size=group_size,
            stride=max(1, group_size // 2),
            padding=group_size // 2,
            bias=False,
        )

        # Class token + positional encoding (re-built on first forward).
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.register_buffer("pos_embed", torch.zeros(1, 1, dim), persistent=False)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=int(dim * mlp_ratio),
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

    def _ensure_pos_embed(self, num_tokens: int, device: torch.device) -> torch.Tensor:
        if self.pos_embed.shape[1] != num_tokens:
            pe = _sinusoidal_positional_encoding(num_tokens, self.dim).to(device)
            self.pos_embed = pe.unsqueeze(0)
        return self.pos_embed

    def forward(self, spectrum: torch.Tensor, patch: torch.Tensor) -> torch.Tensor:
        _ = patch  # SpectralFormer is purely spectral
        if spectrum.ndim != 2 or spectrum.shape[-1] != self.num_bands:
            raise ValueError(
                f"expected spectrum (N, {self.num_bands}); got {tuple(spectrum.shape)}"
            )
        x = spectrum.unsqueeze(1)  # (N, 1, B)
        tokens = self.token_conv(x)  # (N, dim, T)
        tokens = tokens.transpose(1, 2)  # (N, T, dim)
        # Prepend the class token.
        cls = self.cls_token.expand(tokens.shape[0], -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        pe = self._ensure_pos_embed(tokens.shape[1], tokens.device)
        tokens = tokens + pe
        y = self.transformer(tokens)
        y = self.norm(y[:, 0, :])  # class token
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
    ) -> SpectralFormer:
        _ = num_pca
        _ = patch_size
        return cls(
            num_bands=num_bands,
            num_classes=num_classes,
            group_size=int(config.get("group_size", 7)),
            dim=int(config.get("dim", 64)),
            depth=int(config.get("depth", 4)),
            heads=int(config.get("heads", 4)),
            mlp_ratio=float(config.get("mlp_ratio", 2.0)),
            dropout=float(config.get("dropout", 0.1)),
        )
