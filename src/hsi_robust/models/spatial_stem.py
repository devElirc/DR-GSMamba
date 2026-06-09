"""Compact 2D CNN stem for PCA-reduced HSI patches.

This is a **design choice, not a contribution**. It plays the role of the
spatial pathway in the dual-stream architecture, working on a ``patch_size x
patch_size`` window around each labelled pixel after PCA-reducing the spectral
axis to ``pca_components`` channels.

Architecture (kept intentionally compact to fit the 5 M-parameter budget):

    PCA-patch (N, C_in, P, P)
        -> Conv3x3 + GN + GELU -> Conv3x3 + GN + GELU       (block 1: C_in -> 32)
        -> AvgPool2x2 (ceil_mode for odd patches)
        -> Conv3x3 + GN + GELU -> Conv3x3 + GN + GELU       (block 2: 32 -> 64)
        -> AdaptiveAvgPool to 1x1
        -> Linear (64 -> out_dim)

We use :class:`torch.nn.GroupNorm` (not BatchNorm) because the label-scarce
regime has at most ``5 * num_classes`` training samples, so BN running
statistics drift from the test distribution at eval time and collapse OA.
GroupNorm normalises per-sample and side-steps that pathology.
"""

from __future__ import annotations

import torch
from torch import nn


def _group_count(out_ch: int) -> int:
    """Pick a GN group count that always evenly divides ``out_ch``.

    Aims for a group size of roughly 8 channels; falls back to 1 group
    (LayerNorm-equivalent on the channel axis) when ``out_ch < 8``.
    """
    for g in (8, 4, 2, 1):
        if out_ch >= g and out_ch % g == 0:
            return g
    return 1


class _ConvNormAct(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=kernel, padding=kernel // 2, bias=False)
        self.norm = nn.GroupNorm(num_groups=_group_count(out_ch), num_channels=out_ch)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class SpatialCNNStem(nn.Module):
    """2-stage 2D CNN with global average pooling.

    Parameters
    ----------
    in_channels:
        Number of input channels = ``pca_components`` after PCA reduction.
    patch_size:
        Spatial side length ``P`` of the input patch. Asserted positive odd
        integer matching the data pipeline.
    out_dim:
        Output feature dimensionality.
    base_channels:
        Width of the first conv block (the second uses ``2 * base_channels``).
    """

    def __init__(
        self,
        *,
        in_channels: int,
        patch_size: int,
        out_dim: int = 128,
        base_channels: int = 32,
    ) -> None:
        super().__init__()
        if in_channels <= 0 or patch_size <= 0:
            raise ValueError("in_channels and patch_size must be positive")
        self.in_channels = int(in_channels)
        self.patch_size = int(patch_size)
        self.out_dim = int(out_dim)

        c1 = base_channels
        c2 = 2 * base_channels

        self.block1 = nn.Sequential(_ConvNormAct(in_channels, c1), _ConvNormAct(c1, c1))
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.block2 = nn.Sequential(_ConvNormAct(c1, c2), _ConvNormAct(c2, c2))
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.out_proj = nn.Linear(c2, out_dim)

    def forward(self, patch: torch.Tensor) -> torch.Tensor:
        """Encode a batch of patches.

        Parameters
        ----------
        patch:
            ``(N, C_in, P, P)`` PCA-reduced patches centred on labelled pixels.

        Returns
        -------
        ``(N, out_dim)`` per-pixel spatial feature.
        """
        if patch.ndim != 4 or patch.shape[1] != self.in_channels:
            raise ValueError(
                f"expected (N, {self.in_channels}, P, P); got {tuple(patch.shape)}"
            )
        x = self.block1(patch)
        x = self.pool(x)
        x = self.block2(x)
        x = self.gap(x).flatten(1)  # (N, c2)
        return self.out_proj(x)  # (N, out_dim)
