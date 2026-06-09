"""Concat + MLP fusion of spectral and spatial features.

This is a **design choice, not a contribution**. Following common HSI practice
we concatenate the spectral pathway output (from :class:`OPS4Encoder`) and the
spatial pathway output (from :class:`SpatialCNNStem`) and pass the result
through a 2-layer MLP with GELU activation and dropout.

The fused feature ``f \\in R^d`` is what flows into the evidential prototype
head (Eq. (8) of ``method.tex``).
"""

from __future__ import annotations

import torch
from torch import nn


class FusionMLP(nn.Module):
    """Concatenate-then-MLP fusion of two feature streams.

    Parameters
    ----------
    spectral_dim, spatial_dim:
        Dimensionalities of the two input streams.
    out_dim:
        Fused feature dimensionality (= EPH input dim).
    hidden_dim:
        Width of the MLP hidden layer. Defaults to ``out_dim``.
    dropout:
        Dropout probability between the two linear layers.
    """

    def __init__(
        self,
        *,
        spectral_dim: int,
        spatial_dim: int,
        out_dim: int = 128,
        hidden_dim: int | None = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if spectral_dim <= 0 or spatial_dim <= 0 or out_dim <= 0:
            raise ValueError("all dims must be positive")
        self.spectral_dim = int(spectral_dim)
        self.spatial_dim = int(spatial_dim)
        self.out_dim = int(out_dim)
        h = int(hidden_dim if hidden_dim is not None else out_dim)
        self.mlp = nn.Sequential(
            nn.Linear(spectral_dim + spatial_dim, h),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(h, out_dim),
        )

    def forward(self, spectral: torch.Tensor, spatial: torch.Tensor) -> torch.Tensor:
        """Concatenate and project.

        Parameters
        ----------
        spectral:
            ``(N, spectral_dim)`` per-pixel spectral feature.
        spatial:
            ``(N, spatial_dim)`` per-pixel spatial feature.

        Returns
        -------
        ``(N, out_dim)`` fused feature.
        """
        if spectral.ndim != 2 or spectral.shape[-1] != self.spectral_dim:
            raise ValueError(
                f"spectral must be (N, {self.spectral_dim}); got {tuple(spectral.shape)}"
            )
        if spatial.ndim != 2 or spatial.shape[-1] != self.spatial_dim:
            raise ValueError(
                f"spatial must be (N, {self.spatial_dim}); got {tuple(spatial.shape)}"
            )
        if spectral.shape[0] != spatial.shape[0]:
            raise ValueError(
                f"batch dimension mismatch: spectral {spectral.shape[0]} vs spatial {spatial.shape[0]}"
            )
        x = torch.cat([spectral, spatial], dim=-1)
        return self.mlp(x)
