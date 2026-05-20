from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from .dr_gsmamba import SpectralMambaEncoder


class SpectralMLP(nn.Module):
    """Simple spectral-vector baseline."""

    def __init__(self, spectral_dim: int, num_classes: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(spectral_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, patch: torch.Tensor, spectrum: torch.Tensor) -> torch.Tensor:
        return self.net(spectrum)


class PatchCNN2D(nn.Module):
    """2D spectral-spatial CNN baseline over PCA-reduced patches."""

    def __init__(self, spectral_dim: int, num_classes: int, hidden_dim: int = 64):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(spectral_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim * 2),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, patch: torch.Tensor, spectrum: torch.Tensor) -> torch.Tensor:
        x = self.features(patch).flatten(1)
        return self.classifier(x)


class PatchCNN3D(nn.Module):
    """Compact 3D-CNN baseline for spectral-spatial HSI patches."""

    def __init__(self, spectral_dim: int, num_classes: int, hidden_dim: int = 16):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv3d(1, hidden_dim, kernel_size=(7, 3, 3), padding=(3, 1, 1)),
            nn.BatchNorm3d(hidden_dim),
            nn.GELU(),
            nn.Conv3d(hidden_dim, hidden_dim * 2, kernel_size=(5, 3, 3), padding=(2, 1, 1)),
            nn.BatchNorm3d(hidden_dim * 2),
            nn.GELU(),
            nn.AdaptiveAvgPool3d(1),
        )
        self.classifier = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, patch: torch.Tensor, spectrum: torch.Tensor) -> torch.Tensor:
        x = self.features(patch.unsqueeze(1)).flatten(1)
        return self.classifier(x)


class HybridSNLite(nn.Module):
    """HybridSN-style lightweight baseline: 3D spectral-spatial features followed by 2D CNN."""

    def __init__(self, spectral_dim: int, num_classes: int, hidden_dim: int = 16):
        super().__init__()
        self.spectral_spatial = nn.Sequential(
            nn.Conv3d(1, hidden_dim, kernel_size=(7, 3, 3), padding=(3, 1, 1)),
            nn.BatchNorm3d(hidden_dim),
            nn.GELU(),
            nn.Conv3d(hidden_dim, hidden_dim, kernel_size=(5, 3, 3), padding=(2, 1, 1)),
            nn.BatchNorm3d(hidden_dim),
            nn.GELU(),
        )
        self.spatial = nn.Sequential(
            nn.Conv2d(hidden_dim * spectral_dim, hidden_dim * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim * 4),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 4, 128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, patch: torch.Tensor, spectrum: torch.Tensor) -> torch.Tensor:
        x = self.spectral_spatial(patch.unsqueeze(1))
        b, c, d, h, w = x.shape
        x = x.reshape(b, c * d, h, w)
        x = self.spatial(x).flatten(1)
        return self.classifier(x)


class SpectralFormerLite(nn.Module):
    """SpectralFormer-style baseline using band tokens and Transformer encoding."""

    def __init__(self, spectral_dim: int, num_classes: int, hidden_dim: int = 128, depth: int = 2, heads: int = 4):
        super().__init__()
        if hidden_dim % heads != 0:
            heads = 1
        self.embed = nn.Linear(1, hidden_dim)
        self.position = nn.Parameter(torch.zeros(1, spectral_dim, hidden_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth)
        self.norm = nn.LayerNorm(hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, patch: torch.Tensor, spectrum: torch.Tensor) -> torch.Tensor:
        x = self.embed(spectrum.unsqueeze(-1)) + self.position[:, : spectrum.shape[1]]
        x = self.norm(self.encoder(x)).mean(dim=1)
        return self.classifier(x)


class SSFTTLite(nn.Module):
    """SSFTT-style spectral-spatial Transformer baseline over patch tokens."""

    def __init__(self, spectral_dim: int, num_classes: int, hidden_dim: int = 128, depth: int = 2, heads: int = 4):
        super().__init__()
        if hidden_dim % heads != 0:
            heads = 1
        self.stem = nn.Sequential(
            nn.Conv2d(spectral_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
        )
        self.cls = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth)
        self.norm = nn.LayerNorm(hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, patch: torch.Tensor, spectrum: torch.Tensor) -> torch.Tensor:
        x = self.stem(patch).flatten(2).transpose(1, 2)
        cls = self.cls.expand(x.shape[0], -1, -1)
        tokens = torch.cat([cls, x], dim=1)
        tokens = self.norm(self.encoder(tokens))
        return self.classifier(tokens[:, 0])


class NonlocalGCNLite(nn.Module):
    """Nonlocal GCN-style patch baseline with learned affinity graph over patch nodes."""

    def __init__(self, spectral_dim: int, num_classes: int, hidden_dim: int = 128):
        super().__init__()
        self.node_embed = nn.Linear(spectral_dim, hidden_dim)
        self.gcn1 = nn.Linear(hidden_dim, hidden_dim)
        self.gcn2 = nn.Linear(hidden_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, patch: torch.Tensor, spectrum: torch.Tensor) -> torch.Tensor:
        nodes = patch.flatten(2).transpose(1, 2)
        x = self.node_embed(nodes)
        norm_nodes = F.normalize(x, dim=-1)
        adj = torch.softmax(torch.bmm(norm_nodes, norm_nodes.transpose(1, 2)) / max(x.shape[-1] ** 0.5, 1.0), dim=-1)
        x = self.norm1(x + F.gelu(self.gcn1(torch.bmm(adj, x))))
        x = self.norm2(x + F.gelu(self.gcn2(torch.bmm(adj, x))))
        graph_feat = x.mean(dim=1)
        center_feat = x[:, x.shape[1] // 2]
        return self.classifier(torch.cat([center_feat, graph_feat], dim=-1))


class MambaSpectralLite(nn.Module):
    """Mamba/selective-scan spectral baseline without graph, prototype, or DRO."""

    def __init__(self, spectral_dim: int, num_classes: int, hidden_dim: int = 128, depth: int = 2):
        super().__init__()
        self.encoder = SpectralMambaEncoder(spectral_dim, hidden_dim, depth)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, patch: torch.Tensor, spectrum: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.encoder(spectrum))


def build_baseline(name: str, spectral_dim: int, num_classes: int, hidden_dim: int) -> nn.Module:
    if name == "spectral_mlp":
        return SpectralMLP(spectral_dim, num_classes, hidden_dim=hidden_dim)
    if name == "cnn2d":
        return PatchCNN2D(spectral_dim, num_classes, hidden_dim=max(32, hidden_dim // 2))
    if name == "cnn3d":
        return PatchCNN3D(spectral_dim, num_classes, hidden_dim=max(8, hidden_dim // 8))
    if name == "hybridsn_lite":
        return HybridSNLite(spectral_dim, num_classes, hidden_dim=max(8, hidden_dim // 8))
    if name == "spectralformer_lite":
        return SpectralFormerLite(spectral_dim, num_classes, hidden_dim=hidden_dim)
    if name == "ssftt_lite":
        return SSFTTLite(spectral_dim, num_classes, hidden_dim=hidden_dim)
    if name == "nonlocal_gcn_lite":
        return NonlocalGCNLite(spectral_dim, num_classes, hidden_dim=hidden_dim)
    if name == "mamba_lite":
        return MambaSpectralLite(spectral_dim, num_classes, hidden_dim=hidden_dim)
    raise ValueError(f"Unknown baseline '{name}'.")
