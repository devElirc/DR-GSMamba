from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class SpectralSSMBlock(nn.Module):
    """Legacy lightweight state-space-style spectral mixer."""

    def __init__(self, dim: int):
        super().__init__()
        self.in_proj = nn.Linear(dim, dim * 2)
        self.depthwise = nn.Conv1d(dim, dim, kernel_size=5, padding=2, groups=dim)
        self.gate = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        u, v = self.in_proj(self.norm(x)).chunk(2, dim=-1)
        u = self.depthwise(u.transpose(1, 2)).transpose(1, 2)
        y = u * torch.sigmoid(self.gate(v))
        return residual + self.out_proj(y)


class SelectiveScanMambaBlock(nn.Module):
    """Mamba-style selective state-space block implemented in plain PyTorch.

    This follows the selective-scan idea: input-dependent delta, B, and C
    parameters drive a recurrent state update along the spectral sequence.
    It is slower than fused CUDA kernels, but it keeps the method reproducible
    on machines where official selective-scan extensions are unavailable.
    """

    def __init__(self, dim: int, state_dim: int = 8, expand: int = 1, dt_rank: int = 8):
        super().__init__()
        self.dim = dim
        self.inner_dim = dim * expand
        self.state_dim = state_dim
        self.in_proj = nn.Linear(dim, self.inner_dim * 2)
        self.conv = nn.Conv1d(self.inner_dim, self.inner_dim, kernel_size=3, padding=1, groups=self.inner_dim)
        self.x_proj = nn.Linear(self.inner_dim, dt_rank + 2 * state_dim)
        self.dt_proj = nn.Linear(dt_rank, self.inner_dim)
        self.A_log = nn.Parameter(torch.log(torch.arange(1, state_dim + 1, dtype=torch.float32)).repeat(self.inner_dim, 1))
        self.D = nn.Parameter(torch.ones(self.inner_dim))
        self.out_proj = nn.Linear(self.inner_dim, dim)
        self.norm = nn.LayerNorm(dim)

    def selective_scan(self, u: torch.Tensor, delta: torch.Tensor, b_par: torch.Tensor, c_par: torch.Tensor) -> torch.Tensor:
        batch, length, channels = u.shape
        state = u.new_zeros(batch, channels, self.state_dim)
        a = -torch.exp(self.A_log).to(dtype=u.dtype, device=u.device)
        outputs = []
        for step in range(length):
            dt = F.softplus(delta[:, step]).unsqueeze(-1)
            d_a = torch.exp(dt * a.unsqueeze(0))
            d_b = dt * b_par[:, step].unsqueeze(1) * u[:, step].unsqueeze(-1)
            state = state * d_a + d_b
            y = (state * c_par[:, step].unsqueeze(1)).sum(dim=-1) + self.D * u[:, step]
            outputs.append(y)
        return torch.stack(outputs, dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        xz = self.in_proj(self.norm(x))
        x_part, z_part = xz.chunk(2, dim=-1)
        x_part = self.conv(x_part.transpose(1, 2)).transpose(1, 2)
        x_part = F.silu(x_part)
        params = self.x_proj(x_part)
        dt_token, b_par, c_par = torch.split(params, [self.dt_proj.in_features, self.state_dim, self.state_dim], dim=-1)
        delta = self.dt_proj(dt_token)
        y = self.selective_scan(x_part, delta, b_par, c_par)
        y = y * F.silu(z_part)
        return residual + self.out_proj(y)


class SpectralEncoder(nn.Module):
    def __init__(self, spectral_dim: int, hidden_dim: int, depth: int):
        super().__init__()
        self.embed = nn.Linear(1, hidden_dim)
        self.blocks = nn.ModuleList([SpectralSSMBlock(hidden_dim) for _ in range(depth)])
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, spectrum: torch.Tensor) -> torch.Tensor:
        x = self.embed(spectrum.unsqueeze(-1))
        for block in self.blocks:
            x = block(x)
        return self.pool(x.transpose(1, 2)).squeeze(-1)


class SpectralMambaEncoder(nn.Module):
    def __init__(self, spectral_dim: int, hidden_dim: int, depth: int):
        super().__init__()
        self.embed = nn.Linear(1, hidden_dim)
        self.blocks = nn.ModuleList([SelectiveScanMambaBlock(hidden_dim) for _ in range(depth)])
        self.norm = nn.LayerNorm(hidden_dim)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, spectrum: torch.Tensor) -> torch.Tensor:
        x = self.embed(spectrum.unsqueeze(-1))
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return self.pool(x.transpose(1, 2)).squeeze(-1)


class SpectralTransformerEncoder(nn.Module):
    """Transformer replacement used for controlled spectral-backbone ablations."""

    def __init__(self, spectral_dim: int, hidden_dim: int, depth: int, num_heads: int = 4):
        super().__init__()
        if hidden_dim % num_heads != 0:
            num_heads = 1
        self.embed = nn.Linear(1, hidden_dim)
        self.position = nn.Parameter(torch.zeros(1, spectral_dim, hidden_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
        )
        self.blocks = nn.TransformerEncoder(layer, num_layers=max(1, depth))
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, spectrum: torch.Tensor) -> torch.Tensor:
        x = self.embed(spectrum.unsqueeze(-1)) + self.position[:, : spectrum.shape[1]]
        return self.norm(self.blocks(x)).mean(dim=1)


class SpectralCNNEncoder(nn.Module):
    """Compact 1D-CNN replacement used to test whether sequence modeling matters."""

    def __init__(self, spectral_dim: int, hidden_dim: int, depth: int):
        super().__init__()
        layers: list[nn.Module] = [nn.Conv1d(1, hidden_dim, kernel_size=5, padding=2), nn.GELU()]
        for _ in range(max(1, depth) - 1):
            layers.extend(
                [
                    nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2, groups=hidden_dim),
                    nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1),
                    nn.BatchNorm1d(hidden_dim),
                    nn.GELU(),
                ]
            )
        self.net = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, spectrum: torch.Tensor) -> torch.Tensor:
        x = self.net(spectrum.unsqueeze(1))
        return self.pool(x).squeeze(-1)


def build_spectral_encoder(backend: str, spectral_dim: int, hidden_dim: int, depth: int, num_heads: int) -> nn.Module:
    if backend in {"mamba", "selective_scan"}:
        return SpectralMambaEncoder(spectral_dim, hidden_dim, depth)
    if backend == "ssm":
        return SpectralEncoder(spectral_dim, hidden_dim, depth)
    if backend == "transformer":
        return SpectralTransformerEncoder(spectral_dim, hidden_dim, depth, num_heads=num_heads)
    if backend == "cnn":
        return SpectralCNNEncoder(spectral_dim, hidden_dim, depth)
    raise ValueError(f"Unknown spectral backend: {backend}. Expected one of: mamba, selective_scan, ssm, transformer, cnn.")


class SpatialStem(nn.Module):
    def __init__(self, in_channels: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim // 2, 3, padding=1),
            nn.BatchNorm2d(hidden_dim // 2),
            nn.GELU(),
            nn.Conv2d(hidden_dim // 2, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
        )

    def forward(self, patch: torch.Tensor) -> torch.Tensor:
        return self.net(patch)


class PatchGraphEncoder(nn.Module):
    def __init__(self, hidden_dim: int, layers: int = 2):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(layers)])

    def forward(self, nodes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        norm_nodes = F.normalize(nodes, dim=-1)
        adj = torch.bmm(norm_nodes, norm_nodes.transpose(1, 2))
        adj = torch.softmax(adj / max(nodes.shape[-1] ** 0.5, 1.0), dim=-1)
        x = nodes
        for layer, norm in zip(self.layers, self.norms):
            msg = torch.bmm(adj, x)
            x = norm(x + F.gelu(layer(msg)))
        return x.mean(dim=1), x, adj


class PrototypeHead(nn.Module):
    def __init__(self, hidden_dim: int, num_classes: int):
        super().__init__()
        self.prototypes = nn.Parameter(torch.randn(num_classes, hidden_dim) * 0.02)
        self.scale = nn.Parameter(torch.tensor(10.0))

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = F.normalize(features, dim=-1)
        prototypes = F.normalize(self.prototypes, dim=-1)
        logits = self.scale.clamp(1.0, 30.0) * features @ prototypes.t()
        uncertainty = 1.0 - torch.softmax(logits, dim=-1).max(dim=-1).values
        return logits, uncertainty

    def normalized_prototypes(self) -> torch.Tensor:
        return F.normalize(self.prototypes, dim=-1)


class DRGSMamba(nn.Module):
    def __init__(
        self,
        spectral_dim: int,
        num_classes: int,
        hidden_dim: int = 128,
        depth: int = 4,
        use_spectral: bool = True,
        use_graph: bool = True,
        use_prototype: bool = True,
        spectral_backend: str = "mamba",
        spectral_heads: int = 4,
        **_: int,
    ):
        super().__init__()
        self.use_spectral = use_spectral
        self.use_graph = use_graph
        self.use_prototype = use_prototype
        self.spectral_backend = spectral_backend
        self.spectral = build_spectral_encoder(spectral_backend, spectral_dim, hidden_dim, depth, spectral_heads)
        self.spatial = SpatialStem(spectral_dim, hidden_dim)
        self.graph = PatchGraphEncoder(hidden_dim)
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.node_classifier = nn.Linear(hidden_dim, num_classes)
        self.uncertainty_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.GELU(), nn.Linear(hidden_dim // 2, 1))
        self.prototype = PrototypeHead(hidden_dim, num_classes)

    def forward(self, patch: torch.Tensor, spectrum: torch.Tensor) -> dict[str, torch.Tensor]:
        spectral_feat = self.spectral(spectrum) if self.use_spectral else patch.new_zeros(patch.shape[0], self.classifier.in_features)
        fmap = self.spatial(patch)
        pooled = F.adaptive_avg_pool2d(fmap, 1).flatten(1)
        nodes = fmap.flatten(2).transpose(1, 2)
        graph_feat, graph_nodes, adj = self.graph(nodes)
        if not self.use_graph:
            graph_feat = torch.zeros_like(graph_feat)
            eye = torch.eye(adj.shape[-1], device=adj.device, dtype=adj.dtype).unsqueeze(0)
            adj = eye.expand(adj.shape[0], -1, -1)
        feat = self.fusion(torch.cat([spectral_feat, pooled, graph_feat], dim=-1))
        logits = self.classifier(feat)
        node_logits = self.node_classifier(graph_nodes)
        proto_logits, confidence_uncertainty = self.prototype(feat)
        if not self.use_prototype:
            proto_logits = torch.zeros_like(proto_logits)
        learned_uncertainty = F.softplus(self.uncertainty_head(feat).squeeze(-1))
        uncertainty = 0.5 * confidence_uncertainty + 0.5 * torch.tanh(learned_uncertainty)
        return {
            "logits": logits + proto_logits,
            "linear_logits": logits,
            "proto_logits": proto_logits,
            "node_logits": node_logits,
            "features": feat,
            "uncertainty": uncertainty,
            "learned_uncertainty": learned_uncertainty,
            "prototypes": self.prototype.normalized_prototypes(),
            "adjacency": adj,
        }
