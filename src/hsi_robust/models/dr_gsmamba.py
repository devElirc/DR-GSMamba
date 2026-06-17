"""Full DR-GSMamba backbone + Evidential Prototype Head.

Assembles the four model modules of Phase 2C into one ``nn.Module`` whose
forward returns the bundle expected by the training loop and the loss layer:

    raw spectrum (N, B) ---- OP-S4 encoder ---->  f_spec (N, d_spec)
                                                                       \\
                                                                        FusionMLP ---> f (N, d) --[CP-Graph refine]--> f' (N, d) --> EPH ---> evidence, alpha, probs, vacuity, aleatoric
                                                                       /
    PCA patch (N, C, P, P) -- spatial CNN ----->  f_spat (N, d_spat)

The model exposes one ``forward(spectrum, patch)`` returning a dict (per the
roadmap), and a ``from_config(config, num_bands, num_pca, patch_size, num_classes)``
factory that reads the model YAML (``configs/model/dr_gsmamba.yaml``).
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from hsi_robust.models.cp_graph import CPGraphRefinement
from hsi_robust.models.evidential_head import EvidentialPrototypeHead
from hsi_robust.models.fusion import FusionMLP
from hsi_robust.models.op_s4 import OPS4Encoder
from hsi_robust.models.spatial_stem import SpatialCNNStem


class DRGSMamba(nn.Module):
    """Two-stream backbone with optional CP-Graph refinement and EPH classifier.

    Returns a dict::

        {
            "evidence":   (N, K),
            "alpha":      (N, K),
            "probs":      (N, K),
            "vacuity":    (N,),
            "aleatoric":  (N,),
            "fused_feat": (N, feature_dim),
            "cos":        (N, K),
            "temperature": (),
        }
    """

    def __init__(
        self,
        *,
        num_bands: int,
        num_pca: int,
        patch_size: int,
        num_classes: int,
        feature_dim: int = 128,
        op_s4_hidden: int = 64,
        op_s4_state: int = 16,
        op_s4_layers: int = 2,
        op_s4_bidir: bool = True,
        op_s4_hippo_init: bool = True,
        op_s4_band_gate: bool = True,
        spatial_base_channels: int = 32,
        spatial_norm_type: str = "gn",
        spatial_dropout: float = 0.0,
        cp_graph_k: int = 8,
        cp_graph_tau_g: float = 1.0,
        use_cp_graph: bool = True,
        eph_tau_init: float = 5.0,
        eph_tau_min: float = 1.0,
        eph_tau_max: float = 30.0,
        eph_prototype_init: float = 0.02,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.num_classes = int(num_classes)
        self.use_cp_graph = bool(use_cp_graph)

        # Spectral branch -- OP-S4 encoder.
        self.op_s4 = OPS4Encoder(
            num_bands=num_bands,
            d_model=op_s4_hidden,
            d_state=op_s4_state,
            num_layers=op_s4_layers,
            out_dim=op_s4_hidden,
            bidirectional=op_s4_bidir,
            dropout=dropout,
            use_hippo_init=op_s4_hippo_init,
            use_band_gate=op_s4_band_gate,
        )
        spectral_dim = op_s4_hidden

        # Spatial branch -- compact 2D CNN.
        self.spatial = SpatialCNNStem(
            in_channels=num_pca,
            patch_size=patch_size,
            out_dim=2 * spatial_base_channels,
            base_channels=spatial_base_channels,
            norm_type=spatial_norm_type,
            dropout=spatial_dropout,
        )
        spatial_dim = 2 * spatial_base_channels

        # Fusion + optional graph refinement + head.
        self.fusion = FusionMLP(
            spectral_dim=spectral_dim,
            spatial_dim=spatial_dim,
            out_dim=feature_dim,
            dropout=dropout,
        )
        self.cp_graph = (
            CPGraphRefinement(k=cp_graph_k, tau_g=cp_graph_tau_g) if use_cp_graph else None
        )
        self.head = EvidentialPrototypeHead(
            feature_dim=feature_dim,
            num_classes=num_classes,
            tau_init=eph_tau_init,
            tau_min=eph_tau_min,
            tau_max=eph_tau_max,
            prototype_init_scale=eph_prototype_init,
        )

    @classmethod
    def from_config(
        cls,
        model_cfg: dict[str, Any],
        *,
        num_bands: int,
        num_pca: int,
        patch_size: int,
        num_classes: int,
    ) -> DRGSMamba:
        """Build a ``DRGSMamba`` from a parsed ``configs/model/*.yaml`` dict.

        Only the fields present in the YAML are consumed; everything else falls
        back to constructor defaults.
        """
        op = dict(model_cfg.get("op_s4") or {})
        sp = dict(model_cfg.get("spatial_stem") or {})
        cg = dict(model_cfg.get("cp_graph") or {})
        ev = dict(model_cfg.get("evidential_head") or {})
        # ``hidden_dims: [first_block_width]`` is accepted for backward compat;
        # the canonical key is ``base_channels``.
        base_channels = sp.get("base_channels")
        if base_channels is None:
            base_channels = (sp.get("hidden_dims") or [32])[0]
        return cls(
            num_bands=num_bands,
            num_pca=num_pca,
            patch_size=patch_size,
            num_classes=num_classes,
            feature_dim=int(model_cfg.get("feature_dim", 128)),
            op_s4_hidden=int(op.get("hidden_dim", 64)),
            op_s4_state=int(op.get("state_dim", 16)),
            op_s4_layers=int(op.get("num_layers", 2)),
            op_s4_bidir=bool(op.get("bidirectional", True)),
            op_s4_hippo_init=bool(op.get("hippo_init", True)),
            op_s4_band_gate=bool(op.get("band_importance_gating", True)),
            spatial_base_channels=int(base_channels),
            spatial_norm_type=str(sp.get("norm_type", "gn")),
            spatial_dropout=float(sp.get("dropout", 0.0)),
            cp_graph_k=int(cg.get("k", 8)),
            cp_graph_tau_g=float(cg.get("tau_g", 1.0)),
            use_cp_graph=bool(model_cfg.get("use_cp_graph", True)),
            eph_tau_init=float(ev.get("tau_init", 5.0)),
            eph_tau_min=float(ev.get("tau_min", 1.0)),
            eph_tau_max=float(ev.get("tau_max", 30.0)),
            eph_prototype_init=float(ev.get("prototype_init_scale", 0.02)),
        )

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, spectrum: torch.Tensor, patch: torch.Tensor) -> dict[str, torch.Tensor]:
        """Run the full two-stream forward pass.

        Parameters
        ----------
        spectrum:
            ``(N, num_bands)`` raw (already standardised) per-pixel spectrum.
        patch:
            ``(N, num_pca, patch_size, patch_size)`` PCA-reduced patch around
            the same pixel.

        Returns
        -------
        Dict with the keys listed in the class docstring.
        """
        f_spec = self.op_s4(spectrum)
        f_spat = self.spatial(patch)
        fused = self.fusion(f_spec, f_spat)
        if self.cp_graph is not None and fused.shape[0] >= 2:
            fused = self.cp_graph(fused)
        head_out = self.head(fused)
        head_out["fused_feat"] = fused
        return head_out
