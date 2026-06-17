"""Unit tests for Phase 2C model modules.

Covers forward-shape, gradient flow, parameter budget, determinism, and a
couple of basic mathematical invariants (Dirichlet probabilities sum to 1,
vacuity in (0, 1], etc.).
"""

from __future__ import annotations

import pytest
import torch

from hsi_robust.models import (
    CPGraphRefinement,
    DRGSMamba,
    EvidentialPrototypeHead,
    FusionMLP,
    OPS4Block,
    OPS4Encoder,
    SpatialCNNStem,
    build_cp_graph,
)

# --------------------------------------------------------------------------- #
# OP-S4
# --------------------------------------------------------------------------- #


def test_ops4_block_forward_shape() -> None:
    block = OPS4Block(d_model=16, d_state=8, dropout=0.0)
    x = torch.randn(4, 20, 16)
    y = block(x)
    assert y.shape == (4, 20, 16)
    assert torch.isfinite(y).all()


def test_ops4_encoder_forward_shape_and_grad() -> None:
    encoder = OPS4Encoder(num_bands=30, d_model=16, d_state=8, num_layers=2, out_dim=24)
    x = torch.randn(5, 30, requires_grad=True)
    y = encoder(x)
    assert y.shape == (5, 24)
    y.sum().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_ops4_encoder_bidirectional_doubles_internal_dim() -> None:
    # When bidirectional, internal pooled dim before out_proj is 2 * d_model.
    bidir = OPS4Encoder(num_bands=10, d_model=8, num_layers=1, out_dim=8, bidirectional=True)
    uni = OPS4Encoder(num_bands=10, d_model=8, num_layers=1, out_dim=8, bidirectional=False)
    x = torch.randn(2, 10)
    assert bidir(x).shape == (2, 8)
    assert uni(x).shape == (2, 8)
    # Bidirectional has roughly 2x the SSM parameters.
    assert sum(p.numel() for p in bidir.parameters()) > sum(p.numel() for p in uni.parameters())


# --------------------------------------------------------------------------- #
# Spatial CNN stem
# --------------------------------------------------------------------------- #


def test_spatial_stem_forward_shape_grad() -> None:
    stem = SpatialCNNStem(in_channels=30, patch_size=9, out_dim=64, base_channels=16)
    x = torch.randn(6, 30, 9, 9, requires_grad=True)
    y = stem(x)
    assert y.shape == (6, 64)
    y.sum().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_spatial_stem_rejects_wrong_channels() -> None:
    stem = SpatialCNNStem(in_channels=10, patch_size=7)
    with pytest.raises(ValueError):
        stem(torch.randn(2, 11, 7, 7))


def test_spatial_stem_norm_type_toggle() -> None:
    """Decision D-09 ablation knob: 'gn' (default) vs 'bn' must both run."""
    gn = SpatialCNNStem(in_channels=6, patch_size=5, out_dim=16, base_channels=8, norm_type="gn")
    bn = SpatialCNNStem(in_channels=6, patch_size=5, out_dim=16, base_channels=8, norm_type="bn")
    x = torch.randn(4, 6, 5, 5)
    assert gn(x).shape == (4, 16)
    assert bn(x).shape == (4, 16)
    # Mismatched key must raise.
    with pytest.raises(ValueError):
        SpatialCNNStem(in_channels=6, patch_size=5, norm_type="layer")


def test_spatial_stem_dropout_active() -> None:
    stem = SpatialCNNStem(in_channels=4, patch_size=5, out_dim=8, base_channels=8, dropout=0.5)
    assert isinstance(stem.dropout, torch.nn.Dropout)
    assert stem.dropout.p == 0.5
    stem.train()
    torch.manual_seed(0)
    out1 = stem(torch.randn(8, 4, 5, 5))
    torch.manual_seed(1)
    out2 = stem(torch.randn(8, 4, 5, 5))
    assert not torch.allclose(out1, out2)


def test_ops4_encoder_disables_band_gate_and_hippo_init() -> None:
    enc = OPS4Encoder(
        num_bands=12,
        d_model=8,
        num_layers=1,
        out_dim=8,
        bidirectional=False,
        use_hippo_init=False,
        use_band_gate=False,
    )
    assert enc.band_gate is None
    assert enc.fwd_blocks[0].use_hippo_init is False
    y = enc(torch.randn(3, 12))
    assert y.shape == (3, 8)


# --------------------------------------------------------------------------- #
# CP-Graph
# --------------------------------------------------------------------------- #


def test_build_cp_graph_shapes_and_normalisation() -> None:
    f = torch.randn(8, 16)
    idx, w = build_cp_graph(f, k=3, tau_g=1.0)
    assert idx.shape == (8, 3)
    assert w.shape == (8, 3)
    assert (idx >= 0).all() and (idx < 8).all()
    # Self-loops excluded.
    self_idx = torch.arange(8)
    assert (idx != self_idx.unsqueeze(1)).all()
    # Each row of weights sums to 1.
    row_sums = w.sum(dim=1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)


def test_build_cp_graph_clamps_k() -> None:
    # k larger than N - 1 must be clamped, not crash.
    f = torch.randn(4, 8)
    idx, _ = build_cp_graph(f, k=10, tau_g=1.0)
    assert idx.shape[1] == 3  # N - 1


def test_cp_graph_refinement_preserves_shape() -> None:
    layer = CPGraphRefinement(k=3, tau_g=1.0)
    f = torch.randn(6, 32, requires_grad=True)
    out = layer(f)
    assert out.shape == f.shape
    out.sum().backward()
    assert f.grad is not None


def test_cp_graph_refinement_passthrough_for_single_sample() -> None:
    layer = CPGraphRefinement(k=3, tau_g=1.0)
    f = torch.randn(1, 8)
    out = layer(f)
    # With a single sample no graph can be built; should return identity.
    assert torch.equal(out, f)


# --------------------------------------------------------------------------- #
# Fusion
# --------------------------------------------------------------------------- #


def test_fusion_mlp_shape() -> None:
    fusion = FusionMLP(spectral_dim=24, spatial_dim=64, out_dim=128, dropout=0.0)
    s = torch.randn(7, 24)
    p = torch.randn(7, 64)
    f = fusion(s, p)
    assert f.shape == (7, 128)


def test_fusion_mlp_batch_mismatch() -> None:
    fusion = FusionMLP(spectral_dim=8, spatial_dim=8, out_dim=8)
    with pytest.raises(ValueError):
        fusion(torch.randn(4, 8), torch.randn(5, 8))


# --------------------------------------------------------------------------- #
# Evidential prototype head
# --------------------------------------------------------------------------- #


def test_eph_invariants() -> None:
    torch.manual_seed(0)
    head = EvidentialPrototypeHead(feature_dim=32, num_classes=10)
    f = torch.randn(12, 32)
    out = head(f)
    # Shapes.
    assert out["evidence"].shape == (12, 10)
    assert out["alpha"].shape == (12, 10)
    assert out["probs"].shape == (12, 10)
    assert out["vacuity"].shape == (12,)
    assert out["aleatoric"].shape == (12,)
    # Evidence non-negative, alpha >= 1.
    assert (out["evidence"] >= 0).all()
    assert (out["alpha"] >= 1.0 - 1e-6).all()
    # Probabilities form a simplex.
    sums = out["probs"].sum(dim=1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)
    # Vacuity in (0, 1].
    assert (out["vacuity"] > 0).all()
    assert (out["vacuity"] <= 1.0 + 1e-6).all()
    # Aleatoric in [0, (K-1)/K].
    k = 10
    assert (out["aleatoric"] >= -1e-7).all()
    assert (out["aleatoric"] <= (k - 1) / k + 1e-6).all()
    # Temperature respects the clip range.
    tau = float(out["temperature"].detach())
    assert head.tau_min - 1e-6 <= tau <= head.tau_max + 1e-6


def test_eph_temperature_clipping() -> None:
    head = EvidentialPrototypeHead(feature_dim=4, num_classes=3, tau_init=20.0)
    # Force raw_tau extremely large -> temperature clipped to tau_max.
    with torch.no_grad():
        head.raw_tau.fill_(1e6)
    assert float(head.temperature().detach()) == pytest.approx(head.tau_max)
    # Force raw_tau extremely negative -> softplus ~ 0 -> clipped up to tau_min.
    with torch.no_grad():
        head.raw_tau.fill_(-1e6)
    assert float(head.temperature().detach()) == pytest.approx(head.tau_min)


def test_ops4_block_fft_matches_recurrent_reference() -> None:
    """Sanity: FFT convolution form agrees with a pure-Python recurrence."""
    torch.manual_seed(42)
    block = OPS4Block(d_model=8, d_state=4, dropout=0.0)
    block.eval()
    # Disable the residual path / activation / norm so we compare only the SSM.
    x = torch.randn(2, 16, 8)
    # Reference: explicit recurrent form (h_0 = 0).
    with torch.no_grad():
        x_n = block.norm(x)
        dt = torch.exp(block.log_dt).unsqueeze(-1)
        A = -torch.exp(block.log_A)
        a = torch.exp(dt * A)
        n, t, _ = x_n.shape
        h = x_n.new_zeros(n, block.d_model, block.d_state)
        ref_outs = []
        for step in range(t):
            xt = x_n[:, step, :].unsqueeze(-1)
            h = a.unsqueeze(0) * h + block.B.unsqueeze(0) * xt
            yt = (block.C.unsqueeze(0) * h).sum(dim=-1) + block.D * x_n[:, step, :]
            ref_outs.append(yt)
        y_ref = torch.stack(ref_outs, dim=1)
        # FFT form, also skipping act + dropout + residual.
        kernel = block._kernel(t, dtype=x_n.dtype, device=x_n.device)
        fft_len = 2 * t
        x_fft = torch.fft.rfft(x_n, n=fft_len, dim=1)
        k_fft = torch.fft.rfft(kernel, n=fft_len, dim=0)
        y_fft = torch.fft.irfft(x_fft * k_fft.unsqueeze(0), n=fft_len, dim=1)[:, :t, :]
        y_fft = y_fft + block.D * x_n
    assert torch.allclose(y_ref, y_fft, atol=1e-5)


# --------------------------------------------------------------------------- #
# DR-GSMamba assembly
# --------------------------------------------------------------------------- #


def test_drgsmamba_forward_shapes() -> None:
    torch.manual_seed(0)
    model = DRGSMamba(
        num_bands=30,
        num_pca=8,
        patch_size=9,
        num_classes=5,
        feature_dim=32,
        op_s4_hidden=16,
        op_s4_state=8,
        op_s4_layers=2,
        spatial_base_channels=16,
        cp_graph_k=2,
    )
    spec = torch.randn(4, 30)
    patch = torch.randn(4, 8, 9, 9)
    out = model(spec, patch)
    n, k = 4, 5
    assert out["evidence"].shape == (n, k)
    assert out["alpha"].shape == (n, k)
    assert out["probs"].shape == (n, k)
    assert out["vacuity"].shape == (n,)
    assert out["aleatoric"].shape == (n,)
    assert out["fused_feat"].shape == (n, 32)
    # Backprop end-to-end.
    out["probs"].sum().backward()


def test_drgsmamba_param_count_under_budget() -> None:
    # Default-ish config sized for Indian Pines (200 bands, 30 PCA, 9x9, 16 classes).
    model = DRGSMamba(
        num_bands=200,
        num_pca=30,
        patch_size=9,
        num_classes=16,
        feature_dim=128,
        op_s4_hidden=64,
        op_s4_state=16,
        op_s4_layers=2,
        spatial_base_channels=32,
    )
    n_params = model.num_parameters()
    assert n_params < 5_000_000, f"model has {n_params} parameters, exceeding 5M budget"


def test_drgsmamba_deterministic_under_seed() -> None:
    def _run() -> torch.Tensor:
        torch.manual_seed(123)
        model = DRGSMamba(
            num_bands=20,
            num_pca=6,
            patch_size=5,
            num_classes=4,
            feature_dim=16,
            op_s4_hidden=8,
            op_s4_state=4,
            op_s4_layers=1,
            spatial_base_channels=8,
            cp_graph_k=2,
            dropout=0.0,
        )
        model.eval()
        torch.manual_seed(7)
        spec = torch.randn(3, 20)
        patch = torch.randn(3, 6, 5, 5)
        return model(spec, patch)["probs"]

    out1 = _run()
    out2 = _run()
    assert torch.allclose(out1, out2, atol=1e-6)


def test_drgsmamba_from_config() -> None:
    cfg = {
        "feature_dim": 32,
        "op_s4": {"hidden_dim": 16, "num_layers": 1, "bidirectional": True},
        "spatial_stem": {"hidden_dims": [16]},
        "cp_graph": {"k": 2, "tau_g": 1.0},
        "evidential_head": {"tau_init": 4.0, "tau_min": 1.0, "tau_max": 30.0},
    }
    model = DRGSMamba.from_config(cfg, num_bands=20, num_pca=6, patch_size=5, num_classes=4)
    out = model(torch.randn(2, 20), torch.randn(2, 6, 5, 5))
    assert out["probs"].shape == (2, 4)


def test_drgsmamba_from_config_threads_all_knobs() -> None:
    """Decision D-11 contract: every model-YAML key must drive code."""
    cfg = {
        "feature_dim": 24,
        "op_s4": {
            "hidden_dim": 12,
            "state_dim": 6,
            "num_layers": 1,
            "bidirectional": False,
            "hippo_init": False,
            "band_importance_gating": False,
        },
        "spatial_stem": {
            "base_channels": 8,
            "norm_type": "bn",
            "dropout": 0.3,
        },
        "cp_graph": {"k": 2, "tau_g": 0.7},
        "evidential_head": {"tau_init": 3.0, "tau_min": 1.0, "tau_max": 20.0},
    }
    model = DRGSMamba.from_config(cfg, num_bands=10, num_pca=4, patch_size=5, num_classes=3)
    # op_s4 knobs honoured.
    assert model.op_s4.bidirectional is False
    assert model.op_s4.use_band_gate is False
    assert model.op_s4.fwd_blocks[0].use_hippo_init is False
    assert model.op_s4.d_model == 12
    assert model.op_s4.fwd_blocks[0].d_state == 6
    # spatial knobs honoured.
    assert model.spatial.norm_type == "bn"
    assert isinstance(model.spatial.dropout, torch.nn.Dropout)
    assert model.spatial.dropout.p == 0.3
    # Forward pass still works.
    out = model(torch.randn(4, 10), torch.randn(4, 4, 5, 5))
    assert out["probs"].shape == (4, 3)
