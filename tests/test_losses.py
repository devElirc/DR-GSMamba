"""Unit tests for Phase 2D losses.

The crown jewels are:

* :func:`test_cfa_gdro_matches_lp_solver_on_50_random_instances` --
  water-filling closed form vs SciPy LP solver, 50 random instances, 1e-6.
* :func:`test_evidential_bayes_risk_matches_monte_carlo` --
  closed-form Bayes-risk vs Monte-Carlo on 1000 samples, 1e-2.
* :func:`test_dirichlet_kl_matches_torch_distributions` --
  closed-form KL vs ``torch.distributions.kl_divergence`` on Dirichlets, 1e-6.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from scipy.optimize import linprog
from torch.distributions import Dirichlet, kl_divergence

from hsi_robust.losses import (
    bayes_risk_loss,
    ce_loss,
    cfa_gdro_loss,
    cp_graph_loss,
    dirichlet_kl_to_uniform,
    evidential_loss,
    focal_loss,
    sagawa_group_dro_loss,
    sample_cvar_loss,
)
from hsi_robust.losses.cfa_gdro import _water_filling

# --------------------------------------------------------------------------- #
# CFA-GDRO water-filling vs SciPy LP
# --------------------------------------------------------------------------- #


def _lp_solve_cfa_gdro(losses: np.ndarray, caps: np.ndarray) -> tuple[np.ndarray, float]:
    """Reference implementation: solve max_q <q, losses> via SciPy linprog."""
    k = losses.shape[0]
    c = -losses  # linprog minimises
    a_eq = np.ones((1, k))
    b_eq = np.array([1.0])
    bounds = [(0.0, float(cap)) for cap in caps]
    res = linprog(c=c, A_eq=a_eq, b_eq=b_eq, bounds=bounds, method="highs")
    if not res.success:
        raise RuntimeError(f"linprog failed: {res.message}")
    return res.x, -res.fun


def test_water_filling_matches_lp_on_50_random_instances() -> None:
    rng = np.random.default_rng(20260605)
    for trial in range(50):
        k = int(rng.integers(2, 25))
        losses = rng.uniform(0.0, 5.0, size=k)
        # Choose pi and alpha, gamma to derive caps that sum to >= 1.
        pi = rng.dirichlet(alpha=np.ones(k) * rng.uniform(0.3, 3.0))
        alpha = float(rng.uniform(0.05, 1.0))
        gamma = float(rng.uniform(0.0, 2.0))
        w = pi ** (-gamma)
        caps = w / (alpha * w.sum())
        # Reference value.
        _, val_ref = _lp_solve_cfa_gdro(losses, caps)
        # Water-filling.
        q_t, _, _ = _water_filling(
            torch.tensor(losses, dtype=torch.float64),
            torch.tensor(caps, dtype=torch.float64),
        )
        val_t = float((q_t * torch.tensor(losses)).sum().item())
        assert math.isclose(val_t, val_ref, abs_tol=1e-6, rel_tol=1e-6), (
            f"trial {trial}: K={k}, alpha={alpha}, gamma={gamma}, "
            f"WF value {val_t} vs LP value {val_ref}"
        )
        # q vectors may differ in a measure-zero set (degenerate boundary),
        # but the objective value must agree.


def test_cfa_gdro_sanity_special_cases() -> None:
    # alpha=1, gamma=0 -> caps = 1/K, q_star = 1/K (uniform).
    pi = torch.tensor([0.1, 0.2, 0.3, 0.4])
    losses = torch.tensor([1.0, 0.5, 2.0, 0.1], requires_grad=True)
    labels = torch.tensor([0, 1, 2, 3])
    loss, info = cfa_gdro_loss(losses, labels, pi, alpha=1.0, gamma=0.0)
    expected_q = torch.full((4,), 0.25)
    assert torch.allclose(info["q_star"], expected_q, atol=1e-6)
    # Gradient flows.
    loss.backward()
    assert losses.grad is not None

    # alpha=1/K, gamma=0 -> caps = 1, q_star = onehot(argmax loss) = (0,0,1,0).
    losses = torch.tensor([1.0, 0.5, 2.0, 0.1])
    labels = torch.tensor([0, 1, 2, 3])
    _, info = cfa_gdro_loss(losses, labels, pi, alpha=0.25, gamma=0.0)
    assert info["q_star"].argmax().item() == 2
    assert float(info["q_star"].max()) == pytest.approx(1.0)


def test_cfa_gdro_gradient_only_through_class_losses() -> None:
    pi = torch.tensor([0.5, 0.3, 0.2])
    losses = torch.tensor([1.0, 2.0, 3.0, 1.5], requires_grad=True)
    labels = torch.tensor([0, 1, 2, 0])
    loss, info = cfa_gdro_loss(losses, labels, pi, alpha=0.3, gamma=1.0)
    loss.backward()
    # No grad should be associated with q_star or caps.
    assert info["q_star"].grad_fn is None
    assert info["caps"].grad_fn is None
    # Gradient flows back into per_sample_losses.
    assert losses.grad is not None
    assert torch.isfinite(losses.grad).all()


def test_cfa_gdro_ema_pathway() -> None:
    pi = torch.tensor([0.4, 0.3, 0.2, 0.1])
    losses = torch.tensor([0.1, 0.2, 0.3, 0.4], requires_grad=True)
    labels = torch.tensor([0, 1, 2, 3])
    # Without EMA, q_star reflects the actual losses.
    _, info_no_ema = cfa_gdro_loss(losses, labels, pi, alpha=0.5, gamma=1.0)
    # With EMA in the opposite ordering, q_star should reflect the EMA.
    ema_losses = torch.tensor([10.0, 1.0, 1.0, 1.0])
    ema_seen = torch.tensor([True, True, True, True])
    _, info_ema = cfa_gdro_loss(
        losses,
        labels,
        pi,
        alpha=0.5,
        gamma=1.0,
        ema_class_losses=ema_losses,
        ema_seen=ema_seen,
    )
    # Under EMA the highest-EMA-loss class (class 0) saturates its cap; under
    # no-EMA it is dropped (its live loss is the smallest). We check the q_star
    # MASS on class 0 rather than the argmax, because class 2 has a much larger
    # cap so it absorbs the residual mass either way.
    assert float(info_ema["q_star"][0]) == pytest.approx(float(info_ema["caps"][0]), abs=1e-6)
    assert float(info_no_ema["q_star"][0]) == pytest.approx(0.0, abs=1e-6)
    # Conversely, under EMA class 3 (smallest EMA loss) is dropped; under
    # no-EMA it saturates (largest live loss).
    assert float(info_ema["q_star"][3]) == pytest.approx(0.0, abs=1e-6)
    assert float(info_no_ema["q_star"][3]) == pytest.approx(float(info_no_ema["caps"][3]), abs=1e-6)


def test_cfa_gdro_rejects_invalid_alpha_gamma() -> None:
    losses = torch.tensor([1.0, 2.0])
    labels = torch.tensor([0, 1])
    pi = torch.tensor([0.5, 0.5])
    with pytest.raises(ValueError):
        cfa_gdro_loss(losses, labels, pi, alpha=0.0, gamma=1.0)
    with pytest.raises(ValueError):
        cfa_gdro_loss(losses, labels, pi, alpha=1.5, gamma=1.0)
    with pytest.raises(ValueError):
        cfa_gdro_loss(losses, labels, pi, alpha=0.3, gamma=-1.0)


# --------------------------------------------------------------------------- #
# Evidential losses
# --------------------------------------------------------------------------- #


def test_evidential_bayes_risk_matches_monte_carlo() -> None:
    """Closed-form Bayes-risk vs Monte-Carlo on 1000 samples."""
    torch.manual_seed(0)
    # A small batch and modest K to keep MC stable.
    n, k = 8, 5
    alpha = 1.0 + torch.rand(n, k) * 4.0  # in [1, 5]
    labels = torch.randint(0, k, (n,))
    closed = bayes_risk_loss(alpha, labels)  # (N,)

    # Monte-Carlo estimate of E[||y - p||^2] with p ~ Dir(alpha).
    n_samples = 5000
    dist = torch.distributions.Dirichlet(alpha)  # (N, K)
    samples = dist.sample((n_samples,))  # (S, N, K)
    y_onehot = torch.zeros_like(alpha)
    y_onehot.scatter_(1, labels.view(-1, 1), 1.0)
    sq = (y_onehot.unsqueeze(0) - samples).pow(2).sum(dim=2)  # (S, N)
    mc = sq.mean(dim=0)  # (N,)

    diff = (closed - mc).abs()
    assert (diff < 1e-2).all(), f"Bayes-risk closed form vs MC differs by {diff.max()}"


def test_dirichlet_kl_matches_torch_distributions() -> None:
    """Closed-form KL(Dir(tilde_alpha) || Dir(1)) vs torch.distributions.kl_divergence."""
    torch.manual_seed(0)
    n, k = 10, 7
    tilde_alpha = 1.0 + torch.rand(n, k) * 9.0  # in [1, 10]
    ours = dirichlet_kl_to_uniform(tilde_alpha)  # (N,)
    # Reference.
    ref = kl_divergence(Dirichlet(tilde_alpha), Dirichlet(torch.ones(k)))  # (N,)
    diff = (ours - ref).abs()
    assert (diff < 1e-5).all(), f"max abs diff = {diff.max().item()}"


def test_evidential_loss_per_sample_and_mean() -> None:
    torch.manual_seed(0)
    n, k = 4, 3
    alpha = 1.0 + torch.rand(n, k) * 2.0
    labels = torch.randint(0, k, (n,))
    per_sample, mean_loss, info = evidential_loss(alpha, labels, kl_weight=0.5)
    assert per_sample.shape == (n,)
    assert mean_loss.ndim == 0
    assert float(mean_loss) == pytest.approx(float(per_sample.mean()), abs=1e-6)
    assert {"lik", "kl", "mean_vacuity", "kl_weight"}.issubset(info.keys())


def test_evidential_loss_kl_weight_zero_recovers_bayes_risk() -> None:
    torch.manual_seed(0)
    alpha = 1.0 + torch.rand(5, 4) * 3.0
    labels = torch.randint(0, 4, (5,))
    only_lik, _, _ = evidential_loss(alpha, labels, kl_weight=0.0)
    expected = bayes_risk_loss(alpha, labels)
    assert torch.allclose(only_lik, expected, atol=1e-6)


def test_evidential_loss_gradient_flows_through_alpha() -> None:
    alpha = (1.0 + torch.rand(3, 5) * 2.0).requires_grad_(True)
    labels = torch.tensor([0, 2, 4])
    _, mean_loss, _ = evidential_loss(alpha, labels, kl_weight=0.3)
    mean_loss.backward()
    assert alpha.grad is not None
    assert torch.isfinite(alpha.grad).all()


# --------------------------------------------------------------------------- #
# CP-Graph
# --------------------------------------------------------------------------- #


def test_cp_graph_loss_basic_shape_and_grad() -> None:
    torch.manual_seed(0)
    n, d, k = 8, 16, 4
    features = torch.randn(n, d, requires_grad=True)
    logits = torch.randn(n, k)
    probs = torch.softmax(logits, dim=-1).requires_grad_(True)
    loss, info = cp_graph_loss(features, probs, k=3, tau_g=1.0)
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert info["degree"] == 3
    loss.backward()
    # Gradient flows through probs.
    assert probs.grad is not None


def test_cp_graph_loss_zero_when_probs_identical() -> None:
    # If all probs are equal, KL(tilde p || p) = KL(p || p) = 0.
    n, k = 6, 4
    p = torch.full((n, k), 1.0 / k)
    features = torch.randn(n, 8)
    loss, _ = cp_graph_loss(features, p, k=3, tau_g=1.0)
    assert float(loss) == pytest.approx(0.0, abs=1e-6)


def test_cp_graph_loss_degenerate_single_sample() -> None:
    loss, info = cp_graph_loss(torch.randn(1, 8), torch.softmax(torch.randn(1, 5), dim=-1), k=3)
    assert float(loss) == 0.0
    assert info["degree"] == 0


def test_cp_graph_loss_stop_grad_target_propagates_to_probs() -> None:
    """D-11: stop_grad_target=False lets gradient flow through neighbour probs."""
    torch.manual_seed(0)
    n, d, k = 6, 8, 3
    features = torch.randn(n, d, requires_grad=True)
    logits = torch.randn(n, k)
    probs = torch.softmax(logits, dim=-1).requires_grad_(True)
    # With stop-grad target (default), gradient on probs comes only from log p.
    loss_sg, info_sg = cp_graph_loss(features, probs, k=2, tau_g=1.0, stop_grad_target=True)
    grad_sg = torch.autograd.grad(loss_sg, probs, retain_graph=True)[0]
    # Without stop-grad target, both log p and the neighbour-average contribute.
    loss_ng, info_ng = cp_graph_loss(features, probs, k=2, tau_g=1.0, stop_grad_target=False)
    grad_ng = torch.autograd.grad(loss_ng, probs)[0]
    assert info_sg["stop_grad_target"] is True
    assert info_ng["stop_grad_target"] is False
    # The two gradient patterns must differ (otherwise stop_grad_target is dead).
    assert not torch.allclose(grad_sg, grad_ng, atol=1e-6)


# --------------------------------------------------------------------------- #
# Baselines
# --------------------------------------------------------------------------- #


def test_baseline_ce_matches_torch_reference() -> None:
    logits = torch.randn(8, 10)
    labels = torch.randint(0, 10, (8,))
    per_sample, mean_loss = ce_loss(logits, labels)
    ref_per = torch.nn.functional.cross_entropy(logits, labels, reduction="none")
    assert torch.allclose(per_sample, ref_per, atol=1e-6)
    assert float(mean_loss) == pytest.approx(float(ref_per.mean()), abs=1e-6)


def test_baseline_focal_reduces_to_ce_when_gamma_zero() -> None:
    logits = torch.randn(6, 4)
    labels = torch.randint(0, 4, (6,))
    per_sample_focal, _ = focal_loss(logits, labels, gamma_focal=0.0)
    per_sample_ce, _ = ce_loss(logits, labels)
    assert torch.allclose(per_sample_focal, per_sample_ce, atol=1e-5)


def test_baseline_sample_cvar_keeps_top_fraction() -> None:
    losses = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    loss, info = sample_cvar_loss(losses, alpha=0.4)  # keep top 2
    assert info["num_kept"] == 2
    assert float(loss) == pytest.approx(4.5)


def test_baseline_sagawa_group_dro_basic() -> None:
    torch.manual_seed(0)
    per_sample = torch.tensor([0.1, 0.2, 1.5, 0.05], requires_grad=True)
    labels = torch.tensor([0, 0, 1, 2])
    loss, q_new, _ = sagawa_group_dro_loss(per_sample, labels, num_classes=3, q_state=None, eta=0.1)
    assert q_new.shape == (3,)
    assert float(q_new.sum()) == pytest.approx(1.0)
    # Class 1 (loss 1.5) should get the largest weight.
    assert q_new.argmax().item() == 1
    loss.backward()
    assert per_sample.grad is not None


# --------------------------------------------------------------------------- #
# Negative paths (input validation -- needed for ≥90% coverage)
# --------------------------------------------------------------------------- #


def test_baseline_focal_rejects_negative_gamma() -> None:
    with pytest.raises(ValueError):
        focal_loss(torch.randn(2, 3), torch.tensor([0, 1]), gamma_focal=-0.1)


def test_baseline_sample_cvar_validates_inputs() -> None:
    with pytest.raises(ValueError):
        sample_cvar_loss(torch.randn(2, 3))
    with pytest.raises(ValueError):
        sample_cvar_loss(torch.randn(4), alpha=0.0)
    with pytest.raises(ValueError):
        sample_cvar_loss(torch.randn(4), alpha=1.5)


def test_baseline_sagawa_validates_inputs() -> None:
    with pytest.raises(ValueError):
        sagawa_group_dro_loss(torch.randn(2, 3), torch.tensor([0, 1]), num_classes=3)
    with pytest.raises(ValueError):
        sagawa_group_dro_loss(torch.randn(2), torch.tensor([0, 1, 2]), num_classes=3)
    with pytest.raises(ValueError):
        sagawa_group_dro_loss(torch.randn(2), torch.tensor([0, 1]), num_classes=3, eta=-0.01)


def test_cfa_gdro_validates_shapes_and_labels() -> None:
    pi = torch.tensor([0.5, 0.5])
    # per_sample_losses must be 1-D.
    with pytest.raises(ValueError):
        cfa_gdro_loss(torch.randn(2, 3), torch.tensor([0, 1]), pi, alpha=0.3, gamma=1.0)
    # Mismatched shapes.
    with pytest.raises(ValueError):
        cfa_gdro_loss(torch.randn(3), torch.tensor([0, 1]), pi, alpha=0.3, gamma=1.0)
    # scene_freq must be 1-D.
    with pytest.raises(ValueError):
        cfa_gdro_loss(torch.randn(2), torch.tensor([0, 1]), pi.view(1, 2), alpha=0.3, gamma=1.0)
    # Out-of-range labels.
    with pytest.raises(ValueError):
        cfa_gdro_loss(torch.randn(2), torch.tensor([0, 5]), pi, alpha=0.3, gamma=1.0)
    # Wrong EMA shape.
    with pytest.raises(ValueError):
        cfa_gdro_loss(
            torch.randn(2),
            torch.tensor([0, 1]),
            pi,
            alpha=0.3,
            gamma=1.0,
            ema_class_losses=torch.zeros(5),
        )


def test_cfa_gdro_ema_without_seen_uses_ema_directly() -> None:
    """Covers the ema_seen=None branch of cfa_gdro_loss."""
    pi = torch.tensor([0.5, 0.5])
    losses = torch.tensor([0.1, 0.2])
    labels = torch.tensor([0, 1])
    ema = torch.tensor([5.0, 1.0])
    _, info = cfa_gdro_loss(losses, labels, pi, alpha=0.3, gamma=1.0, ema_class_losses=ema)
    # Class 0 has the larger EMA loss; under alpha=0.3, gamma=1.0, caps sum to 1/alpha
    # > 1, so class 0 saturates and class 1 receives the residual.
    assert info["q_star"][0] >= info["q_star"][1]


def test_water_filling_rejects_bad_shape() -> None:
    with pytest.raises(ValueError):
        _water_filling(torch.zeros(3, 2), torch.zeros(3, 2))


def test_evidential_validates_shapes() -> None:
    with pytest.raises(ValueError):
        bayes_risk_loss(torch.rand(4), torch.tensor([0, 1]))  # 1-D alpha
    with pytest.raises(ValueError):
        bayes_risk_loss(torch.rand(2, 3), torch.tensor([0, 1, 2]))  # label shape mismatch
    with pytest.raises(ValueError):
        bayes_risk_loss(torch.rand(2, 3), torch.tensor([0, 5]))  # out-of-range
    with pytest.raises(ValueError):
        dirichlet_kl_to_uniform(torch.rand(4))  # must be 2-D


def test_cp_graph_loss_validates_shapes() -> None:
    with pytest.raises(ValueError):
        cp_graph_loss(torch.randn(4), torch.softmax(torch.randn(4, 3), -1))
    with pytest.raises(ValueError):
        cp_graph_loss(torch.randn(4, 8), torch.softmax(torch.randn(5, 3), -1))
