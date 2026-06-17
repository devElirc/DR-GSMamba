"""CP-Graph consistency loss (one-directional KL with stop-gradient target).

Implements the boxed equations of ``docs/math/cp_graph.md`` §3--§4.

Mathematically, for each pixel ``i`` in a mini-batch we compute the
softmax-normalised cosine k-NN weights ``w_{ij}``, average the neighbour
predictive probabilities ``p_j`` into a target ``tilde p_i``, then minimise
``KL(sg(tilde p_i) || p_i)``. The target is stop-gradient (BYOL-style).

The shared graph builder is :func:`hsi_robust.models.cp_graph.build_cp_graph`.
"""

from __future__ import annotations

import math

import torch

from hsi_robust.models.cp_graph import build_cp_graph


def cp_graph_loss(
    features: torch.Tensor,
    probs: torch.Tensor,
    k: int = 8,
    tau_g: float = 1.0,
    eps: float = 1e-12,
    stop_grad_target: bool = True,
) -> tuple[torch.Tensor, dict]:
    """One-directional KL consistency loss on EPH probabilities over a k-NN graph.

    Parameters
    ----------
    features:
        ``(N, d)`` fused features ``f_i`` (used to build the graph).
    probs:
        ``(N, K)`` predictive probabilities ``p_i`` from EPH (must sum to 1
        along the last dim; the function does not check).
    k:
        Desired graph degree. Clamped to ``N - 1`` if too large.
    tau_g:
        Graph temperature on cosine similarities.
    eps:
        Numerical floor used inside the KL.
    stop_grad_target:
        If ``True`` (default, matches the math note §4 boxed equation) the
        neighbour-averaged target ``tilde p`` is detached from the autograd
        graph. Setting ``False`` lets gradients flow through neighbour
        predictions as well -- this is the symmetric-style variant we ablate
        in Phase 7 to verify the BYOL-style stop-gradient is in fact the
        better default.

    Returns
    -------
    loss:
        Scalar tensor (mean KL over the batch). Differentiable in ``probs``
        and ``features`` (via the unnormalised cosine inside the softmax);
        edge weights are detached and, by default, the target is detached.
    info:
        dict with ``mean_neighbour_kl``, ``mean_weight_entropy``, ``degree``,
        ``stop_grad_target`` (echoed back).

    Notes
    -----
    Falls through with a zero loss if ``N <= 1`` (degenerate graph).
    """
    if features.ndim != 2 or probs.ndim != 2:
        raise ValueError("features and probs must be 2-D tensors")
    if features.shape[0] != probs.shape[0]:
        raise ValueError("features and probs must share batch dimension")
    n = features.shape[0]
    if n <= 1 or k <= 0:
        zero = features.new_zeros(())
        return zero, {
            "mean_neighbour_kl": 0.0,
            "mean_weight_entropy": 0.0,
            "degree": 0,
            "stop_grad_target": bool(stop_grad_target),
        }

    idx, w = build_cp_graph(features, k=k, tau_g=tau_g)
    w_detached = w.detach()  # edge weights detached per math note §3

    # Neighbour predictive probabilities: (N, k', K). The math contract
    # detaches ``tilde p`` to make it a BYOL-style stop-gradient target. Pass
    # ``stop_grad_target=False`` for the symmetric-KL ablation.
    if stop_grad_target:
        neigh_probs = probs.detach()[idx]  # (N, k', K)
        tilde_p = (w_detached.unsqueeze(-1) * neigh_probs).sum(dim=1)
        tilde_p = tilde_p.detach()
    else:
        neigh_probs = probs[idx]  # gradient flows through neighbour rows
        tilde_p = (w_detached.unsqueeze(-1) * neigh_probs).sum(dim=1)

    # KL(sg(tilde p) || p) per sample, then mean.
    log_p = torch.log(probs.clamp(min=eps))
    log_tilde = torch.log(tilde_p.clamp(min=eps))
    per_sample_kl = (tilde_p * (log_tilde - log_p)).sum(dim=1)  # (N,)
    loss = per_sample_kl.mean()

    with torch.no_grad():
        weight_entropy = -(w_detached * w_detached.clamp(min=eps).log()).sum(dim=1).mean()
        # Normalise by log(degree) so entropy lies in [0, 1].
        deg = w_detached.shape[1]
        norm = math.log(max(deg, 2))
        weight_entropy_norm = (weight_entropy / norm).item()

    info = {
        "mean_neighbour_kl": float(per_sample_kl.mean().detach()),
        "mean_weight_entropy": float(weight_entropy_norm),
        "degree": int(w_detached.shape[1]),
        "stop_grad_target": bool(stop_grad_target),
    }
    return loss, info
