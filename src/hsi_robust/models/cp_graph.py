"""Cross-Pixel Affinity Graph (CP-Graph) — in-batch k-NN graph reasoning.

This module is a **design choice, not a contribution** (see
``docs/math/cp_graph.md``). It provides:

* :func:`build_cp_graph` -- the shared, batch-local cosine k-NN graph builder
  used by both this layer and ``losses/cp_graph.py``;
* :class:`CPGraphRefinement` -- an optional edge-weighted message-passing layer
  that refines the fused feature with a learnable mix coefficient (residual).

The graph is rebuilt fresh every mini-batch (no persistent state) and is
directed (cosine k-NN is not symmetric).
"""

from __future__ import annotations

import torch
from torch import nn


def build_cp_graph(
    features: torch.Tensor,
    k: int,
    tau_g: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """In-batch cosine k-NN graph with softmax-normalised edge weights.

    Parameters
    ----------
    features:
        ``(N, d)`` tensor of feature vectors.
    k:
        Desired graph degree. If ``k >= N``, it is clamped to ``N - 1``.
    tau_g:
        Softmax temperature on cosine similarities.

    Returns
    -------
    neighbor_idx:
        ``(N, k')`` long tensor with the indices of the k' nearest neighbours
        of each row (k' = min(k, N - 1)).
    edge_weights:
        ``(N, k')`` float tensor with the softmax-normalised cosine
        similarities. Rows sum to 1.

    Notes
    -----
    The unnormalised cosine *is* differentiable w.r.t. ``features`` -- callers
    that want to detach the weights (as the consistency loss does) must do so
    explicitly via ``edge_weights.detach()``.
    """
    if features.ndim != 2:
        raise ValueError(f"features must be 2-D; got shape {tuple(features.shape)}")
    if k <= 0:
        raise ValueError(f"k must be positive; got {k}")
    n, _ = features.shape
    k_eff = int(min(k, n - 1))
    if k_eff <= 0:
        raise ValueError("need at least N=2 samples to build a CP-Graph")

    f_norm = nn.functional.normalize(features, p=2, dim=1, eps=1e-12)
    sim = f_norm @ f_norm.t()  # (N, N), cosine similarities in [-1, 1]
    # Mask self-similarity by setting it to a sentinel below the minimum.
    eye = torch.eye(n, device=features.device, dtype=torch.bool)
    sim = sim.masked_fill(eye, float("-inf"))

    top_sim, top_idx = torch.topk(sim, k=k_eff, dim=1)  # both (N, k_eff)
    edge_weights = torch.softmax(top_sim / tau_g, dim=1)
    return top_idx, edge_weights


class CPGraphRefinement(nn.Module):
    """Edge-weighted message-passing layer with learnable residual mix.

    Refined feature

        f_i^{out} = f_i + sigmoid(mix) * sum_{j in N_k(i)} w_{ij} * f_j

    where ``w_{ij}`` are the softmax-normalised cosine k-NN weights from
    :func:`build_cp_graph`. The mix coefficient is initialised so that the
    layer acts as a near-identity at the start of training (we do *not* want
    untrained edges to swamp the input feature).
    """

    def __init__(self, *, k: int = 8, tau_g: float = 1.0, mix_init: float = -2.0) -> None:
        super().__init__()
        if k <= 0:
            raise ValueError("k must be positive")
        self.k = int(k)
        self.tau_g = float(tau_g)
        self.mix = nn.Parameter(torch.tensor(mix_init))  # sigmoid(-2) ~ 0.12

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Refine features via one round of edge-weighted message passing.

        Parameters
        ----------
        features:
            ``(N, d)`` fused features.

        Returns
        -------
        ``(N, d)`` refined features.
        """
        if features.ndim != 2:
            raise ValueError(f"features must be 2-D; got shape {tuple(features.shape)}")
        n, _ = features.shape
        # Degenerate batch: a single sample cannot form a graph; pass through.
        if n < 2:
            return features
        idx, w = build_cp_graph(features, k=self.k, tau_g=self.tau_g)
        # Gather neighbour features: shape (N, k, d).
        neigh = features[idx]
        # Edge-weighted aggregation: (N, d).
        aggregated = (w.unsqueeze(-1) * neigh).sum(dim=1)
        scale = torch.sigmoid(self.mix)
        return features + scale * aggregated
