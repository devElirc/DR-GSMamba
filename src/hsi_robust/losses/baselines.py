"""Baseline robust / imbalanced losses used by the Phase 7 comparison table.

Implemented here so the trainer can swap a single ``loss_name`` in the
configuration:

* :func:`ce_loss`            -- vanilla cross-entropy (reference baseline).
* :func:`focal_loss`         -- Lin et al. 2017 focal loss (gamma >= 0).
* :func:`sample_cvar_loss`   -- sample-level CVaR (top-alpha-fraction mean).
* :func:`sagawa_group_dro_loss`  -- worst-class group DRO with exponential
  weights (Sagawa et al., 2020 v1).

All five baselines plus CFA-GDRO consume **the same** ``per_sample_losses``
input (computed e.g. by CE or EPH), so the trainer can compare them
apples-to-apples.

Mathematical references:

* Focal loss: ``-(1 - p_y)^gamma_focal * log p_y``.
* Sample CVaR: average of the top ``ceil(alpha * N)`` largest losses in a batch.
* Sagawa group-DRO: ``L = sum_k q_k * bar_loss_k`` with multiplicative weight
  update ``q_k <- q_k * exp(eta * bar_loss_k)`` then renormalised (the inner
  step is computed lazily inside the loss closure so this function stays
  stateless).
"""

from __future__ import annotations

import math

import torch
from torch import nn


def ce_loss(logits: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Vanilla cross-entropy.

    Returns ``(per_sample_loss, mean_loss)``.
    """
    per_sample = nn.functional.cross_entropy(logits, labels, reduction="none")
    return per_sample, per_sample.mean()


def focal_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    gamma_focal: float = 2.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Focal loss with logits input (Lin et al. 2017).

    Returns ``(per_sample_loss, mean_loss)``.
    """
    if gamma_focal < 0:
        raise ValueError("gamma_focal must be non-negative")
    log_probs = nn.functional.log_softmax(logits, dim=-1)
    log_p_y = log_probs.gather(1, labels.view(-1, 1)).squeeze(1)
    p_y = log_p_y.exp().clamp(max=1.0 - 1e-7)
    per_sample = -((1.0 - p_y) ** gamma_focal) * log_p_y
    return per_sample, per_sample.mean()


def sample_cvar_loss(
    per_sample_losses: torch.Tensor, alpha: float = 0.3
) -> tuple[torch.Tensor, dict]:
    """Sample-level CVaR: mean of the top-``alpha`` fraction of losses.

    Parameters
    ----------
    per_sample_losses:
        ``(N,)`` per-sample loss tensor with gradient.
    alpha:
        Worst-fraction size in ``(0, 1]``. ``alpha = 1`` reduces to the mean.

    Returns
    -------
    loss:
        Scalar tensor (mean of the top fraction). Differentiable in the input.
    info:
        dict with ``num_kept`` and ``threshold``.
    """
    if per_sample_losses.ndim != 1:
        raise ValueError("per_sample_losses must be 1-D")
    if not (0.0 < alpha <= 1.0):
        raise ValueError(f"alpha must lie in (0, 1]; got {alpha}")
    n = per_sample_losses.shape[0]
    keep = max(1, math.ceil(alpha * n))
    top_vals, _ = torch.topk(per_sample_losses, k=keep, largest=True)
    loss = top_vals.mean()
    return loss, {"num_kept": int(keep), "threshold": float(top_vals.min().detach())}


def sagawa_group_dro_loss(
    per_sample_losses: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    q_state: torch.Tensor | None = None,
    eta: float = 0.01,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Sagawa et al. (2020) online group-DRO with exponential weights.

    Multiplicative weight update::

        q_k_{t+1} = q_k_t * exp(eta * bar_loss_k_t)
        q_{t+1}   = q_{t+1} / sum_k q_{t+1}_k

    The state ``q_state`` is the running ``q`` vector (initialised uniform when
    ``None``). The function returns the updated state so the trainer can carry
    it across batches without coupling this module to the trainer.

    Returns ``(loss, new_q_state, info)``.
    """
    if per_sample_losses.ndim != 1:
        raise ValueError("per_sample_losses must be 1-D")
    if labels.shape != per_sample_losses.shape:
        raise ValueError("labels and per_sample_losses must share shape")
    if eta <= 0:
        raise ValueError("eta must be positive")
    device = per_sample_losses.device
    dtype = per_sample_losses.dtype

    if q_state is None:
        q = torch.full((num_classes,), 1.0 / num_classes, device=device, dtype=dtype)
    else:
        q = q_state.to(device=device, dtype=dtype).clone()

    # Per-class live mean losses.
    sums = torch.zeros(num_classes, device=device, dtype=dtype)
    sums = sums.scatter_add(0, labels, per_sample_losses)
    counts = torch.zeros(num_classes, device=device, dtype=torch.long)
    counts = counts.scatter_add(0, labels, torch.ones_like(labels, dtype=torch.long))
    safe = counts.clamp(min=1).to(dtype)
    class_losses_mean = sums / safe  # gradient flows
    seen = counts > 0

    # Multiplicative update (only on seen classes).
    with torch.no_grad():
        update = torch.exp(eta * torch.where(seen, class_losses_mean, torch.zeros_like(class_losses_mean)))
        q_new = q * update
        q_new = q_new / q_new.sum().clamp(min=1e-12)

    loss = (q_new.detach() * class_losses_mean).sum()
    info = {
        "q": q_new.detach(),
        "active_set_size": int(seen.sum().item()),
        "class_losses": class_losses_mean.detach(),
    }
    return loss, q_new.detach(), info
