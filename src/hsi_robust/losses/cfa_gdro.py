"""Class-Frequency-Aware Group-DRO (CFA-GDRO) loss.

Implements the math contract of ``docs/math/cfa_gdro.md`` §1--§4 exactly:

* per-class adversarial caps ``c_k = pi_k^{-gamma} / (alpha * sum_j pi_j^{-gamma})``
  per Eq. boxed in §2;
* water-filling solver (§4.2) for the inner LP supremum;
* sub-gradient form (§4.3): ``L = (q_star.detach() * class_losses).sum()`` so
  gradients flow through the per-class losses only;
* EMA-stabilised inputs (§4.4): when ``ema_class_losses`` is passed in, the
  water-filling consumes the EMA estimate (stable signal across batches) while
  the gradient still flows through the live batch class losses.

This module is purely the *loss closure*. The EMA *update* is handled by the
separate ``training.ema_class_loss`` module so this function stays pure and
testable (no hidden state).
"""

from __future__ import annotations

import torch


def _per_class_means(
    per_sample_losses: torch.Tensor, labels: torch.Tensor, num_classes: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(class_losses_mean, counts)`` of shape ``(K,)`` each.

    ``class_losses_mean`` carries gradients back into ``per_sample_losses``.
    Classes absent from the batch get ``0`` (no gradient).
    """
    device = per_sample_losses.device
    dtype = per_sample_losses.dtype
    sums = torch.zeros(num_classes, device=device, dtype=dtype)
    sums = sums.scatter_add(0, labels, per_sample_losses)
    counts = torch.zeros(num_classes, device=device, dtype=torch.long)
    counts = counts.scatter_add(0, labels, torch.ones_like(labels, dtype=torch.long))
    safe = counts.clamp(min=1).to(dtype)
    return sums / safe, counts


def _water_filling(
    losses: torch.Tensor, caps: torch.Tensor
) -> tuple[torch.Tensor, float, int]:
    """Closed-form solver of ``max_q <q, losses>  s.t. 0 <= q <= caps, sum q = 1``.

    Inputs must be 1-D and live on the same device. Returns ``(q_star, nu, n_active)``
    where ``q_star`` is detached.
    """
    if losses.shape != caps.shape or losses.ndim != 1:
        raise ValueError("losses and caps must be 1-D tensors of equal shape")
    k = losses.shape[0]
    # Sort losses descending; both tensors must be on the same device.
    sorted_losses, sort_idx = torch.sort(losses, descending=True)
    sorted_caps = caps[sort_idx]

    q_sorted = torch.zeros_like(sorted_caps)
    remaining = 1.0
    nu = float(sorted_losses[0].item())  # safe default if k == 1
    n_active = 0
    for i in range(k):
        cap_i = float(sorted_caps[i].item())
        if remaining <= 1e-12:
            break
        if remaining <= cap_i + 1e-12:
            q_sorted[i] = remaining
            nu = float(sorted_losses[i].item())
            n_active += 1
            remaining = 0.0
            break
        # Saturated class.
        q_sorted[i] = cap_i
        remaining -= cap_i
        nu = float(sorted_losses[i].item())
        n_active += 1

    # Unsort.
    q_star = torch.zeros_like(losses)
    q_star.scatter_(0, sort_idx, q_sorted)
    return q_star.detach(), nu, n_active


def cfa_gdro_loss(
    per_sample_losses: torch.Tensor,
    labels: torch.Tensor,
    scene_freq: torch.Tensor,
    alpha: float,
    gamma: float,
    ema_class_losses: torch.Tensor | None = None,
    ema_seen: torch.Tensor | None = None,
    ema_momentum: float = 0.9,
) -> tuple[torch.Tensor, dict]:
    """Class-Frequency-Aware Group-DRO loss closure.

    See module docstring and ``docs/math/cfa_gdro.md`` §8 for the contract.

    Notes
    -----
    * ``ema_momentum`` appears in the signature so the contract matches the
      math note byte-for-byte; the EMA *update* itself lives in
      :class:`training.ema_class_loss.EMAClassLoss`.
    * When ``ema_class_losses`` is ``None`` (or all classes are unseen), the
      solver falls back to the *batch* per-class means; otherwise it uses the
      EMA estimate for the solver (gradient still flows through the batch means).
    """
    if per_sample_losses.ndim != 1:
        raise ValueError("per_sample_losses must be 1-D")
    if labels.shape != per_sample_losses.shape:
        raise ValueError("labels and per_sample_losses must share shape")
    if scene_freq.ndim != 1:
        raise ValueError("scene_freq must be 1-D")
    if not (0.0 < alpha <= 1.0):
        raise ValueError(f"alpha must lie in (0, 1]; got {alpha}")
    if gamma < 0:
        raise ValueError(f"gamma must be non-negative; got {gamma}")

    k_classes = scene_freq.shape[0]
    if labels.numel() and (labels.min() < 0 or labels.max() >= k_classes):
        raise ValueError(
            f"labels must be in [0, {k_classes - 1}]; got [{labels.min()}, {labels.max()}]"
        )

    # ---- per-class caps c_k (math note §2) --------------------------------
    # Move scene_freq to the device/dtype of the live losses to keep things tidy.
    pi = scene_freq.to(device=per_sample_losses.device, dtype=per_sample_losses.dtype)
    w = pi.clamp(min=1e-12).pow(-gamma)
    caps = w / (alpha * w.sum())
    caps = caps.detach()

    # ---- per-class live batch losses --------------------------------------
    class_losses_mean, counts = _per_class_means(per_sample_losses, labels, k_classes)
    seen = counts > 0  # (K,)

    # ---- solver input -----------------------------------------------------
    if ema_class_losses is not None:
        if ema_class_losses.shape != (k_classes,):
            raise ValueError(
                f"ema_class_losses must have shape ({k_classes},); got {tuple(ema_class_losses.shape)}"
            )
        solver_in = ema_class_losses.detach().to(
            device=class_losses_mean.device, dtype=class_losses_mean.dtype
        )
        # Fill any never-seen-yet entry (per ema_seen) with the live batch mean
        # so the solver does not get tricked into selecting classes that have
        # no real loss signal yet.
        if ema_seen is not None:
            ema_seen_b = ema_seen.to(device=class_losses_mean.device)
            solver_in = torch.where(ema_seen_b, solver_in, class_losses_mean.detach())
    else:
        solver_in = class_losses_mean.detach()

    q_star, nu, n_active = _water_filling(solver_in, caps)

    # ---- final loss (Danskin gradient) ------------------------------------
    loss = (q_star * class_losses_mean).sum()

    info: dict[str, object] = {
        "q_star": q_star,
        "class_losses": class_losses_mean.detach(),
        "caps": caps,
        "threshold_nu": float(nu),
        "active_set_size": int(n_active),
        "seen": seen.detach(),
        "counts": counts.detach(),
    }
    return loss, info
