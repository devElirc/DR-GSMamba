"""Evidential Prototype Head loss (Bayes-risk + annealed KL).

Implements §4 of ``docs/math/evidential_prototype_head.md`` exactly.

* :func:`bayes_risk_loss` -- closed form of
  ``E_{p ~ Dir(alpha)}[ ||y - p||_2^2 ]``.
* :func:`dirichlet_kl_to_uniform` -- closed form of
  ``KL(Dir(tilde_alpha) || Dir(1))`` (the "wrong-class" regulariser).
* :func:`evidential_loss` -- contract signature combining the two.
"""

from __future__ import annotations

import torch


def bayes_risk_loss(alpha: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Closed-form expected squared error under ``Dir(alpha)``.

    Implements the boxed Eq. of §4.1 of the math note::

        L_lik = sum_k [(y_k - p_k)^2 + p_k (1 - p_k) / (S + 1)]

    Parameters
    ----------
    alpha:
        ``(N, K)`` Dirichlet concentration vector (``alpha_k >= 1``).
    labels:
        ``(N,)`` integer class labels in ``[0, K)``.

    Returns
    -------
    ``(N,)`` per-sample Bayes-risk loss.
    """
    if alpha.ndim != 2:
        raise ValueError("alpha must be 2-D")
    n, k = alpha.shape
    if labels.shape != (n,):
        raise ValueError(f"labels must be ({n},); got {tuple(labels.shape)}")
    if (labels < 0).any() or (labels >= k).any():
        raise ValueError(f"labels must lie in [0, {k}); got [{labels.min()}, {labels.max()}]")
    s = alpha.sum(dim=1, keepdim=True)  # (N, 1)
    p = alpha / s  # (N, K)
    y = torch.zeros_like(p)
    y.scatter_(1, labels.view(-1, 1), 1.0)
    mean_term = (y - p).pow(2)  # (N, K)
    var_term = p * (1.0 - p) / (s + 1.0)  # (N, K)
    return (mean_term + var_term).sum(dim=1)  # (N,)


def dirichlet_kl_to_uniform(tilde_alpha: torch.Tensor) -> torch.Tensor:
    """Closed-form ``KL(Dir(tilde_alpha) || Dir(1))``.

    Implements the simplified formula in §4.2 of the math note:

        KL = lgamma(sum tilde_alpha) - sum lgamma(tilde_alpha)
             - lgamma(K)
             + sum (tilde_alpha - 1) * (digamma(tilde_alpha) - digamma(sum tilde_alpha))

    Parameters
    ----------
    tilde_alpha:
        ``(N, K)`` wrong-class Dirichlet concentration (>= 1).

    Returns
    -------
    ``(N,)`` per-sample KL divergence to the uniform Dirichlet on the simplex.
    """
    if tilde_alpha.ndim != 2:
        raise ValueError("tilde_alpha must be 2-D")
    s = tilde_alpha.sum(dim=1, keepdim=True)  # (N, 1)
    k = float(tilde_alpha.shape[1])
    psi_diff = torch.digamma(tilde_alpha) - torch.digamma(s)  # (N, K)
    kl = (
        torch.lgamma(s.squeeze(-1))
        - torch.lgamma(tilde_alpha).sum(dim=1)
        - torch.lgamma(torch.tensor(k, device=tilde_alpha.device, dtype=tilde_alpha.dtype))
        + ((tilde_alpha - 1.0) * psi_diff).sum(dim=1)
    )
    return kl  # (N,)


def evidential_loss(
    alpha: torch.Tensor,
    labels: torch.Tensor,
    kl_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Combined EPH loss: Bayes-risk + ``kl_weight * KL`` (annealed).

    Parameters
    ----------
    alpha:
        ``(N, K)`` Dirichlet concentration from :class:`EvidentialPrototypeHead`.
    labels:
        ``(N,)`` integer labels in ``[0, K)``.
    kl_weight:
        Current value of the annealed KL coefficient ``lambda_t`` from §4.3.

    Returns
    -------
    per_sample_loss:
        ``(N,)``, fed into CFA-GDRO solver.
    mean_loss:
        ``()``, the "mean EPH" term of the total loss.
    info:
        dict with keys ``lik`` (mean Bayes-risk), ``kl`` (mean KL),
        ``mean_vacuity`` (= ``K / S``), ``kl_weight`` (echoed back).
    """
    lik = bayes_risk_loss(alpha, labels)  # (N,)

    # Construct the wrong-class concentration tilde_alpha (§4.2 of math note):
    #   tilde_alpha_k = 1                if k == y
    #                   alpha_k          otherwise
    y = torch.zeros_like(alpha)
    y.scatter_(1, labels.view(-1, 1), 1.0)
    tilde_alpha = y + (1.0 - y) * alpha
    kl = dirichlet_kl_to_uniform(tilde_alpha)  # (N,)

    per_sample = lik + float(kl_weight) * kl
    mean_loss = per_sample.mean()

    with torch.no_grad():
        s = alpha.sum(dim=1)
        mean_vacuity = (float(alpha.shape[1]) / s).mean()

    info = {
        "lik": lik.mean().detach(),
        "kl": kl.mean().detach(),
        "mean_vacuity": mean_vacuity.detach(),
        "kl_weight": float(kl_weight),
    }
    return per_sample, mean_loss, info
