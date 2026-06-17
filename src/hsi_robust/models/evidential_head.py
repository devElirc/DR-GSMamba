"""Evidential Prototype Head (EPH) -- Contribution 2.

Implements §2 of ``docs/math/evidential_prototype_head.md``:

* evidence ``e_k = softplus(tau * cos(f, m_k))``,
* Dirichlet concentration ``alpha_k = e_k + 1`` with total ``S = sum_k alpha_k``,
* predictive probabilities ``p_k = alpha_k / S``,
* closed-form uncertainty decomposition into vacuity ``K / S`` and aleatoric
  ``sum_k p_k (1 - p_k)``.

The temperature ``tau`` is a *learnable* scalar parameterised through a
softplus and clipped (via a hard ``clamp``) to ``[tau_min, tau_max] = [1, 30]``
per §1 of the math note.

Prototypes ``m_k`` are unit-norm-ish at initialisation thanks to small Gaussian
init; they are *not* enforced to be unit norm during training because the
cosine in the evidence formula already L2-normalises them on the fly.
"""

from __future__ import annotations

import torch
from torch import nn


class EvidentialPrototypeHead(nn.Module):
    """Prototype-based Dirichlet-evidential classifier head.

    Parameters
    ----------
    feature_dim:
        Dimension ``d`` of the fused per-pixel feature.
    num_classes:
        Number of classes ``K``.
    tau_init:
        Initial value of the temperature ``tau`` (after the softplus + clip).
    tau_min, tau_max:
        Hard clip range for ``tau``. Matches §1 of the math note.
    prototype_init_scale:
        Std-dev of the zero-mean Gaussian initialisation for the prototypes.
    """

    def __init__(
        self,
        *,
        feature_dim: int,
        num_classes: int,
        tau_init: float = 10.0,
        tau_min: float = 1.0,
        tau_max: float = 30.0,
        prototype_init_scale: float = 0.02,
    ) -> None:
        super().__init__()
        if feature_dim <= 0 or num_classes <= 0:
            raise ValueError("feature_dim and num_classes must be positive")
        if not (0 < tau_min < tau_max):
            raise ValueError("require 0 < tau_min < tau_max")
        self.feature_dim = int(feature_dim)
        self.num_classes = int(num_classes)
        self.tau_min = float(tau_min)
        self.tau_max = float(tau_max)

        # Learnable prototypes m_k in R^d.
        self.prototypes = nn.Parameter(prototype_init_scale * torch.randn(num_classes, feature_dim))
        # Parameterise tau through softplus on a raw scalar so it stays positive.
        # Use the inverse softplus of tau_init to get the right starting point.
        raw = float(torch.log(torch.expm1(torch.tensor(float(tau_init)))).item())
        self.raw_tau = nn.Parameter(torch.tensor(raw))

    def temperature(self) -> torch.Tensor:
        """Return the clipped, positive temperature ``tau``."""
        tau = nn.functional.softplus(self.raw_tau)
        return tau.clamp(min=self.tau_min, max=self.tau_max)

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        """Map fused features to the full EPH output bundle.

        Parameters
        ----------
        features:
            ``(N, d)`` fused features ``f_i``.

        Returns
        -------
        dict with keys (all 2-D except ``vacuity`` / ``aleatoric`` which are 1-D):
            * ``evidence``   : ``(N, K)`` softplus evidence ``e_k``.
            * ``alpha``      : ``(N, K)`` Dirichlet concentration ``alpha_k = e_k + 1``.
            * ``probs``      : ``(N, K)`` predictive probabilities ``alpha_k / S``.
            * ``vacuity``    : ``(N,)`` epistemic uncertainty ``K / S``.
            * ``aleatoric``  : ``(N,)`` aleatoric uncertainty ``sum_k p_k(1-p_k)``.
            * ``cos``        : ``(N, K)`` raw cosine similarities (useful for debugging).
            * ``temperature``: ``()`` scalar tau (useful for logging).
        """
        if features.ndim != 2 or features.shape[-1] != self.feature_dim:
            raise ValueError(
                f"features must be (N, {self.feature_dim}); got {tuple(features.shape)}"
            )
        f_norm = nn.functional.normalize(features, p=2, dim=1, eps=1e-12)
        m_norm = nn.functional.normalize(self.prototypes, p=2, dim=1, eps=1e-12)
        cos = f_norm @ m_norm.t()  # (N, K)
        tau = self.temperature()
        evidence = nn.functional.softplus(tau * cos)  # (N, K), >= 0
        alpha = evidence + 1.0  # (N, K), >= 1
        s = alpha.sum(dim=1, keepdim=True)  # (N, 1)
        probs = alpha / s  # (N, K)
        vacuity = float(self.num_classes) / s.squeeze(-1)  # (N,), in (0, 1]
        aleatoric = (probs * (1.0 - probs)).sum(dim=1)  # (N,)

        return {
            "evidence": evidence,
            "alpha": alpha,
            "probs": probs,
            "vacuity": vacuity,
            "aleatoric": aleatoric,
            "cos": cos,
            "temperature": tau,
        }
