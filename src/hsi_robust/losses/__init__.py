"""Loss functions (Phase 2D).

Public surface:

* :func:`cfa_gdro_loss`   -- contribution from ``docs/math/cfa_gdro.md`` §8.
* :func:`evidential_loss` -- Bayes-risk + annealed KL from
  ``docs/math/evidential_prototype_head.md`` §8.
* :func:`bayes_risk_loss`, :func:`dirichlet_kl_to_uniform` -- the two pieces
  of the evidential loss, exposed so tests can probe each independently.
* :func:`cp_graph_loss`   -- one-directional KL consistency from
  ``docs/math/cp_graph.md`` §8.
* :func:`ce_loss`, :func:`focal_loss`, :func:`sample_cvar_loss`,
  :func:`sagawa_group_dro_loss` -- baseline objectives.
"""

from __future__ import annotations

from hsi_robust.losses.baselines import (
    ce_loss,
    focal_loss,
    sagawa_group_dro_loss,
    sample_cvar_loss,
)
from hsi_robust.losses.cfa_gdro import cfa_gdro_loss
from hsi_robust.losses.cp_graph import cp_graph_loss
from hsi_robust.losses.evidential import (
    bayes_risk_loss,
    dirichlet_kl_to_uniform,
    evidential_loss,
)

__all__ = [
    "bayes_risk_loss",
    "ce_loss",
    "cfa_gdro_loss",
    "cp_graph_loss",
    "dirichlet_kl_to_uniform",
    "evidential_loss",
    "focal_loss",
    "sagawa_group_dro_loss",
    "sample_cvar_loss",
]
