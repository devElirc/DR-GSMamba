"""Loss functions.

Public surface (to be populated in Phase 2D):

* ``cfa_gdro_loss``     - the contribution objective from
  ``docs/math/cfa_gdro.md`` Section 8.
* ``evidential_loss``   - Bayes-risk + annealed KL from
  ``docs/math/evidential_prototype_head.md`` Section 8.
* ``cp_graph_loss``     - in-batch consistency regulariser from
  ``docs/math/cp_graph.md`` Section 8.
* ``ce_loss`` / ``focal_loss`` / ``cvar_loss`` / ``group_dro_loss`` - baseline
  objectives used in the loss-only ablation matrix.
"""

from __future__ import annotations

__all__: list[str] = []
