"""Training utilities.

Public surface (to be populated in Phase 2E):

* ``Trainer``            - deterministic train loop with checkpointing.
* ``build_optimizer``    - AdamW + cosine schedule + warmup.
* ``EMAClassLossTracker`` - the exponential moving average over per-class losses
  required by ``docs/math/cfa_gdro.md`` Section 4.4.
"""

from __future__ import annotations

__all__: list[str] = []
