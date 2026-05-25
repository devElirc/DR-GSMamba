"""Cross-cutting utilities: seeding, config loading, logging.

Public surface (to be populated alongside Phase 2B onwards):

* ``seed_everything``    - deterministic seeding for python / numpy / torch / cuda.
* ``load_config``        - YAML loader with simple `defaults` resolution
  (no Hydra runtime dependency, see roadmap M2A.3).
* ``get_logger``         - JSON-line logger writing to ``outputs/<run_id>/log.jsonl``.
"""

from __future__ import annotations

__all__: list[str] = []
