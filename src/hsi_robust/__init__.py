"""hsi_robust -- reproducible reference code for the CFA-GDRO + EPH HSI paper.

This is the top-level package. The contribution claims are deliberately limited to:

* :mod:`hsi_robust.losses.cfa_gdro`  - Class-Frequency-Aware Group-DRO objective.
* :mod:`hsi_robust.losses.evidential` - Evidential prototype head losses.

Everything else (backbone, graph, training loop, baselines) is a design choice that is
ablated, not claimed. The mathematical specification lives in ``docs/math/*.md`` and is
the source of truth; this package implements those contracts.

See ``roadmap.md`` for the phase plan and ``EXPERIMENT_PLAN.md`` for the experimental
design.
"""

from hsi_robust._version import __version__

__all__ = ["__version__"]
