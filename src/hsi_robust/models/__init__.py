"""Backbone and head modules.

Public surface (to be populated in Phase 2C):

* ``OPS4SpectralEncoder``      - bidirectional selective scan on raw spectral bands.
* ``SpatialCNNStem``           - compact 2D-CNN on PCA-reduced patches.
* ``CPGraphPropagation``       - in-batch k-NN graph reasoning (design choice).
* ``EvidentialPrototypeHead``  - the Dirichlet-evidential head specified in
  ``docs/math/evidential_prototype_head.md`` Section 8.
* ``DRGSMamba``                - full model assembling the above.

None of these are exposed yet; the math contracts they will obey live in
``docs/math/``.
"""

from __future__ import annotations

__all__: list[str] = []
