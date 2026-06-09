"""Lightweight YAML config loader.

Deliberately minimal: we do **not** depend on Hydra at runtime (see
roadmap M2A.3). For Phase 2B we expose just :func:`load_yaml`, which the data
pipeline calls with paths like ``configs/datasets/indian_pines.yaml``.

Composition of the top-level ``configs/defaults.yaml`` (which references
nested ``configs/{datasets,model,training}/<name>.yaml`` files via a
``defaults:`` block) is added in Phase 2E when the training scripts need it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Read a YAML file and return its top-level mapping.

    Raises
    ------
    FileNotFoundError:
        If ``path`` does not exist.
    TypeError:
        If the top-level YAML node is not a mapping (we never write list-rooted configs).
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"config file not found: {p}")
    with p.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise TypeError(f"config root must be a mapping; got {type(data).__name__}")
    return data
