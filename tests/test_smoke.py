"""Phase 2A smoke tests.

These are intentionally trivial. They exist only to:

1. Prove ``pip install -e .`` produced an importable package, and
2. Keep pytest's exit code at 0 (without any tests pytest exits with 5).

Real unit tests appear in Phase 2B (data), Phase 2C (models), and Phase 2D
(losses).
"""

from __future__ import annotations

import importlib

import pytest


def test_top_level_import() -> None:
    """``import hsi_robust`` succeeds and exposes a version string."""
    module = importlib.import_module("hsi_robust")
    assert hasattr(module, "__version__")
    assert isinstance(module.__version__, str)
    assert module.__version__ == "0.0.1"


@pytest.mark.parametrize(
    "subpackage",
    ["data", "models", "losses", "training", "eval", "utils"],
)
def test_subpackage_import(subpackage: str) -> None:
    """Every Phase 2A subpackage is importable."""
    importlib.import_module(f"hsi_robust.{subpackage}")


def test_fixtures_compose(tiny_batch: dict[str, object], scene_freq_4) -> None:
    """Sanity-check that the fixtures in conftest.py wire correctly."""
    assert tiny_batch["N"] == 8
    assert tiny_batch["K"] == 4
    assert scene_freq_4.shape == (4,)
