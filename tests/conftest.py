"""Shared pytest fixtures.

Goals:
    * Every test starts from the same RNG state (deterministic-seed fixture).
    * Tiny tensor fixtures avoid the slow path of building real datasets.
    * Imbalanced scene-frequency fixtures exercise the CFA-GDRO code paths
      with the same shape that the real Indian Pines / Pavia U scenes have.

Loss / model / data tests pull from these; do not duplicate the fixtures
elsewhere.
"""

from __future__ import annotations

import os
import random
from collections.abc import Iterator

import numpy as np
import pytest
import torch

# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _deterministic_rngs() -> Iterator[None]:
    """Seed every RNG before each test, restore the state on exit.

    Auto-used so a stray test that forgets to seed cannot pollute the next.
    Uses ``warn_only=True`` to avoid blowing up on torch ops without a
    deterministic implementation (we will tighten this in Phase 2D for the
    loss tests).
    """
    seed = 0
    os.environ["PYTHONHASHSEED"] = str(seed)

    py_state = random.getstate()
    np_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)

    try:
        yield
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
        torch.set_rng_state(torch_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------


@pytest.fixture
def cpu_device() -> torch.device:
    """CPU device for all unit tests; GPU is reserved for full smoke tests."""
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Tiny tensors (Phase 2C / 2D tests use these)
# ---------------------------------------------------------------------------


@pytest.fixture
def tiny_batch() -> dict[str, object]:
    """Returns a small batch of fused features and labels.

    Shapes:
        features : (N=8, d=16)
        logits   : (N=8, K=4)
        labels   : (N=8,)  values in {0, 1, 2, 3}
    """
    n, k, d = 8, 4, 16
    return {
        "features": torch.randn(n, d),
        "logits": torch.randn(n, k),
        "labels": torch.randint(0, k, (n,)),
        "N": n,
        "K": k,
        "d": d,
    }


@pytest.fixture
def per_sample_losses_4() -> torch.Tensor:
    """Synthetic per-sample losses with deliberately uneven class means.

    Shape (N=8,). Class 0 has low loss, class 3 has high loss; this lets
    CFA-GDRO tests verify that the solver puts mass on the high-loss
    classes first.
    """
    return torch.tensor([0.10, 0.20, 0.50, 0.40, 0.90, 1.10, 1.00, 0.30])


@pytest.fixture
def labels_4() -> torch.Tensor:
    """Labels in {0, 1, 2, 3} matched to :func:`per_sample_losses_4` above."""
    return torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])


# ---------------------------------------------------------------------------
# Scene frequencies (imbalanced)
# ---------------------------------------------------------------------------


@pytest.fixture
def scene_freq_4() -> torch.Tensor:
    """Imbalanced 4-class scene frequencies summing to 1.

    The ratio pi_max / pi_min = 10 is in the same order of magnitude as
    Indian Pines (about pi_max / pi_min ~= 90) but small enough to keep
    the CFA-GDRO water-filling tests easy to reason about.
    """
    pi = torch.tensor([0.5, 0.3, 0.15, 0.05])
    assert torch.allclose(pi.sum(), torch.tensor(1.0))
    return pi
