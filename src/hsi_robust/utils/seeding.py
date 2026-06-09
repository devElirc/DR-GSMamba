"""Deterministic seeding for Python, NumPy, and PyTorch.

This helper is the single source of truth for the seeding policy declared in
``configs/defaults.yaml`` (``deterministic: true``) and used by the Phase 2E
trainer and the Phase 2B exit-criterion demo.

``torch.use_deterministic_algorithms(True, warn_only=True)`` is used because a
small number of CPU kernels (notably ``index_put_`` with duplicate indices) do
not have a deterministic implementation; the ``warn_only`` flag turns the hard
crash into a warning so the training loop keeps going. CUDA determinism would
additionally require ``CUBLAS_WORKSPACE_CONFIG`` -- left as a TODO for the CUDA
deployment in Phase 6.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_everything(seed: int, *, deterministic: bool = True) -> None:
    """Seed Python, NumPy, and PyTorch RNGs and (optionally) enable deterministic ops.

    Parameters
    ----------
    seed:
        Integer seed; the same value reproduces the same RNG stream.
    deterministic:
        If True (default), enable ``torch.use_deterministic_algorithms`` and
        disable cuDNN benchmarking. Set to False only when speed matters more
        than reproducibility (not used in the paper experiments).
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
