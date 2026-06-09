"""Phase 2B exit-criterion demo script.

The roadmap M2B exit criterion says:

    A 5-line script can produce a (spectrum, patch, label) mini-batch from
    Indian Pines with seed=0, samples_per_class=5, and the produced split is
    identical on a second invocation.

Run from the repository root:

    python scripts/_phase2b_exit_check.py

If the script prints "Phase 2B exit check passed." with exit code 0, the
exit criterion is met.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import torch

from hsi_robust.data import build_split
from hsi_robust.utils import load_yaml


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent
    config = load_yaml(repo_root / "configs" / "datasets" / "indian_pines.yaml")
    config["samples_per_class"] = 5
    config["data_dir"] = str(repo_root / config["data_dir"])

    # ---- The "5-line" demo (between the markers) ---------------------------
    # 1                                                            (config above)
    # 2                                                            (config above)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        artifacts = build_split(config, seed=0)
    spectrum, patch, label = artifacts.train_dataset[0]
    print(
        f"spectrum: {tuple(spectrum.shape)} {spectrum.dtype}; "
        f"patch: {tuple(patch.shape)} {patch.dtype}; label: {label}"
    )
    # ------------------------------------------------------------------------

    # Determinism check: same seed -> same first item.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        artifacts2 = build_split(config, seed=0)
    spectrum2, patch2, label2 = artifacts2.train_dataset[0]
    assert torch.equal(spectrum, spectrum2), "spectrum differs across invocations"
    assert torch.equal(patch, patch2), "patch differs across invocations"
    assert label == label2, "label differs across invocations"
    np.testing.assert_array_equal(artifacts.train_labels, artifacts2.train_labels)
    np.testing.assert_array_equal(artifacts.train_positions, artifacts2.train_positions)

    print(
        f"train size = {len(artifacts.train_dataset)} "
        f"(= 16 classes x 5 spc); "
        f"test size = {len(artifacts.test_dataset)}; "
        f"scene_freq min/max = {float(artifacts.scene_freq.min()):.4f} / "
        f"{float(artifacts.scene_freq.max()):.4f}"
    )
    print("Phase 2B exit check passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
