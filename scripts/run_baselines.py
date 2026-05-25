"""Run baseline models (shallow + deep) under the same protocol as our method.

Phase 2A stub: implementation lives in Phase 3 (see roadmap M3.1 ... M3.6).

Baselines (faithful re-implementations or official-code wrappers):
    * Shallow: SVM-RBF, RandomForest, kNN-5
    * Deep CNN: 3D-CNN, HybridSN
    * Deep Transformer: SpectralFormer, SSFTT
    * Deep Graph: Nonlocal-GCN
    * Mamba: MambaHSI
"""

from __future__ import annotations

import argparse
import sys

PHASE_TODO = "Phase 2A stub: scripts/run_baselines.py will be implemented in Phase 3 (M3.6)."


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument(
        "--baseline",
        type=str,
        default=None,
        choices=[
            "svm",
            "rf",
            "knn",
            "cnn3d",
            "hybridsn",
            "spectralformer",
            "ssftt",
            "nonlocal_gcn",
            "mambahsi",
            "all",
        ],
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    parse_args(argv)
    print(PHASE_TODO, file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
