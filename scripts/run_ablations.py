"""Run the ablation matrix from EXPERIMENT_PLAN.md.

Phase 2A stub: implementation lives in Phase 7 (see roadmap M7.1 ... M7.3).

Ablation families:
    * Component:  -CFA-GDRO  | -EPH  | -CP-Graph  | -OP-S4
    * Loss-only:  CE / Focal / CVaR / Group-DRO / Class-CVaR / CFA-GDRO(gamma)
    * Spectral backbone: OP-S4 / Transformer / 1D-CNN
"""

from __future__ import annotations

import argparse
import sys

PHASE_TODO = "Phase 2A stub: scripts/run_ablations.py will be implemented in Phase 7 (M7.1)."


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument(
        "--family",
        type=str,
        default="component",
        choices=["component", "loss", "backbone"],
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    parse_args(argv)
    print(PHASE_TODO, file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
