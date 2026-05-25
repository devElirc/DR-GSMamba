"""Run a single configuration across ten random seeds and aggregate metrics.

Phase 2A stub: implementation lives in Phase 6 (see roadmap M6.1).

Drives:
    * the main results grid (4 datasets x 4 label settings x 10 seeds)
    * the reliability protocol (mean +/- std, CoV, worst-split)
"""

from __future__ import annotations

import argparse
import sys

PHASE_TODO = "Phase 2A stub: scripts/run_multi_seed.py will be implemented in Phase 6 (M6.1)."


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, required=False, default=None)
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    )
    parser.add_argument("--label-setting", type=int, default=None, choices=[3, 5, 10, 20])
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    parse_args(argv)
    print(PHASE_TODO, file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
