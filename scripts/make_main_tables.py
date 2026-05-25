"""Aggregate outputs/ into PR-formatted LaTeX tables.

Phase 2A stub: implementation lives in Phase 6 (see roadmap M6.4).

Produces:
    * paper/tables/main_results.tex      : main 4-dataset x 4-label grid
    * paper/tables/reliability.tex       : worst-class / rare-class / split-std
    * paper/tables/calibration.tex       : ECE / mean vacuity
    * paper/tables/efficiency.tex        : params / FLOPs / latency
"""

from __future__ import annotations

import argparse
import sys

PHASE_TODO = "Phase 2A stub: scripts/make_main_tables.py will be implemented in Phase 6 (M6.4)."


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs-dir", type=str, default="outputs")
    parser.add_argument("--tables-dir", type=str, default="paper/tables")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    parse_args(argv)
    print(PHASE_TODO, file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
