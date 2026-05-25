"""Render every figure for the paper from outputs/.

Phase 2A stub: implementation lives in Phase 7 (see roadmap M7.5).

Produces:
    * paper/figs/architecture.pdf
    * paper/figs/reliability_<dataset>.pdf  (reliability diagrams)
    * paper/figs/stability_<metric>.pdf     (boxplots over 10 seeds)
    * paper/figs/qualitative_<dataset>.pdf  (GT / pred / error / vacuity / aleatoric)
"""

from __future__ import annotations

import argparse
import sys

PHASE_TODO = "Phase 2A stub: scripts/make_figures.py will be implemented in Phase 7 (M7.5)."


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs-dir", type=str, default="outputs")
    parser.add_argument("--figs-dir", type=str, default="paper/figs")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    parse_args(argv)
    print(PHASE_TODO, file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
