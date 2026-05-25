"""Run the robustness protocol from EXPERIMENT_PLAN.md.

Phase 2A stub: implementation lives in Phase 7 (see roadmap M7.4).

Protocols:
    * band-permutation : random band shuffle at test time (Indian Pines, Pavia U)
    * label-noise      : 5 / 10 / 20 percent flipped labels at train time
    * ood-calibration  : zero out 20 percent of bands; verify vacuity rises
"""

from __future__ import annotations

import argparse
import sys

PHASE_TODO = "Phase 2A stub: scripts/run_robustness.py will be implemented in Phase 7 (M7.4)."


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument(
        "--protocol",
        type=str,
        default="all",
        choices=["band_permutation", "label_noise", "ood_calibration", "all"],
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    parse_args(argv)
    print(PHASE_TODO, file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
