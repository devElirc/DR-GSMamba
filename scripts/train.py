"""Train a single model on one dataset / one seed / one label setting.

Phase 2A stub: implementation lives in Phase 2E (see roadmap M2E.7).
"""

from __future__ import annotations

import argparse
import sys

PHASE_TODO = "Phase 2A stub: scripts/train.py will be implemented in Phase 2E (M2E.7)."


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default=None, help="path to a YAML config")
    parser.add_argument("--override", action="append", default=[], help="key=value overrides")
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    parse_args(argv)
    print(PHASE_TODO, file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
