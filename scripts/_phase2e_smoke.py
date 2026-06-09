"""Phase 2E exit-criterion smoke run.

Per ``roadmap.md`` M2E.8 (relaxed by decision D-10):

    Smoke run: Indian Pines, alpha=0.3, gamma=1.0, 5 samples/class, 1 seed.

Three-criterion exit bundle:

    (i)   AA            >= 0.70
    (ii)  rare-class    >= 0.80
    (iii) determinism   identical metrics between two same-seed runs

Reason the original "OA >= 80%" gate was dropped: CFA-GDRO biases predictions
toward rare classes by design, which depresses common-class-dominated OA while
raising AA and rare-class accuracy. The AA / rare-class bundle measures the
*reliability* claim the paper actually makes.

Usage::

    python scripts/_phase2e_smoke.py [--epochs 200]

Prints one JSON line per run and a verdict line summarising the three
criteria. Returns 0 on success, non-zero if any criterion fails.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.train import main as train_main  # noqa: E402

AA_THRESHOLD = 0.70
RARE_THRESHOLD = 0.80


def _read_final(run_id: str) -> dict[str, Any]:
    return json.loads((ROOT / "outputs" / run_id / "final.json").read_text())


def _one_run(seed: int, epochs: int, run_id: str) -> int:
    return train_main(
        [
            "--dataset", "indian_pines",
            "--training", "label_5",
            "--seed", str(seed),
            "--epochs", str(epochs),
            "--alpha", "0.3",
            "--gamma", "1.0",
            "--val-every", "20",
            "--save-every", str(epochs),
            "--num-workers", "0",
            "--output-dir", run_id,
        ]
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-a", type=str, default="smoke_v1")
    parser.add_argument("--run-b", type=str, default="smoke_v2")
    args = parser.parse_args(argv)

    for run_id in (args.run_a, args.run_b):
        rc = _one_run(args.seed, args.epochs, run_id)
        if rc != 0:
            print(f"[smoke] run {run_id!r} failed with rc={rc}", file=sys.stderr)
            return rc

    a = _read_final(args.run_a)
    b = _read_final(args.run_b)

    m = a["final_metrics"]
    oa = float(m["OA"])
    aa = float(m["AA"])
    rare = float(m["rare_class"])
    kappa = float(m["kappa"])
    worst = float(m["worst_class"])
    ece = float(m["ECE_15"])

    deterministic = a["final_metrics"] == b["final_metrics"]
    pass_aa = aa >= AA_THRESHOLD
    pass_rare = rare >= RARE_THRESHOLD

    print(
        f"[smoke] OA={oa:.4f} AA={aa:.4f} kappa={kappa:.4f} rare={rare:.4f}"
        f" worst={worst:.4f} ECE15={ece:.4f}"
    )
    print(
        f"[smoke] criteria: AA>=0.70 {'PASS' if pass_aa else 'FAIL'}"
        f" | rare>=0.80 {'PASS' if pass_rare else 'FAIL'}"
        f" | determinism {'PASS' if deterministic else 'FAIL'}"
    )
    return 0 if (pass_aa and pass_rare and deterministic) else 2


if __name__ == "__main__":
    sys.exit(main())
