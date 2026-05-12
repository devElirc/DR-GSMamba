from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import NormalDist

import numpy as np


def load_seed_metrics(path: Path, metric: str) -> np.ndarray:
    data = json.loads(path.read_text(encoding="utf-8"))
    runs = data.get("runs", [])
    if not runs:
        raise ValueError(f"No per-seed runs found in {path}")
    return np.asarray([run[metric] for run in runs], dtype=float)


def paired_ttest(x: np.ndarray, y: np.ndarray) -> dict:
    if x.shape != y.shape:
        raise ValueError("Paired test requires the same number of seeds in both files.")
    diff = x - y
    n = diff.size
    mean = float(diff.mean())
    if n < 2:
        return {"n": int(n), "mean_difference": mean, "t": None, "p_approx": None}
    std = float(diff.std(ddof=1))
    if std == 0.0:
        return {"n": int(n), "mean_difference": mean, "t": None, "p_approx": 0.0 if mean != 0.0 else 1.0}
    t_value = mean / (std / np.sqrt(n))
    p_approx = 2.0 * (1.0 - NormalDist().cdf(abs(t_value)))
    return {"n": int(n), "mean_difference": mean, "t": float(t_value), "p_approx_normal": float(p_approx)}


def main():
    parser = argparse.ArgumentParser(description="Compare two multi-seed summary.json files.")
    parser.add_argument("--full", required=True, help="Path to full-model summary.json")
    parser.add_argument("--baseline", required=True, help="Path to baseline or ablation summary.json")
    parser.add_argument("--metrics", nargs="+", default=["oa", "macro_f1", "worst_class_accuracy", "rare_class_accuracy"])
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    full_path = Path(args.full)
    baseline_path = Path(args.baseline)
    report = {"full": str(full_path), "baseline": str(baseline_path), "metrics": {}}
    for metric in args.metrics:
        full_values = load_seed_metrics(full_path, metric)
        baseline_values = load_seed_metrics(baseline_path, metric)
        report["metrics"][metric] = paired_ttest(full_values, baseline_values)

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
