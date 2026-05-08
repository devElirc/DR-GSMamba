from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from train import train_one_seed
from utils.common import load_config, save_json


def summarize(results: list[dict]) -> dict:
    keys = [
        "oa",
        "aa",
        "kappa",
        "macro_f1",
        "worst_class_accuracy",
        "rare_class_accuracy",
        "ece",
        "mean_uncertainty",
    ]
    summary = {}
    for key in keys:
        values = np.asarray([r[key] for r in results], dtype=float)
        mean = float(values.mean())
        std = float(values.std(ddof=0))
        summary[key] = {
            "mean": mean,
            "std": std,
            "coefficient_of_variation": float(std / (abs(mean) + 1e-12)),
            "worst_split": float(values.min()),
        }
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/synthetic.yaml")
    parser.add_argument("--seeds", type=int, default=None, help="Override number of seeds from config.")
    args = parser.parse_args()
    cfg = load_config(args.config)
    seeds = cfg["training"]["seeds"]
    if args.seeds is not None:
        seeds = list(range(args.seeds))
    results = []
    for seed in seeds:
        results.append(train_one_seed(cfg, seed))
    summary = summarize(results)
    out_dir = Path(cfg["outputs"]["dir"])
    save_json({"seeds": seeds, "summary": summary, "runs": results}, out_dir / "summary.json")
    print(summary)


if __name__ == "__main__":
    main()
