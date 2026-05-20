from __future__ import annotations

import argparse
import copy
from pathlib import Path

import numpy as np

from train import train_one_seed
from utils.common import deep_update, load_config, save_json


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
        if any(key not in r for r in results):
            continue
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


def run_config(cfg: dict) -> dict:
    seeds = cfg["training"]["seeds"]
    results = []
    for seed in seeds:
        results.append(train_one_seed(cfg, seed))
    summary = summarize(results)
    out_dir = Path(cfg["outputs"]["dir"])
    save_json({"seeds": seeds, "summary": summary, "runs": results}, out_dir / "summary.json")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/synthetic.yaml")
    parser.add_argument("--seeds", type=int, default=None, help="Override number of seeds from config.")
    parser.add_argument("--samples-per-class", type=int, default=None, help="Override fixed labels per class.")
    parser.add_argument("--output-dir", default=None, help="Override output directory.")
    parser.add_argument("--disable-dro", action="store_true", help="Set robust loss weight to zero for DRO ablation.")
    parser.add_argument("--disable-prototype", action="store_true", help="Disable prototype head and losses.")
    parser.add_argument("--disable-uncertainty", action="store_true", help="Set uncertainty loss weight to zero.")
    parser.add_argument("--disable-graph", action="store_true", help="Disable graph context and graph smoothness.")
    parser.add_argument("--disable-spectral", action="store_true", help="Disable spectral state-space branch.")
    parser.add_argument(
        "--spectral-backend",
        choices=["mamba", "selective_scan", "ssm", "transformer", "cnn"],
        default=None,
        help="Replace the spectral branch for controlled ablations.",
    )
    args = parser.parse_args()
    cfg = copy.deepcopy(load_config(args.config))
    updates: dict = {}
    if args.seeds is not None:
        updates.setdefault("training", {})["seeds"] = list(range(args.seeds))
    if args.samples_per_class is not None:
        updates.setdefault("dataset", {})["samples_per_class"] = args.samples_per_class
    if args.output_dir is not None:
        updates.setdefault("outputs", {})["dir"] = args.output_dir
    if args.disable_dro:
        updates.setdefault("loss", {})["robust_weight"] = 0.0
    if args.disable_prototype:
        updates.setdefault("model", {})["use_prototype"] = False
        updates.setdefault("loss", {})["prototype_weight"] = 0.0
        updates.setdefault("loss", {})["prototype_supervised_weight"] = 0.0
    if args.disable_uncertainty:
        updates.setdefault("loss", {})["uncertainty_weight"] = 0.0
    if args.disable_graph:
        updates.setdefault("model", {})["use_graph"] = False
        updates.setdefault("loss", {})["graph_smooth_weight"] = 0.0
    if args.disable_spectral:
        updates.setdefault("model", {})["use_spectral"] = False
    if args.spectral_backend is not None:
        updates.setdefault("model", {})["spectral_backend"] = args.spectral_backend
    deep_update(cfg, updates)
    summary = run_config(cfg)
    print(summary)


if __name__ == "__main__":
    main()
