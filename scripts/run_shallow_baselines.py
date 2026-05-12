from __future__ import annotations

import argparse
import copy
from pathlib import Path
import sys

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets import create_dataloaders
from run_experiments import summarize
from utils.common import deep_update, load_config, save_json, set_seed
from utils.metrics import classification_metrics


def collect_center_spectra(loader):
    x_values, y_values = [], []
    for batch in loader:
        x_values.append(batch["spectrum"].numpy())
        y_values.append(batch["label"].numpy())
    return np.concatenate(x_values, axis=0), np.concatenate(y_values, axis=0)


def build_model(name: str, seed: int):
    if name == "svm_rbf":
        return SVC(C=10.0, gamma="scale")
    if name == "random_forest":
        return RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1, class_weight="balanced")
    if name == "knn":
        return KNeighborsClassifier(n_neighbors=5)
    raise ValueError(f"Unknown baseline: {name}")


def run_baseline(cfg: dict, model_name: str, seed: int) -> dict:
    set_seed(seed)
    loaders, num_classes = create_dataloaders(cfg, seed)
    train_x, train_y = collect_center_spectra(loaders["train"])
    test_x, test_y = collect_center_spectra(loaders["test"])
    model = build_model(model_name, seed)
    model.fit(train_x, train_y)
    pred = model.predict(test_x)
    prob = model.predict_proba(test_x) if hasattr(model, "predict_proba") else None
    return classification_metrics(test_y, pred, num_classes, y_prob=prob)


def main():
    parser = argparse.ArgumentParser(description="Run classical shallow HSI baselines on center spectra.")
    parser.add_argument("--config", default="configs/synthetic.yaml")
    parser.add_argument("--models", nargs="+", default=["svm_rbf", "random_forest", "knn"])
    parser.add_argument("--seeds", type=int, default=None)
    parser.add_argument("--samples-per-class", type=int, default=None)
    parser.add_argument("--out-dir", default="outputs/shallow_baselines")
    args = parser.parse_args()

    cfg = copy.deepcopy(load_config(args.config))
    if args.seeds is not None:
        cfg["training"]["seeds"] = list(range(args.seeds))
    if args.samples_per_class is not None:
        deep_update(cfg, {"dataset": {"samples_per_class": args.samples_per_class}})

    out_dir = Path(args.out_dir)
    all_reports = {}
    for model_name in args.models:
        runs = [run_baseline(cfg, model_name, seed) for seed in cfg["training"]["seeds"]]
        report = {"model": model_name, "seeds": cfg["training"]["seeds"], "summary": summarize(runs), "runs": runs}
        all_reports[model_name] = report
        save_json(report, out_dir / model_name / "summary.json")

    save_json({"config": args.config, "baselines": all_reports}, out_dir / "summary.json")
    print(f"Saved shallow baseline reports to {out_dir}")


if __name__ == "__main__":
    main()
