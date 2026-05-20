from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets import load_hsi
from utils.common import load_config


def save_map(arr: np.ndarray, out: Path, title: str, cmap: str = "tab20") -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 6))
    plt.imshow(arr, cmap=cmap, interpolation="nearest")
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out, dpi=300)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Generate ground-truth, prediction, and error maps from a metrics.json file.")
    parser.add_argument("--config", default="configs/indian_pines.yaml")
    parser.add_argument("--metrics", default="outputs/indian_pines/seed_0/metrics.json")
    parser.add_argument("--out-dir", default="paper/figures/maps")
    args = parser.parse_args()

    cfg = load_config(args.config)
    data = load_hsi(cfg, seed=0)
    metrics = json.loads(Path(args.metrics).read_text(encoding="utf-8"))["test"]
    indices = np.asarray(metrics["prediction_indices"], dtype=int)
    pred = np.asarray(metrics["prediction_labels"], dtype=int) + 1
    true = np.asarray(metrics["prediction_true"], dtype=int) + 1
    uncertainty = np.asarray(metrics.get("prediction_uncertainty", np.zeros_like(pred, dtype=float)), dtype=float)

    pred_map = np.zeros_like(data.labels, dtype=np.int64)
    true_eval_map = np.zeros_like(data.labels, dtype=np.int64)
    error_map = np.zeros_like(data.labels, dtype=np.int64)
    uncertainty_map = np.zeros_like(data.labels, dtype=float)
    width = data.labels.shape[1]
    rows = indices // width
    cols = indices % width
    pred_map[rows, cols] = pred
    true_eval_map[rows, cols] = true
    error_map[rows, cols] = (pred != true).astype(np.int64)
    uncertainty_map[rows, cols] = uncertainty

    out_dir = Path(args.out_dir)
    save_map(data.labels, out_dir / "ground_truth.png", "Ground Truth")
    save_map(pred_map, out_dir / "prediction_map.png", "Prediction Map")
    save_map(true_eval_map, out_dir / "test_ground_truth.png", "Test Ground Truth")
    save_map(error_map, out_dir / "error_map.png", "Error Map", cmap="Reds")
    save_map(uncertainty_map, out_dir / "uncertainty_map.png", "Uncertainty Map", cmap="magma")
    print(f"Saved classification maps to {out_dir}")


if __name__ == "__main__":
    main()
