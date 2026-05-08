from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", default="outputs/synthetic/seed_0/metrics.json")
    parser.add_argument("--out", default="paper/figures/confusion_matrix.png")
    args = parser.parse_args()
    metrics = json.loads(Path(args.metrics).read_text(encoding="utf-8"))["test"]
    cm = metrics["confusion_matrix"]
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, cmap="Blues")
    plt.colorbar(fraction=0.046, pad=0.04)
    for row in range(len(cm)):
        for col in range(len(cm[row])):
            plt.text(col, row, str(cm[row][col]), ha="center", va="center", fontsize=7)
    plt.xlabel("Predicted")
    plt.ylabel("Ground Truth")
    plt.xticks(range(len(cm)))
    plt.yticks(range(len(cm)))
    plt.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=300)


if __name__ == "__main__":
    main()
