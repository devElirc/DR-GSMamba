"""Run one or more Phase-3 baselines on a single dataset / seed / label setting.

Usage examples::

    # Run every baseline on Indian Pines at 5 samples/class, seed 0.
    python scripts/run_baselines.py --dataset indian_pines --baselines all

    # Single baseline, custom seed and label budget.
    python scripts/run_baselines.py --dataset indian_pines \\
                                    --baselines hybridsn --seed 1 \\
                                    --samples-per-class 10

    # Shallow only (fast).
    python scripts/run_baselines.py --dataset indian_pines --baselines shallow

The script writes one ``outputs/<dataset>_<baseline>_seed<seed>/`` directory per
baseline with ``final.json`` containing the metrics and (for deep baselines)
``metrics.json`` + checkpoints. A summary table is printed to stdout and
saved to ``outputs/baselines_summary_<dataset>_seed<seed>.json``.

This is the M3.6 entry point. Reproduction-vs-paper validation lives in
Phase 5 (M5.2).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from hsi_robust.baselines import (
    BaselineTrainConfig,
    BaselineTrainer,
    all_baseline_names,
    deep_baseline_names,
    is_deep_baseline,
    is_shallow_baseline,
    make_deep_baseline,
    make_shallow_model,
    shallow_baseline_names,
)
from hsi_robust.data import build_split
from hsi_robust.eval.calibration import expected_calibration_error
from hsi_robust.eval.metrics import compute_metrics
from hsi_robust.utils import load_yaml, seed_everything

ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = ROOT / "configs"


def _resolve_baselines(name_list: list[str]) -> list[str]:
    """Expand selection keywords like 'all', 'shallow', 'deep'."""
    resolved: list[str] = []
    for raw in name_list:
        n = raw.strip().lower()
        if n == "all":
            resolved.extend(all_baseline_names())
        elif n == "shallow":
            resolved.extend(shallow_baseline_names())
        elif n == "deep":
            resolved.extend(deep_baseline_names())
        else:
            resolved.append(n)
    # Deduplicate while preserving order.
    seen = set()
    out: list[str] = []
    for n in resolved:
        if n not in seen:
            out.append(n)
            seen.add(n)
    return out


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=str, default="indian_pines")
    parser.add_argument(
        "--baselines",
        type=str,
        nargs="+",
        default=["all"],
        help=(
            "Baseline names; pass 'all' / 'shallow' / 'deep' for groups, "
            "or list individual names: " + ", ".join(all_baseline_names())
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--samples-per-class", type=int, default=5)
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="override the deep baselines' epoch count (per-config default otherwise)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="override the deep baselines' batch size",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="DataLoader workers for the deep baselines",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=None,
        help="root output dir (defaults to outputs/)",
    )
    return parser.parse_args(argv)


def _run_shallow(name: str, baseline_cfg: dict[str, Any], artifacts, seed: int) -> dict[str, Any]:
    sk_params = dict(baseline_cfg.get("sklearn") or {})
    model = make_shallow_model(name, config=sk_params, seed=seed)
    t0 = time.time()
    model.fit(artifacts.train_dataset)
    fit_seconds = time.time() - t0
    preds, probs = model.predict(artifacts.test_dataset)
    labels = artifacts.test_dataset.labels
    metrics = compute_metrics(
        labels.astype(np.int64),
        preds.astype(np.int64),
        num_classes=artifacts.num_classes,
        scene_freq=artifacts.scene_freq.detach().cpu().numpy(),
    )
    ece, _ = expected_calibration_error(probs, labels.astype(np.int64), num_bins=15)
    metrics["ECE_15"] = float(ece)
    return {
        "name": name,
        "family": "shallow",
        "fit_seconds": float(fit_seconds),
        "metrics": metrics,
    }


def _run_deep(
    name: str,
    baseline_cfg: dict[str, Any],
    artifacts,
    args: argparse.Namespace,
    output_dir: Path,
) -> dict[str, Any]:
    device = torch.device(args.device)
    model_cfg = dict(baseline_cfg.get("model") or {})
    training_cfg = dict(baseline_cfg.get("training") or {})
    if args.epochs is not None:
        training_cfg["epochs"] = int(args.epochs)
    if args.batch_size is not None:
        training_cfg["batch_size"] = int(args.batch_size)
    if args.num_workers is not None:
        training_cfg["num_workers"] = int(args.num_workers)
        training_cfg["pin_memory"] = bool(args.num_workers) and training_cfg.get(
            "pin_memory", False
        )

    model = make_deep_baseline(
        name,
        model_cfg,
        num_bands=artifacts.num_bands,
        num_pca=artifacts.pca_components,
        patch_size=artifacts.patch_size,
        num_classes=artifacts.num_classes,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(
        f"[{name}] params={n_params:,} ({n_params / 1e6:.3f} M)  device={device}",
        flush=True,
    )

    cfg = BaselineTrainConfig.from_yaml_dict(training_cfg)
    trainer = BaselineTrainer(
        model=model,
        scene_freq=artifacts.scene_freq,
        train_dataset=artifacts.train_dataset,
        val_dataset=artifacts.test_dataset,
        config=cfg,
        output_dir=output_dir,
        device=device,
        seed=args.seed,
        on_log=lambda row: None,  # silent per-epoch by default
    )
    t0 = time.time()
    state = trainer.fit()
    fit_seconds = time.time() - t0
    final_metrics = trainer.evaluate()
    return {
        "name": name,
        "family": "deep",
        "num_params": int(n_params),
        "fit_seconds": float(fit_seconds),
        "best_oa": float(state.best_oa),
        "metrics": final_metrics,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    seed_everything(args.seed)
    baselines = _resolve_baselines(args.baselines)
    print(f"[run] baselines = {baselines}", flush=True)

    # Build the shared data split ONCE -- every baseline sees identical pixels.
    ds_cfg = load_yaml(CONFIG_ROOT / "datasets" / f"{args.dataset}.yaml")
    ds_cfg["data_dir"] = str(ROOT / ds_cfg["data_dir"])
    ds_cfg["samples_per_class"] = int(args.samples_per_class)
    artifacts = build_split(ds_cfg, seed=args.seed)
    print(
        f"[data] {ds_cfg['name']}: train={len(artifacts.train_dataset)} "
        f"test={len(artifacts.test_dataset)} K={artifacts.num_classes} "
        f"B={artifacts.num_bands} P={artifacts.patch_size} "
        f"PCA={artifacts.pca_components}",
        flush=True,
    )

    output_root = Path(args.output_root) if args.output_root else ROOT / "outputs"
    summary_rows: list[dict[str, Any]] = []

    for name in baselines:
        baseline_cfg = load_yaml(CONFIG_ROOT / "baselines" / f"{name}.yaml")
        out_dir = output_root / f"{args.dataset}_{name}_seed{args.seed}"
        out_dir.mkdir(parents=True, exist_ok=True)

        if is_shallow_baseline(name):
            row = _run_shallow(name, baseline_cfg, artifacts, args.seed)
        elif is_deep_baseline(name):
            row = _run_deep(name, baseline_cfg, artifacts, args, out_dir)
        else:
            raise ValueError(f"unknown baseline '{name}'")

        (out_dir / "final.json").write_text(json.dumps(row, indent=2))
        summary_rows.append(row)
        m = row["metrics"]
        print(
            f"[done {name:14s}] OA={m['OA']*100:.2f} AA={m['AA']*100:.2f} "
            f"kappa={m['kappa']:.3f} ECE={m.get('ECE_15', float('nan')):.3f}",
            flush=True,
        )

    summary = {
        "dataset": ds_cfg["name"],
        "seed": args.seed,
        "samples_per_class": int(args.samples_per_class),
        "rows": summary_rows,
    }
    summary_path = (
        output_root
        / f"baselines_summary_{ds_cfg['name']}_seed{args.seed}_spc{args.samples_per_class}.json"
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"[summary] wrote {summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
