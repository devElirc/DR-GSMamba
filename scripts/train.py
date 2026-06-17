"""Train a single ``DRGSMamba`` model on one dataset / one seed / one label setting.

Usage::

    python scripts/train.py --dataset indian_pines --training default --seed 0
    python scripts/train.py --dataset indian_pines --training label_5 --seed 0 --epochs 100

Composes the resolved YAML configs into a single dict, then drives the
:class:`hsi_robust.training.Trainer` end-to-end.

This is the M2E.7 implementation and the entry point exercised by the M2E.8
smoke run (see ``scripts/_phase2e_smoke.py``).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

from hsi_robust.data import build_split
from hsi_robust.models import DRGSMamba
from hsi_robust.training import TrainConfig, Trainer
from hsi_robust.utils import load_yaml, seed_everything

ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = ROOT / "configs"


def _resolve_configs(args: argparse.Namespace) -> tuple[dict, dict, dict]:
    defaults = load_yaml(CONFIG_ROOT / "defaults.yaml")
    ds_name = args.dataset or defaults["defaults"]["dataset"]
    model_name = args.model or defaults["defaults"]["model"]
    train_name = args.training or defaults["defaults"]["training"]

    ds_cfg = load_yaml(CONFIG_ROOT / "datasets" / f"{ds_name}.yaml")
    model_cfg = load_yaml(CONFIG_ROOT / "model" / f"{model_name}.yaml")
    # Compose the training config: start from default.yaml and overlay any
    # label-scarcity override (which only specifies samples_per_class).
    train_cfg_default = load_yaml(CONFIG_ROOT / "training" / "default.yaml")
    train_cfg = dict(train_cfg_default)
    if train_name != "default":
        overlay = load_yaml(CONFIG_ROOT / "training" / f"{train_name}.yaml")
        train_cfg.update(overlay)

    # Bring `samples_per_class` from training override into dataset config.
    if "samples_per_class" in train_cfg:
        ds_cfg["samples_per_class"] = int(train_cfg["samples_per_class"])

    # CLI overrides.
    if args.epochs is not None:
        train_cfg["epochs"] = int(args.epochs)
    if args.batch_size is not None:
        train_cfg["batch_size"] = int(args.batch_size)
    if args.alpha is not None:
        train_cfg.setdefault("loss", {}).setdefault("cfa_gdro", {})["alpha"] = float(args.alpha)
    if args.gamma is not None:
        train_cfg.setdefault("loss", {}).setdefault("cfa_gdro", {})["gamma"] = float(args.gamma)
    if args.lambda_rob is not None:
        train_cfg.setdefault("loss", {}).setdefault("cfa_gdro", {})["weight"] = float(
            args.lambda_rob
        )
    if args.samples_per_class is not None:
        ds_cfg["samples_per_class"] = int(args.samples_per_class)
    if args.val_every is not None:
        train_cfg["val_every"] = int(args.val_every)
    if args.save_every is not None:
        train_cfg["save_every"] = int(args.save_every)
    if args.num_workers is not None:
        train_cfg["num_workers"] = int(args.num_workers)
        train_cfg["pin_memory"] = bool(args.num_workers) and train_cfg.get("pin_memory", False)
    return ds_cfg, model_cfg, train_cfg


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--training", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--samples-per-class", type=int, default=None)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--gamma", type=float, default=None)
    parser.add_argument("--lambda-rob", type=float, default=None)
    parser.add_argument(
        "--val-every",
        type=int,
        default=None,
        help="validate every N epochs (overrides training YAML)",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=None,
        help="save a checkpoint every N epochs (overrides training YAML)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="DataLoader workers (overrides training YAML; set to 0 for determinism)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="output directory; defaults to outputs/<dataset>_<training>_seed<seed>/",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
    )
    return parser.parse_args(argv)


def _run_id(args: argparse.Namespace, ds_cfg: dict, train_name: str) -> str:
    bits = [ds_cfg["name"], train_name, f"seed{args.seed}"]
    return "_".join(bits)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    seed_everything(args.seed)

    ds_cfg, model_cfg, train_cfg = _resolve_configs(args)
    train_name = args.training or "default"

    # Resolve data paths relative to the repo root.
    ds_cfg["data_dir"] = str(ROOT / ds_cfg["data_dir"])

    # Build deterministic split.
    artifacts = build_split(ds_cfg, seed=args.seed)
    print(
        f"[data] {ds_cfg['name']}: train={len(artifacts.train_dataset)} "
        f"test={len(artifacts.test_dataset)} "
        f"K={artifacts.num_classes} B={artifacts.num_bands} "
        f"P={artifacts.patch_size} PCA={artifacts.pca_components}",
        flush=True,
    )

    # Build model.
    device = torch.device(args.device)
    model = DRGSMamba.from_config(
        model_cfg,
        num_bands=artifacts.num_bands,
        num_pca=artifacts.pca_components,
        patch_size=artifacts.patch_size,
        num_classes=artifacts.num_classes,
    )
    n_params = model.num_parameters()
    print(f"[model] params={n_params:,} ({n_params / 1e6:.3f} M)", flush=True)

    # Output dir.
    run_id = args.output_dir or _run_id(args, ds_cfg, train_name)
    output_dir = ROOT / "outputs" / run_id
    print(f"[run]  output_dir={output_dir}", flush=True)

    # Trainer.
    cfg = TrainConfig.from_yaml_dict(train_cfg)

    def _log(row: dict) -> None:
        oa = row.get("val/OA")
        oa_str = f"OA={oa:.4f}" if isinstance(oa, float) else "OA=  n/a"
        lr = row.get("lr", float("nan"))
        loss = row.get("train_loss", float("nan"))
        print(
            f"[epoch {row['epoch']:03d}] lr={lr:.2e} train_loss={loss:.4f} {oa_str}",
            flush=True,
        )

    trainer = Trainer(
        model=model,
        scene_freq=artifacts.scene_freq,
        train_dataset=artifacts.train_dataset,
        val_dataset=artifacts.test_dataset,
        config=cfg,
        output_dir=output_dir,
        device=device,
        on_log=_log,
        seed=args.seed,
    )
    state = trainer.fit()

    # Final summary.
    final_metrics = trainer.evaluate()
    final = {
        "best_oa": state.best_oa,
        "final_metrics": final_metrics,
        "num_train": len(artifacts.train_dataset),
        "num_test": len(artifacts.test_dataset),
        "num_classes": artifacts.num_classes,
        "num_params": n_params,
    }
    (output_dir / "final.json").write_text(json.dumps(final, indent=2))
    print(json.dumps(final, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
