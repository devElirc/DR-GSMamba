from __future__ import annotations

import argparse
import copy
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets import create_dataloaders
from models.baselines import build_baseline
from run_experiments import summarize
from utils.common import deep_update, get_device, load_config, save_json, set_seed
from utils.metrics import classification_metrics


def train_epoch(model, loader, optimizer, device):
    model.train()
    losses = []
    for batch in tqdm(loader, leave=False):
        patch = batch["patch"].to(device)
        spectrum = batch["spectrum"].to(device)
        labels = batch["label"].to(device)
        logits = model(patch, spectrum)
        loss = F.cross_entropy(logits, labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses))


@torch.no_grad()
def evaluate(model, loader, device, num_classes):
    model.eval()
    all_true, all_pred, all_prob = [], [], []
    for batch in loader:
        patch = batch["patch"].to(device)
        spectrum = batch["spectrum"].to(device)
        labels = batch["label"].to(device)
        logits = model(patch, spectrum)
        probs = torch.softmax(logits, dim=-1)
        all_true.extend(labels.cpu().numpy().tolist())
        all_pred.extend(logits.argmax(dim=-1).cpu().numpy().tolist())
        all_prob.extend(probs.cpu().numpy().tolist())
    return classification_metrics(all_true, all_pred, num_classes, y_prob=all_prob)


def run_baseline_seed(cfg: dict, model_name: str, seed: int) -> dict:
    set_seed(seed)
    device = get_device(cfg["training"]["device"])
    loaders, num_classes = create_dataloaders(cfg, seed)
    spectral_dim = int(cfg.get("runtime", {}).get("spectral_dim", cfg["dataset"]["pca_components"]))
    model = build_baseline(model_name, spectral_dim, num_classes, cfg["model"]["hidden_dim"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["training"]["lr"], weight_decay=cfg["training"]["weight_decay"])
    best_val, best_state = -1.0, None
    history = []
    for epoch in range(1, cfg["training"]["epochs"] + 1):
        loss = train_epoch(model, loaders["train"], optimizer, device)
        val_metrics = evaluate(model, loaders["val"], device, num_classes)
        history.append({"epoch": epoch, "loss": loss, "val_oa": val_metrics["oa"]})
        if val_metrics["oa"] > best_val:
            best_val = val_metrics["oa"]
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        print(f"baseline={model_name} seed={seed} epoch={epoch} loss={loss:.4f} val_oa={val_metrics['oa']:.4f}")
    model.load_state_dict(best_state)
    test_metrics = evaluate(model, loaders["test"], device, num_classes)
    return {"history": history, "test": test_metrics}


def main():
    parser = argparse.ArgumentParser(description="Run neural HSI baselines under the same splits as DR-GSMamba.")
    parser.add_argument("--config", default="configs/synthetic.yaml")
    parser.add_argument(
        "--models",
        nargs="+",
        default=[
            "spectral_mlp",
            "cnn2d",
            "cnn3d",
            "hybridsn_lite",
            "spectralformer_lite",
            "ssftt_lite",
            "nonlocal_gcn_lite",
            "mamba_lite",
        ],
    )
    parser.add_argument("--seeds", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--samples-per-class", type=int, default=None)
    parser.add_argument("--out-dir", default="outputs/deep_baselines")
    args = parser.parse_args()

    base_cfg = copy.deepcopy(load_config(args.config))
    if args.seeds is not None:
        base_cfg["training"]["seeds"] = list(range(args.seeds))
    if args.epochs is not None:
        base_cfg["training"]["epochs"] = args.epochs
    if args.samples_per_class is not None:
        deep_update(base_cfg, {"dataset": {"samples_per_class": args.samples_per_class}})

    out_root = Path(args.out_dir)
    all_reports = {}
    for model_name in args.models:
        runs = []
        for seed in base_cfg["training"]["seeds"]:
            result = run_baseline_seed(copy.deepcopy(base_cfg), model_name, seed)
            runs.append(result["test"])
            seed_dir = out_root / model_name / f"seed_{seed}"
            save_json({"seed": seed, "history": result["history"], "test": result["test"]}, seed_dir / "metrics.json")
        report = {"model": model_name, "seeds": base_cfg["training"]["seeds"], "summary": summarize(runs), "runs": runs}
        all_reports[model_name] = report
        save_json(report, out_root / model_name / "summary.json")
    save_json({"config": args.config, "baselines": all_reports}, out_root / "summary.json")
    print(f"Saved deep baseline reports to {out_root}")


if __name__ == "__main__":
    main()
