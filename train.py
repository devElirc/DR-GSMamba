from __future__ import annotations

import argparse
from pathlib import Path
import warnings

import numpy as np
import torch
from tqdm import tqdm

from datasets import create_dataloaders
from losses import RobustRiskTracker, dr_gsmamba_loss
from models import DRGSMamba
from utils.common import get_device, load_config, save_json, set_seed
from utils.metrics import classification_metrics


def run_epoch(model, loader, optimizer, device, cfg, train: bool, tracker: RobustRiskTracker | None = None):
    model.train(train)
    all_true, all_pred = [], []
    losses = []
    for batch in tqdm(loader, leave=False):
        patch = batch["patch"].to(device)
        spectrum = batch["spectrum"].to(device)
        labels = batch["label"].to(device)
        with torch.set_grad_enabled(train):
            outputs = model(patch, spectrum)
            loss, _ = dr_gsmamba_loss(outputs, labels, cfg, tracker=tracker if train else None)
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
        losses.append(float(loss.detach().cpu()))
        all_true.extend(labels.detach().cpu().numpy().tolist())
        all_pred.extend(outputs["logits"].argmax(dim=-1).detach().cpu().numpy().tolist())
    return np.mean(losses), np.asarray(all_true), np.asarray(all_pred)


@torch.no_grad()
def evaluate(model, loader, device, num_classes):
    model.eval()
    all_true, all_pred, all_prob, all_unc = [], [], [], []
    for batch in loader:
        patch = batch["patch"].to(device)
        spectrum = batch["spectrum"].to(device)
        labels = batch["label"].to(device)
        outputs = model(patch, spectrum)
        probs = torch.softmax(outputs["logits"], dim=-1)
        all_true.extend(labels.cpu().numpy().tolist())
        all_pred.extend(outputs["logits"].argmax(dim=-1).cpu().numpy().tolist())
        all_prob.extend(probs.cpu().numpy().tolist())
        all_unc.extend(outputs["uncertainty"].cpu().numpy().tolist())
    metrics = classification_metrics(all_true, all_pred, num_classes, y_prob=all_prob)
    metrics["mean_uncertainty"] = float(np.mean(all_unc))
    return metrics


def train_one_seed(cfg: dict, seed: int) -> dict:
    set_seed(seed)
    device = get_device(cfg["training"]["device"])
    loaders, num_classes = create_dataloaders(cfg, seed)
    spectral_dim = int(cfg.get("runtime", {}).get("spectral_dim", cfg["dataset"]["pca_components"]))
    model = DRGSMamba(
        spectral_dim=spectral_dim,
        num_classes=num_classes,
        hidden_dim=cfg["model"]["hidden_dim"],
        depth=cfg["model"]["depth"],
        use_spectral=cfg["model"].get("use_spectral", True),
        use_graph=cfg["model"].get("use_graph", True),
        use_prototype=cfg["model"].get("use_prototype", True),
        spectral_backend=cfg["model"].get("spectral_backend", "ssm"),
        spectral_heads=cfg["model"].get("spectral_heads", 4),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"]["weight_decay"],
    )
    best_val, best_state = -1.0, None
    history = []
    tracker = RobustRiskTracker(
        num_classes=num_classes,
        momentum=float(cfg.get("loss", {}).get("risk_ema_momentum", 0.9)),
        device=device,
    )
    for epoch in range(1, cfg["training"]["epochs"] + 1):
        loss, y_true, y_pred = run_epoch(model, loaders["train"], optimizer, device, cfg, train=True, tracker=tracker)
        train_metrics = classification_metrics(y_true, y_pred, num_classes)
        val_metrics = evaluate(model, loaders["val"], device, num_classes)
        history.append({"epoch": epoch, "loss": loss, "train_oa": train_metrics["oa"], "val_oa": val_metrics["oa"]})
        if val_metrics["oa"] > best_val:
            best_val = val_metrics["oa"]
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        print(f"seed={seed} epoch={epoch} loss={loss:.4f} train_oa={train_metrics['oa']:.4f} val_oa={val_metrics['oa']:.4f}")
    model.load_state_dict(best_state)
    test_metrics = evaluate(model, loaders["test"], device, num_classes)
    out_dir = Path(cfg["outputs"]["dir"]) / f"seed_{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    if cfg.get("outputs", {}).get("save_checkpoints", True):
        try:
            torch.save(model.state_dict(), out_dir / "best_model.pt")
        except OSError as exc:
            warnings.warn(f"Could not save checkpoint to {out_dir / 'best_model.pt'}: {exc}", RuntimeWarning)
        except RuntimeError as exc:
            warnings.warn(f"Could not save checkpoint to {out_dir / 'best_model.pt'}: {exc}", RuntimeWarning)
    save_json({"seed": seed, "history": history, "test": test_metrics}, out_dir / "metrics.json")
    return test_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/synthetic.yaml")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    cfg = load_config(args.config)
    metrics = train_one_seed(cfg, args.seed)
    print(metrics)


if __name__ == "__main__":
    main()
