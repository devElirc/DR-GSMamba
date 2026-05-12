from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models import DRGSMamba
from utils.common import get_device, load_config, save_json


def count_parameters(model: torch.nn.Module) -> dict:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return {"total": int(total), "trainable": int(trainable)}


def profile_latency(model: torch.nn.Module, patch: torch.Tensor, spectrum: torch.Tensor, warmup: int, runs: int) -> dict:
    model.eval()
    device = patch.device
    with torch.no_grad():
        for _ in range(warmup):
            model(patch, spectrum)
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(runs):
            model(patch, spectrum)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
    return {
        "runs": runs,
        "batch_latency_ms": float(elapsed * 1000.0 / runs),
        "sample_latency_ms": float(elapsed * 1000.0 / (runs * patch.shape[0])),
    }


def main():
    parser = argparse.ArgumentParser(description="Profile DR-GSMamba parameter count and inference latency.")
    parser.add_argument("--config", default="configs/synthetic.yaml")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--runs", type=int, default=50)
    parser.add_argument("--out", default="outputs/profile/model_profile.json")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = get_device(cfg["training"]["device"])
    spectral_dim = int(cfg["dataset"]["pca_components"])
    patch_size = int(cfg["dataset"]["patch_size"])
    num_classes = int(cfg["dataset"].get("synthetic", {}).get("classes", 16))
    model = DRGSMamba(
        spectral_dim=spectral_dim,
        num_classes=num_classes,
        hidden_dim=cfg["model"]["hidden_dim"],
        depth=cfg["model"]["depth"],
        use_spectral=cfg["model"].get("use_spectral", True),
        use_graph=cfg["model"].get("use_graph", True),
        use_prototype=cfg["model"].get("use_prototype", True),
    ).to(device)
    patch = torch.randn(args.batch_size, spectral_dim, patch_size, patch_size, device=device)
    spectrum = torch.randn(args.batch_size, spectral_dim, device=device)
    report = {
        "config": args.config,
        "device": str(device),
        "input": {"batch_size": args.batch_size, "spectral_dim": spectral_dim, "patch_size": patch_size},
        "parameters": count_parameters(model),
        "latency": profile_latency(model, patch, spectrum, args.warmup, args.runs),
        "note": "Compare this report with the same profiling protocol for Transformer and Mamba baselines before claiming efficiency.",
    }
    save_json(report, args.out)
    print(report)


if __name__ == "__main__":
    main()
