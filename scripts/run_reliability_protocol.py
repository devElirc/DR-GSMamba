from __future__ import annotations

import argparse
import copy
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from run_experiments import run_config
from utils.common import deep_update, load_config, save_json


LOW_LABEL_SETTINGS = [3, 5, 10, 20]

ABLATIONS = {
    "full": {},
    "without_dro": {"loss": {"robust_weight": 0.0}},
    "without_spectral_ssm": {"model": {"use_spectral": False}},
    "without_graph": {"model": {"use_graph": False}, "loss": {"graph_smooth_weight": 0.0}},
    "without_prototype": {
        "model": {"use_prototype": False},
        "loss": {"prototype_weight": 0.0, "prototype_supervised_weight": 0.0},
    },
    "without_uncertainty": {"loss": {"uncertainty_weight": 0.0}},
}


def run_case(base_cfg: dict, output_root: Path, case_name: str, updates: dict) -> dict:
    cfg = copy.deepcopy(base_cfg)
    deep_update(cfg, updates)
    cfg["outputs"]["dir"] = str(output_root / case_name)
    summary = run_config(cfg)
    return {"case": case_name, "updates": updates, "summary": summary}


def dro_delta(full_summary: dict, ablated_summary: dict) -> dict:
    metrics = ["macro_f1", "worst_class_accuracy", "rare_class_accuracy", "oa"]
    return {
        metric: float(full_summary[metric]["mean"] - ablated_summary[metric]["mean"])
        for metric in metrics
        if metric in full_summary and metric in ablated_summary
    }


def main():
    parser = argparse.ArgumentParser(description="Run the DR-GSMamba reliability protocol.")
    parser.add_argument("--config", default="configs/synthetic.yaml")
    parser.add_argument("--output-root", default="outputs/protocol")
    parser.add_argument("--seeds", type=int, default=None, help="Override seed count for every case.")
    parser.add_argument("--skip-low-label", action="store_true")
    parser.add_argument("--skip-ablations", action="store_true")
    args = parser.parse_args()

    base_cfg = copy.deepcopy(load_config(args.config))
    if args.seeds is not None:
        base_cfg["training"]["seeds"] = list(range(args.seeds))

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    report: dict = {"config": args.config, "low_label": [], "ablations": [], "dro_validation": {}}

    if not args.skip_low_label:
        for samples in LOW_LABEL_SETTINGS:
            case = run_case(
                base_cfg,
                output_root / "low_label",
                f"samples_per_class_{samples}",
                {"dataset": {"samples_per_class": samples}},
            )
            report["low_label"].append(case)

    if not args.skip_ablations:
        full_summary = None
        without_dro_summary = None
        for name, updates in ABLATIONS.items():
            case = run_case(base_cfg, output_root / "ablations", name, updates)
            report["ablations"].append(case)
            if name == "full":
                full_summary = case["summary"]
            elif name == "without_dro":
                without_dro_summary = case["summary"]

        if full_summary and without_dro_summary:
            report["dro_validation"] = {
                "full_minus_without_dro": dro_delta(full_summary, without_dro_summary),
                "interpretation": (
                    "The DRO claim is supported only if these deltas are positive for "
                    "rare-class, worst-class, and stability-oriented metrics across real datasets."
                ),
            }

    save_json(report, output_root / "protocol_summary.json")
    print(f"Saved reliability protocol summary to {output_root / 'protocol_summary.json'}")


if __name__ == "__main__":
    main()
