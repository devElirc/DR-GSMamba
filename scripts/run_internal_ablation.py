"""Internal loss ablation -- same backbone x {CE, focal, sample-CVaR,
Sagawa-Group-DRO, CFA-GDRO} x N seeds.

Directly implements Kass Kass's 2026-06-18 feedback (decision D-14): before
spending GPU on external baselines (HybridSN, SpectralFormer, SSFTT,
MambaHSI), prove that CFA-GDRO actually beats the simpler robust / imbalanced
objectives on the *same backbone + same data + same seed grid + same epochs*.

What gets compared (per Kass's checklist)::

    same backbone + CE
    same backbone + focal loss
    same backbone + sample-level CVaR
    same backbone + standard Sagawa Group-DRO
    same backbone + CFA-GDRO

Output structure::

    outputs/internal_ablation/<run-id>/
        manifest.json          -- full reproduction metadata
        <loss>/seed<k>/final.json   -- per-run training output
        summary.json           -- aggregated table (mean ± std over seeds)
        summary.md             -- human-readable Markdown table

Usage::

    python scripts/run_internal_ablation.py
    python scripts/run_internal_ablation.py --dataset indian_pines --training label_5 \
        --seeds 0 1 2 --epochs 200

Decision rule (read this before signing off on Phase 6):

    Proceed to external baselines (Phase 3 / Phase 6) **only if** CFA-GDRO,
    averaged over the seed grid, beats:

        * sagawa_group_dro on `worst_class`, AND
        * sample_cvar      on `rare_class`, AND
        * ce               on `OA` within 2 percentage points.

    Otherwise: tune alpha / gamma / lambda_evi first, or rework the title
    per the contingency in roadmap §7.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.train import main as train_main  # noqa: E402

LOSSES: tuple[str, ...] = ("ce", "focal", "sample_cvar", "sagawa_group_dro", "cfa_gdro")

# Metrics we aggregate. The first three are the "Reliable" claim; the next two
# are the calibration claim; the last is the wall-time sanity check.
HEADLINE_METRICS = (
    "OA",
    "AA",
    "kappa",
    "worst_class",
    "rare_class",
    "ECE_15",
    "ECE_15_T",
    "temperature",
)


def _output_root(run_id: str) -> Path:
    return ROOT / "outputs" / "internal_ablation" / run_id


def _read_final(run_dir: Path) -> dict[str, Any]:
    return json.loads((run_dir / "final.json").read_text())


def _train_one(
    *,
    loss_name: str,
    seed: int,
    dataset: str,
    training: str,
    epochs: int,
    output_dir: str,
    val_every: int,
    save_every: int,
    extra_args: list[str],
) -> int:
    argv = [
        "--dataset",
        dataset,
        "--training",
        training,
        "--loss-name",
        loss_name,
        "--seed",
        str(seed),
        "--epochs",
        str(epochs),
        "--val-every",
        str(val_every),
        "--save-every",
        str(save_every),
        "--num-workers",
        "0",
        "--output-dir",
        output_dir,
        *extra_args,
    ]
    return train_main(argv)


def _aggregate(per_seed: list[dict[str, Any]]) -> dict[str, Any]:
    """Return ``{metric: {mean, std, n, values}}`` over a list of per-seed runs."""
    out: dict[str, Any] = {}
    for key in HEADLINE_METRICS:
        vals = [
            float(r["final_metrics"][key])
            for r in per_seed
            if isinstance(r.get("final_metrics", {}).get(key), (int, float))
        ]
        if not vals:
            out[key] = {"mean": None, "std": None, "n": 0, "values": []}
            continue
        mean = statistics.fmean(vals)
        std = statistics.pstdev(vals) if len(vals) > 1 else 0.0
        out[key] = {"mean": mean, "std": std, "n": len(vals), "values": vals}
    return out


def _format_md(summary: dict[str, dict[str, Any]]) -> str:
    """Render the loss-by-metric table as Markdown for the report."""
    cols = ("OA", "AA", "worst_class", "rare_class", "ECE_15", "ECE_15_T")
    header = "| Loss | " + " | ".join(cols) + " |"
    sep = "| --- | " + " | ".join(["---"] * len(cols)) + " |"
    rows: list[str] = []
    for loss in LOSSES:
        agg = summary.get(loss, {})
        cells: list[str] = [loss]
        for c in cols:
            stat = agg.get(c) or {}
            m = stat.get("mean")
            s = stat.get("std")
            if m is None:
                cells.append("n/a")
            else:
                cells.append(f"{m:.4f} ± {s:.4f}")
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, sep, *rows])


def _verdict(summary: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Apply the D-14 decision rule and return a structured verdict."""

    def _mean(loss: str, metric: str) -> float | None:
        stat = summary.get(loss, {}).get(metric) or {}
        m = stat.get("mean")
        return float(m) if isinstance(m, (int, float)) else None

    cfa_worst = _mean("cfa_gdro", "worst_class")
    gdro_worst = _mean("sagawa_group_dro", "worst_class")
    cfa_rare = _mean("cfa_gdro", "rare_class")
    cvar_rare = _mean("sample_cvar", "rare_class")
    cfa_oa = _mean("cfa_gdro", "OA")
    ce_oa = _mean("ce", "OA")

    def _gt(a: float | None, b: float | None) -> bool:
        return a is not None and b is not None and a > b

    def _within(a: float | None, b: float | None, tol: float) -> bool:
        # tiny float epsilon so a "2 pp" gap of exactly 0.02 (which often
        # arrives as 0.0200000000...18 from float arithmetic) is accepted.
        return a is not None and b is not None and abs(a - b) <= tol + 1e-9

    worst_check = _gt(cfa_worst, gdro_worst)
    rare_check = _gt(cfa_rare, cvar_rare)
    oa_check = _within(cfa_oa, ce_oa, tol=0.02)
    proceed = bool(worst_check and rare_check and oa_check)
    return {
        "cfa_beats_gdro_on_worst_class": worst_check,
        "cfa_beats_cvar_on_rare_class": rare_check,
        "cfa_within_2pct_of_ce_on_OA": oa_check,
        "proceed_to_phase_6_baselines": proceed,
        "values": {
            "cfa_worst": cfa_worst,
            "gdro_worst": gdro_worst,
            "cfa_rare": cfa_rare,
            "cvar_rare": cvar_rare,
            "cfa_oa": cfa_oa,
            "ce_oa": ce_oa,
        },
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=str, default="indian_pines")
    parser.add_argument("--training", type=str, default="label_5")
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[0, 1, 2],
        help="Seed grid (default: 0 1 2 -- the minimum for ± std error bars)",
    )
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--val-every", type=int, default=20)
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Output sub-directory name; default is timestamped",
    )
    parser.add_argument(
        "--losses",
        type=str,
        nargs="+",
        default=list(LOSSES),
        choices=list(LOSSES),
        help="Subset of losses to run (default: all 5)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run_id = args.run_id or time.strftime("ablation_%Y%m%d_%H%M%S")
    root = _output_root(run_id)
    root.mkdir(parents=True, exist_ok=True)

    manifest = {
        "feedback_reference": "feedback.txt 2026-06-18 (Kass Kass)",
        "decision": "D-14",
        "dataset": args.dataset,
        "training": args.training,
        "seeds": args.seeds,
        "epochs": args.epochs,
        "losses": args.losses,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2))

    summary: dict[str, dict[str, Any]] = {}
    n_done = 0
    n_total = len(args.losses) * len(args.seeds)

    for loss_name in args.losses:
        loss_dir = root / loss_name
        loss_dir.mkdir(exist_ok=True)
        per_seed: list[dict[str, Any]] = []
        for seed in args.seeds:
            n_done += 1
            run_dir = loss_dir / f"seed{seed}"
            rel_output = str(
                (Path("internal_ablation") / run_id / loss_name / f"seed{seed}").as_posix()
            )
            print(
                f"[{n_done:>3}/{n_total}] loss={loss_name:<18} seed={seed} "
                f"-> outputs/{rel_output}",
                flush=True,
            )
            rc = _train_one(
                loss_name=loss_name,
                seed=seed,
                dataset=args.dataset,
                training=args.training,
                epochs=args.epochs,
                output_dir=rel_output,
                val_every=args.val_every,
                save_every=args.epochs,
                extra_args=[],
            )
            if rc != 0:
                print(f"[ablation] run rc={rc} -- skipping aggregation", file=sys.stderr)
                continue
            per_seed.append(_read_final(run_dir))
        summary[loss_name] = _aggregate(per_seed)

    out = {
        "manifest": manifest,
        "summary": summary,
        "verdict": _verdict(summary),
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    (root / "summary.json").write_text(json.dumps(out, indent=2))
    (root / "summary.md").write_text(_format_md(summary) + "\n")

    print()
    print(_format_md(summary))
    print()
    print(json.dumps(out["verdict"], indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
