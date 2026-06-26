"""Unit tests for the internal-ablation runner (D-14).

We do not run the full training loop here -- that is covered by
``test_training.py`` and the end-to-end smoke. We test the *aggregation* and
the *decision rule* directly, because those are what the paper's claim
"CFA-GDRO beats the simpler robust losses" rests on.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load_runner():
    """Import scripts/run_internal_ablation.py without invoking ``main``."""
    spec = importlib.util.spec_from_file_location(
        "run_internal_ablation", ROOT / "scripts" / "run_internal_ablation.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def runner():
    return _load_runner()


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #


def test_aggregate_returns_mean_and_std(runner) -> None:
    per_seed = [
        {"final_metrics": {"OA": 0.5, "AA": 0.6, "worst_class": 0.2, "ECE_15": 0.4}},
        {"final_metrics": {"OA": 0.6, "AA": 0.7, "worst_class": 0.3, "ECE_15": 0.3}},
        {"final_metrics": {"OA": 0.7, "AA": 0.8, "worst_class": 0.4, "ECE_15": 0.2}},
    ]
    agg = runner._aggregate(per_seed)
    assert agg["OA"]["mean"] == pytest.approx(0.6, abs=1e-9)
    assert agg["AA"]["mean"] == pytest.approx(0.7, abs=1e-9)
    assert agg["worst_class"]["mean"] == pytest.approx(0.3, abs=1e-9)
    assert agg["OA"]["n"] == 3
    assert agg["OA"]["std"] > 0


def test_aggregate_handles_missing_metric_gracefully(runner) -> None:
    """A run that crashed mid-way (no ECE_15_T) must not break aggregation."""
    per_seed = [
        {"final_metrics": {"OA": 0.5}},
        {"final_metrics": {"OA": 0.6}},
    ]
    agg = runner._aggregate(per_seed)
    assert agg["OA"]["n"] == 2
    assert agg["ECE_15_T"]["n"] == 0
    assert agg["ECE_15_T"]["mean"] is None


# --------------------------------------------------------------------------- #
# Decision rule (the D-14 contract)
# --------------------------------------------------------------------------- #


def _summary(
    *,
    cfa_worst: float,
    gdro_worst: float,
    cfa_rare: float,
    cvar_rare: float,
    cfa_oa: float,
    ce_oa: float,
) -> dict:
    def _stat(v: float) -> dict:
        return {"mean": v, "std": 0.0, "n": 3, "values": [v, v, v]}

    return {
        "cfa_gdro": {
            "worst_class": _stat(cfa_worst),
            "rare_class": _stat(cfa_rare),
            "OA": _stat(cfa_oa),
        },
        "sagawa_group_dro": {"worst_class": _stat(gdro_worst)},
        "sample_cvar": {"rare_class": _stat(cvar_rare)},
        "ce": {"OA": _stat(ce_oa)},
        "focal": {},
    }


def test_verdict_proceeds_when_all_three_criteria_pass(runner) -> None:
    summary = _summary(
        cfa_worst=0.45,
        gdro_worst=0.40,
        cfa_rare=0.95,
        cvar_rare=0.80,
        cfa_oa=0.78,
        ce_oa=0.80,
    )
    verdict = runner._verdict(summary)
    assert verdict["cfa_beats_gdro_on_worst_class"] is True
    assert verdict["cfa_beats_cvar_on_rare_class"] is True
    assert verdict["cfa_within_2pct_of_ce_on_OA"] is True
    assert verdict["proceed_to_phase_6_baselines"] is True


def test_verdict_blocks_when_worst_class_does_not_improve(runner) -> None:
    """CFA-GDRO ties Sagawa on worst-class -- blocked per D-14."""
    summary = _summary(
        cfa_worst=0.40,
        gdro_worst=0.40,
        cfa_rare=0.95,
        cvar_rare=0.80,
        cfa_oa=0.78,
        ce_oa=0.80,
    )
    verdict = runner._verdict(summary)
    assert verdict["cfa_beats_gdro_on_worst_class"] is False
    assert verdict["proceed_to_phase_6_baselines"] is False


def test_verdict_blocks_when_oa_drops_more_than_2pct(runner) -> None:
    summary = _summary(
        cfa_worst=0.45,
        gdro_worst=0.40,
        cfa_rare=0.95,
        cvar_rare=0.80,
        cfa_oa=0.70,
        ce_oa=0.85,
    )
    verdict = runner._verdict(summary)
    assert verdict["cfa_within_2pct_of_ce_on_OA"] is False
    assert verdict["proceed_to_phase_6_baselines"] is False


def test_verdict_handles_missing_metrics(runner) -> None:
    """If sample_cvar never produced rare_class, we must not crash."""
    summary = _summary(
        cfa_worst=0.45,
        gdro_worst=0.40,
        cfa_rare=0.95,
        cvar_rare=0.80,
        cfa_oa=0.78,
        ce_oa=0.80,
    )
    del summary["sample_cvar"]
    verdict = runner._verdict(summary)
    assert verdict["cfa_beats_cvar_on_rare_class"] is False
    assert verdict["proceed_to_phase_6_baselines"] is False


# --------------------------------------------------------------------------- #
# Markdown formatting
# --------------------------------------------------------------------------- #


def test_format_md_produces_5_loss_rows(runner) -> None:
    summary = _summary(
        cfa_worst=0.45,
        gdro_worst=0.40,
        cfa_rare=0.95,
        cvar_rare=0.80,
        cfa_oa=0.78,
        ce_oa=0.80,
    )
    md = runner._format_md(summary)
    for loss in runner.LOSSES:
        assert f"| {loss} |" in md


# --------------------------------------------------------------------------- #
# Deep-merge of training-YAML overlays
# --------------------------------------------------------------------------- #


def _load_train_script():
    spec = importlib.util.spec_from_file_location("train_script", ROOT / "scripts" / "train.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_deep_merge_preserves_unrelated_keys() -> None:
    """An overlay that only changes ``loss.name`` must keep ``loss.cfa_gdro``."""
    train_script = _load_train_script()
    base = {
        "epochs": 200,
        "loss": {
            "name": "cfa_gdro",
            "cfa_gdro": {"alpha": 0.3, "gamma": 1.0},
            "evidential": {"weight": 1.0},
        },
    }
    overlay = {"loss": {"name": "ce"}}
    merged = train_script._deep_merge(base, overlay)
    assert merged["epochs"] == 200
    assert merged["loss"]["name"] == "ce"
    assert merged["loss"]["cfa_gdro"]["alpha"] == 0.3  # not erased
    assert merged["loss"]["evidential"]["weight"] == 1.0
    # base must not be mutated.
    assert base["loss"]["name"] == "cfa_gdro"


def test_deep_merge_overrides_scalars() -> None:
    train_script = _load_train_script()
    base = {"a": 1, "b": {"c": 2, "d": 3}}
    overlay = {"a": 10, "b": {"c": 20}}
    merged = train_script._deep_merge(base, overlay)
    assert merged == {"a": 10, "b": {"c": 20, "d": 3}}
