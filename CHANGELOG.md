# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

Nothing yet. Phase 2B (data module) is the next milestone — see `roadmap.md` §3.

---

## [0.0.1] — 2026-05-23 — Phase 2A scaffold

### Added

- **Phase 0 — Story lock.** Working title locked to *Class-Frequency-Aware Group-DRO for Reliable Label-Scarce Hyperspectral Image Classification*. Decisions D-01 through D-04 recorded in `roadmap.md` §6.
- **Phase 1 — Math + method design.**
  - `docs/math/cfa_gdro.md`: CFA-GDRO objective, KKT analysis, sorted water-filling solver, EMA stabiliser, Proposition 1 (Per-class dominance) with Corollary A (rare-class) and Corollary B (worst-class), and an auxiliary chi-squared-ball remark.
  - `docs/math/evidential_prototype_head.md`: evidence from cosine-prototype similarity, Dirichlet posterior, closed-form vacuity and aleatoric uncertainty, Bayes-risk plus annealed KL loss.
  - `docs/math/cp_graph.md`: in-batch k-NN graph, softmax-weighted cosine edges, one-directional KL consistency with stop-gradient target.
  - `paper/abstract_v1.md`: 250-word abstract, five contributions, explicit non-claims list.
  - `paper/sections/method.tex`: LaTeX skeleton with every locked equation, water-filling algorithm, and Prop + Corollaries block.
- **Phase 2A — Scaffolding.**
  - `pyproject.toml` (PEP 621, src-layout, ruff + black + mypy + pytest configuration).
  - `src/hsi_robust/{__init__,_version,data,models,losses,training,eval,utils}/__init__.py` package skeleton.
  - `configs/{defaults.yaml, datasets/, model/, training/}` layout with placeholders for Indian Pines, Pavia University, Salinas, Houston 2013, the full DR-GSMamba model, and the four label-scarcity regimes (3 / 5 / 10 / 20 samples per class).
  - `requirements.txt` and `requirements-dev.txt`.
  - `tests/conftest.py` with auto-used deterministic seeding and tiny-tensor / scene-frequency fixtures; `tests/test_smoke.py` proves the package and every subpackage import.
  - `scripts/{train.py, run_multi_seed.py, run_baselines.py, run_ablations.py, run_robustness.py, make_main_tables.py, make_figures.py}` as importable stubs that exit 0 with a phase pointer.
  - `.pre-commit-config.yaml` (trailing whitespace, end-of-file, large-file guard, ruff, black).
  - `LICENSE` (MIT) and this `CHANGELOG.md`.
