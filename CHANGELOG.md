# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Phase 2B — Data module.**
  - `src/hsi_robust/data/io.py`: `.mat` / `.npy` cube + ground-truth loader with shape and label-range sanity checks against the dataset config.
  - `src/hsi_robust/data/scene_freq.py`: `flatten_labeled_pixels` (auto-remap 1-indexed scene gt to 0-indexed) and `compute_scene_freq` (scene-level π_k, the source-of-truth quantity for CFA-GDRO per `docs/math/cfa_gdro.md` §1).
  - `src/hsi_robust/data/sampler.py`: `stratified_fixed_per_class_split` with `np.random.default_rng(seed)` for bit-deterministic train/test draws; emits a `UserWarning` when a class is starved (e.g., class 9 of Indian Pines at `samples_per_class=20`).
  - `src/hsi_robust/data/transforms.py`: `PerBandStandardize`, `PCAReducer` (full SVD solver, fixed random_state), and reflection-padded patch helpers.
  - `src/hsi_robust/data/hsi_dataset.py`: `HSIDataset` returning `(raw_spectrum, pca_patch, label)` tuples, plus `SplitArtifacts` dataclass and `build_split` / `build_split_from_arrays` orchestrators.
  - `src/hsi_robust/utils/config.py`: `load_yaml` helper (no Hydra runtime dependency).
  - `tests/test_data.py`: 18 unit tests covering scene-freq summation, sampler determinism and starvation handling, leak-free fit of standardiser + PCA, dataset shapes / dtypes, and an Indian Pines end-to-end determinism test guarded by data availability.
  - `scripts/_phase2b_exit_check.py`: minimal demo proving the M2B exit criterion (5-line mini-batch from Indian Pines + identical-on-second-invocation).

Phase 2C (model module) is the next milestone — see `roadmap.md` §3.

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
  - `configs/{defaults.yaml, datasets/, model/, training/}` layout with placeholders for Indian Pines, Pavia University, Salinas, Houston 2013, the full CFA-GDRO model, and the four label-scarcity regimes (3 / 5 / 10 / 20 samples per class).
  - `requirements.txt` and `requirements-dev.txt`.
  - `tests/conftest.py` with auto-used deterministic seeding and tiny-tensor / scene-frequency fixtures; `tests/test_smoke.py` proves the package and every subpackage import.
  - `scripts/{train.py, run_multi_seed.py, run_baselines.py, run_ablations.py, run_robustness.py, make_main_tables.py, make_figures.py}` as importable stubs that exit 0 with a phase pointer.
  - `.pre-commit-config.yaml` (trailing whitespace, end-of-file, large-file guard, ruff, black).
  - `LICENSE` (MIT) and this `CHANGELOG.md`.
