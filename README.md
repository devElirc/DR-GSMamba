# CFA-GDRO for Reliable Label-Scarce Hyperspectral Image Classification

**Working title**
*Class-Frequency-Aware Group-DRO for Reliable Label-Scarce Hyperspectral Image Classification.*

This repository is being rebuilt from scratch around a single core contribution: a class-frequency-aware group-distributionally-robust objective (CFA-GDRO) for label-scarce hyperspectral image classification, complemented by an evidential prototype head for calibrated uncertainty.

**Phase 1 (Math + Method Design), Phase 2A (Scaffolding), and Phase 2B (Data module) are complete.** The next milestone is Phase 2C (model module). The detailed plan from Phase 0 through submission lives in `roadmap.md`; the experimental design lives in `EXPERIMENT_PLAN.md`.

## What is in the repository right now

- `data/` &mdash; HSI cubes and ground-truth files (Indian Pines, Pavia University, Pavia Centre, Salinas; Houston 2013 placeholder).
- `related_paper/` &mdash; recent HSI literature used for related-work analysis and baseline selection.
- `task_details/` &mdash; client brief, reviewer feedback from the prior SCATNet submission, and related material.
- `LITERATURE_TRACKING.md` &mdash; rolling notes on recent HSI papers relevant to the project.
- `roadmap.md` &mdash; full Phase 0 → Phase 9 plan with milestones, exit criteria, and the critical path.
- `EXPERIMENT_PLAN.md` &mdash; the experimental design that the roadmap operationalises.
- `docs/math/` &mdash; formal definitions and proof sketches written in Phase 1 (`cfa_gdro.md`, `evidential_prototype_head.md`, `cp_graph.md`).
- `paper/abstract_v1.md` &mdash; locked abstract and contributions list.
- `paper/sections/method.tex` &mdash; LaTeX skeleton with the locked equations.

## Phase status

| Phase | Description | Status |
| --- | --- | --- |
| 0 | Story lock (Option 1, title locked) | Done |
| 1 | Math and method design | Done |
| 2 | Build the framework from scratch | In progress (2A and 2B done; 2C next) |
| 3 | Baselines (faithful re-implementations / official wrappers) | Pending |
| 4 | Datasets (Houston 2013 or WHU-Hi-LongKou) | Pending |
| 5 | Pre-benchmark sanity (unit tests, first ablation) | Pending |
| 6 | Main benchmarks (4 datasets, 10 seeds, 4 label settings) | Pending |
| 7 | Ablations and robustness | Pending |
| 8 | Paper writing | Pending |
| 9 | Polish and submission to Pattern Recognition | Pending |

A reproducible code package skeleton with the full data pipeline is now present. The training-ready package will appear at the end of Phase 2.
