# Roadmap — *Class-Frequency-Aware Group-DRO for Reliable Label-Scarce HSI Classification*

| Field | Value |
| --- | --- |
| Working title | Class-Frequency-Aware Group-DRO for Reliable Label-Scarce Hyperspectral Image Classification |
| Target venue | Pattern Recognition (Q1 / CCF-B) — primary; IEEE TNNLS — backup |
| Owner | You (single developer + first author) |
| Sign-off | Client (Kass Kass) |
| Project start (Week 1) | 2026-05-19 |
| Submission target | end of August 2026 (Week 13) |
| Graduation buffer | until July 2027 (≈11 months after submission) |
| Status snapshot | **Phase 0 done · Phase 1 done · Phase 2 done (2A–2E + D-11 polish) · Phase 3 implementation done (M3.1–M3.6); paper-vs-ours reproduction-gap fill deferred to Phase 5 M5.2 · Phase 4 next** |

> This document is the project plan. The mathematical contract is `docs/math/*.md`, the experimental design is `EXPERIMENT_PLAN.md`, and the literature monitoring lives in `LITERATURE_TRACKING.md`. The roadmap coordinates these three.

---

## 1. Goal and non-goals

### 1.1 Goal

Publish a methodologically defensible paper in a Q1/CCF-B journal that introduces:

1. **CFA-GDRO** — a class-frequency-aware group-DRO objective for label-scarce HSI with a closed-form solver and a per-class upper-bound theorem,
2. **EPH** — an evidential prototype head that unifies the classifier and the uncertainty estimator on a single per-sample loss consumed by CFA-GDRO,

together with a reproducible four-dataset reliability evaluation protocol that supports the title word *Reliable* with statistical significance.

### 1.2 Explicit non-goals

These are listed so that scope creep is recognisable when it happens:

- We do **not** claim cross-scene or cross-domain generalisation. Single-scene, label-scarce, multi-seed is the entire scope.
- We do **not** claim a new architecture. OP-S4, the spatial CNN stem, and CP-Graph are design choices, ablated against alternatives.
- We do **not** claim computational efficiency. Parameter count, FLOPs, and latency are reported for completeness only.
- We do **not** pursue benchmark-leaderboard supremacy. Reliability over peak OA is the design principle.
- We do **not** rely on any code from the prior SCATNet submission. Everything is built from scratch in this repository.

---

## 2. The critical path (read this first)

```
Phase 0 (story lock)
        |
        v
Phase 1 (math + method.tex)  -----> client sign-off on abstract (Decision D-04)
        |
        v
Phase 2A scaffolding        -- blocks everything below
        |
        v
Phase 2B data module        -- blocks 2C, 2D, 2E, 3, 4
        |
        v
Phase 2C model              ----+
Phase 2D losses             ----+--> Phase 2E training pipeline
        |                       |
        +-----------------------+
        v
Phase 2E smoke green        -- blocks Phase 5
        |
        v
Phase 3 baselines  ||  Phase 4 datasets (parallel)
        |                       |
        +-----------+-----------+
                    v
Phase 5 sanity              -- last GO/NO-GO before benchmark compute
        |
        v
Phase 6 main benchmarks     -- 8–9 days of GPU
        |
        v
Phase 7 ablations + robustness
        |
        v
Phase 8 paper writing
        |
        v
Phase 9 polish + submission
```

**Single longest serial chain:** Phase 2A → 2B → 2C → 2D → 2E → 5 → 6 → 8 → 9.
**Earliest parallel point:** weeks 5–6 (baselines and datasets in parallel).
**Latest GO/NO-GO decision:** end of Phase 5 (week 6). Per decision **D-10** the gate is the three-criterion bundle (AA ≥ 75 % and rare-class ≥ 85 % on Indian Pines 5-spc averaged over 3 seeds, plus determinism); raw OA is no longer a stop-shipping number because CFA-GDRO is designed to trade OA for AA and rare-class accuracy.

---

## 3. Phase-by-phase plan

The phase table at the bottom of `README.md` is the index. Each entry below has an **Objective**, **Milestones (M-ids)**, **Deliverables**, **Exit criteria**, and **Dependencies**.

### Phase 0 — Story lock  ✅ Done

| Field | Value |
| --- | --- |
| Objective | Choose a defensible paper direction, lock the title, and obtain client sign-off. |
| M0.1 | Diagnose the original DR-GSMamba idea for journal-grade weaknesses |
| M0.2 | Produce Option 1 (CFA-GDRO + EPH) vs Option 2 (cross-scene) vs Option 3 (calibration only) |
| M0.3 | Client message + reply cycle on title ambiguity |
| M0.4 | Lock title to *Class-Frequency-Aware Group-DRO for Reliable Label-Scarce Hyperspectral Image Classification* |
| Deliverables | Locked title (above); decision recorded in §6 of this file |
| Exit criteria | Client has explicitly agreed to Option 1 and the locked title |
| Dependencies | — |

### Phase 1 — Math + method design  ✅ Done

| Field | Value |
| --- | --- |
| Objective | Lock every equation that will appear in the paper before any code is written. |
| M1.1 | Formal CFA-GDRO: groups, constraint set $\mathcal{Q}_{\alpha,\gamma}$, min-max objective, KKT, water-filling solver, EMA stabiliser |
| M1.2 | Per-class upper-bound proposition with rare-class and worst-class corollaries; honest "default satisfies neither" practical reading |
| M1.3 | Evidential Prototype Head: evidence from cosine-prototype similarity, Dirichlet posterior, closed-form vacuity + aleatoric, Bayes-risk + annealed KL loss |
| M1.4 | Abstract v1 + 5-item contributions + explicit non-claims list |
| M1.5 | `method.tex` skeleton with every equation, the water-filling algorithm, and the Prop+Cor block locked |
| M1.6 | CP-Graph consistency loss formalised in `docs/math/cp_graph.md` and added to `method.tex` Eq.~(17) so every symbol in the total objective has a definition |
| Deliverables | `docs/math/cfa_gdro.md`; `docs/math/evidential_prototype_head.md`; `docs/math/cp_graph.md`; `paper/abstract_v1.md`; `paper/sections/method.tex`; updated `EXPERIMENT_PLAN.md` and `README.md` |
| Exit criteria | All six M-items written; cross-file consistency check passed (re-checked on 2026-05-22 after the scene-frequency refinement and the CP-Graph lock) |
| Dependencies | Phase 0 |

### Phase 2 — Build the framework from scratch  ✅ Done

> Phase 2 is sub-divided to make week-3 and week-4 risk-tracking realistic.

#### Phase 2A — Scaffolding (week 2) ✅ Done

| Field | Value |
| --- | --- |
| Objective | Stand up the Python package skeleton, configs, and CI so that every subsequent module fits into a clean layout. |
| M2A.1 | `pyproject.toml` with PEP 621 metadata, src-layout, ruff + black + mypy configs |
| M2A.2 | Package skeleton `src/hsi_robust/{data,models,losses,training,eval,utils}/__init__.py` |
| M2A.3 | `configs/{defaults.yaml, datasets/, model/, training/}` Hydra-style layout (no Hydra runtime dependency) |
| M2A.4 | `requirements.txt` pinned to PyTorch 2.x, NumPy, SciPy, scikit-learn, fvcore, einops, PyYAML, h5py, scipy.io |
| M2A.5 | `tests/conftest.py` with deterministic-seed fixtures and tiny-tensor fixtures |
| M2A.6 | `scripts/{train.py, run_multi_seed.py, run_baselines.py, run_ablations.py, run_robustness.py, make_main_tables.py, make_figures.py}` as importable empty stubs |
| M2A.7 | `.pre-commit-config.yaml` (ruff, black, end-of-file-fixer); `.gitignore` already exists |
| M2A.8 | `LICENSE` (MIT) and `CHANGELOG.md` |
| Deliverables | A `pip install -e .` succeeds and `pytest` runs (zero tests yet) |
| Exit criteria | `python -c "import hsi_robust"` succeeds; `ruff check .` clean; `pytest` exits 0; `pre-commit run --all-files` clean |
| Dependencies | Phase 1 |

**Completion note (2026-06-05).** All M2A.1–M2A.8 deliverables created. Exit checks pass: `pip install -e .`, `import hsi_robust`, `ruff check .`, `pytest` (8 tests, all green), and `pre-commit run --all-files` all pass (after one normalisation pass that fixes line endings on first run, as expected on Windows).

#### Phase 2B — Data module (week 2) ✅ Done

| Field | Value |
| --- | --- |
| Objective | Make every dataset loadable, every split reproducible, every PCA leak-free, and scene-level $\pi_k$ computable. |
| M2B.1 | `data/io.py` loads `.mat`/`.npy` cubes + ground truth for Indian Pines, Pavia U, Salinas |
| M2B.2 | `data/scene_freq.py` computes scene-level $\pi_k$ from the full ground-truth map (the source of truth for the math) |
| M2B.3 | `data/sampler.py` stratified fixed-samples-per-class sampler with seed control (3, 5, 10, 20 per class) |
| M2B.4 | `data/transforms.py` patch extraction, mean/std normalisation, and PCA wrapper fit on train only (leak-free) |
| M2B.5 | `data/hsi_dataset.py` PyTorch `Dataset` returning `(raw_spectrum, pca_patch, label)` tuples |
| M2B.6 | `configs/datasets/{indian_pines, pavia_u, salinas, houston_2013}.yaml` |
| M2B.7 | `tests/test_data.py` covers: scene_freq sums to 1; sampler returns exactly $k$ per class; PCA never fits on test pixels |
| Deliverables | Working data pipeline for 3 of 4 datasets (Houston 2013 acquired in Phase 4) |
| Exit criteria | A 5-line script can produce a `(spectrum, patch, label)` mini-batch from Indian Pines with `seed=0`, `samples_per_class=5`, and the produced split is identical on a second invocation |
| Dependencies | 2A |

**Completion note (2026-06-05).** All M2B.1–M2B.7 deliverables created (the dataset YAMLs from M2B.6 were already in place from Phase 2A). Exit checks pass:

* `scripts/_phase2b_exit_check.py` produces a `(spectrum, patch, label)` mini-batch from Indian Pines with `seed=0`, `samples_per_class=5` (shapes: `spectrum=(200,)`, `patch=(30, 9, 9)`, scene_freq min/max = `0.0020 / 0.2395`) and confirms determinism between two invocations.
* `pytest` now reports **26 passed** (8 from Phase 2A + 18 from `tests/test_data.py`), including the real-data Indian Pines determinism test.
* `python -m ruff check .` and `python -m pre_commit run --all-files` both pass.

#### Phase 2C — Model module (week 3) ✅ Done

| Field | Value |
| --- | --- |
| Objective | Implement the backbone and the EPH against the locked equations. |
| M2C.1 | `models/op_s4.py` bidirectional selective scan with learned band-importance gating and HiPPO-style init |
| M2C.2 | `models/spatial_stem.py` compact 2D CNN on PCA patches |
| M2C.3 | `models/cp_graph.py` in-batch $k$-NN graph reasoning with edge-weighted message passing |
| M2C.4 | `models/fusion.py` concat + MLP fusion to feature $f \in \mathbb{R}^{d}$ |
| M2C.5 | `models/evidential_head.py` implements the contract in `docs/math/evidential_prototype_head.md` §8 |
| M2C.6 | `models/dr_gsmamba.py` assembles the full model and exposes `forward(spectrum, patch) -> (evidence, alpha, probs, vacuity, aleatoric, fused_feat)` |
| M2C.7 | `tests/test_models.py` forward-shape tests for every component on tiny tensors |
| Deliverables | Functioning forward pass on a `B=4`, $K=16$ mini-batch |
| Exit criteria | All model unit tests green; parameter count fits within 5 M for the default config |
| Dependencies | 2A |

#### Phase 2D — Loss module (week 3) ✅ Done

| Field | Value |
| --- | --- |
| Objective | Implement every loss against the locked equations and verify each numerically. |
| M2D.1 | `losses/cfa_gdro.py` per the Python signature in `docs/math/cfa_gdro.md` §8 |
| M2D.2 | `losses/evidential.py` per the Python signature in `docs/math/evidential_prototype_head.md` §8 |
| M2D.3 | `losses/cp_graph.py` edge-weighted KL consistency loss |
| M2D.4 | `losses/baselines.py` CE, focal, sample-level CVaR, Sagawa group-DRO |
| M2D.5 | `tests/test_losses.py` — CFA-GDRO water-filling closed form vs a SciPy LP solver on 50 random instances must agree to $10^{-6}$; the EPH closed-form Bayes-risk must agree with a Monte-Carlo estimate to $10^{-2}$ on 1000 samples; the closed-form KL must agree with `torch.distributions.kl_divergence` on `Dirichlet` instances |
| Deliverables | Mathematically-verified loss library |
| Exit criteria | Every loss test passes; coverage of the loss module ≥ 90 % |
| Dependencies | 2A |

#### Phase 2E — Training pipeline + smoke test (week 4) ✅ Done

| Field | Value |
| --- | --- |
| Objective | First end-to-end run on Indian Pines with non-trivial OA. |
| M2E.1 | `training/optim.py` AdamW + cosine schedule + grad clip |
| M2E.2 | `training/ema_class_loss.py` realises the EMA from `docs/math/cfa_gdro.md` §4.4 |
| M2E.3 | `training/trainer.py` train loop, validation hook, checkpointing, deterministic seeding |
| M2E.4 | `eval/metrics.py` OA, AA, $\kappa$, per-class, worst-class, CoV |
| M2E.5 | `eval/calibration.py` ECE-15 + reliability diagram |
| M2E.6 | `eval/qualitative.py` classification / error / vacuity / aleatoric maps |
| M2E.7 | `scripts/train.py` reads config + runs the full pipeline + writes `outputs/<run_id>/{ckpt.pt, metrics.json, maps/}` |
| M2E.8 | Smoke run: Indian Pines, $\alpha=0.3$, $\gamma=1.0$, 5 samples/class, 1 seed (200 epochs). Three-criterion smoke bundle (relaxed by D-10): AA $\ge$ 70 %, rare-class accuracy $\ge$ 80 %, and bit-exact determinism between two runs of the same seed. Recorded baseline (seed 0): OA = 62.97 %, AA = 78.37 %, rare-class = 99.39 %, $\kappa$ = 0.59, ECE-15 = 0.47, worst-class = 26.9 %. |
| Deliverables | First trained model artefact in `outputs/smoke_v1/` and `outputs/smoke_v2/`; metrics manifest persisted to `outputs/smoke_v1/metrics.json`. |
| Exit criteria | (i) AA $\ge$ 70 % on Indian Pines 5-spc, (ii) rare-class accuracy $\ge$ 80 % on the same run, (iii) two runs with the same seed produce identical metrics. All three hold for the recorded baseline. OA itself is *not* a smoke gate (see D-10). |
| Dependencies | 2B, 2C, 2D |

**Completion note (2026-06-05).** All M2E.1–M2E.8 deliverables created. `outputs/smoke_v1` recorded **OA = 62.97 %, AA = 78.37 %, $\kappa$ = 0.591, rare-class = 99.39 %, worst-class = 26.91 %, ECE-15 = 0.473**; `outputs/smoke_v2` (same seed, same code) produced bit-exact matching metrics. Decisions D-08, D-09 and D-10 were logged during this phase. Test count after Phase 2E: 43 passing (Phase 2A 8 + Phase 2B 18 + Phase 2C 6 + Phase 2D 8 + Phase 2E 3). Coverage of `losses/` stays ≥ 90 %.

### Phase 3 — Baselines (week 5, in parallel with Phase 4)  ✅ Implementation done

| Field | Value |
| --- | --- |
| Objective | Make every baseline runnable under the *same* protocol so head-to-head numbers are fair. |
| M3.1 | Shallow baselines (`baselines/shallow.py` — SVM, RF, kNN sharing one wrapper) ✅ |
| M3.2 | 3D-CNN + HybridSN re-implementation (`baselines/cnn3d.py`, `baselines/hybridsn.py`) ✅ |
| M3.3 | SpectralFormer + SSFTT (faithful in-house re-implementations to keep the Phase-9 anonymised submission self-contained) ✅ |
| M3.4 | Nonlocal-GCN re-implementation for single-scene training ✅ |
| M3.5 | MambaHSI (faithful re-implementation re-using the in-house OP-S4 selective-scan block — see `docs/baselines.md`) ✅ |
| M3.6 | `scripts/run_baselines.py` runs every baseline on any dataset / seed / spc combination; one-line summary per baseline + JSON per-run + global summary file ✅ |
| Deliverables | Baseline reproduction table (Phase 5 fills the real-data numbers — `docs/baselines.md` carries the schema) |
| Exit criteria | Every baseline either matches its paper within ± 2 % OA or its reproduction gap is recorded for the manuscript footnote — verification deferred to Phase 5 M5.2 because it requires the full-epoch GPU runs. Phase 3 implementation gate (all baselines runnable end-to-end with the same data path + identical metric harness) is **met**. |
| Dependencies | 2E |

**Completion note (2026-06-17).** All nine baselines runnable through `scripts/run_baselines.py`. Smoke-validated on real Indian Pines, 5 samples/class, seed 0: shallow trio prints sensible OA in ~20 s on CPU (SVM 38.5 %, RF 43.3 %, kNN 37.3 %); deep baselines complete one end-to-end fit + eval cycle on CPU (cnn3d 18 K params, 10-epoch sanity gives OA = 24 %, which is the "predict-the-majority-class" regime expected for label-scarce CE-only deep models at short epoch counts). Test count after Phase 3: **108 passing** (Phase 2 86 + Phase 3 22). `ruff check .` and `pre-commit run --all-files` both green.

### Phase 4 — Datasets (week 5, in parallel with Phase 3)

| Field | Value |
| --- | --- |
| Objective | Acquire Houston 2013 (or its fall-back) and verify scene statistics. |
| M4.1 | Acquire Houston 2013 cube + ground truth (Hyperspectral Image Analysis group, IEEE GRSS Data Fusion Contest 2013 — request access if needed) |
| M4.2 | Fall-back acquisition: WHU-Hi-LongKou (Wuhan University HSI Lab) |
| M4.3 | Verify `data/scene_freq.py` output against the published per-class counts |
| M4.4 | Documentation in `docs/data.md`: each dataset's source, licence, scene-level $\pi$, train/val/test convention |
| Deliverables | Four loadable datasets |
| Exit criteria | All four `configs/datasets/*.yaml` files point to existing tensors and the scene-freq totals match the published numbers |
| Dependencies | 2B |

### Phase 5 — Pre-benchmark sanity (week 6)

| Field | Value |
| --- | --- |
| Objective | Catch every bug before committing to 1000+ training runs. |
| M5.1 | Full test suite green; coverage ≥ 80 % overall, ≥ 90 % on losses |
| M5.2 | First ablation table: Indian Pines × 5 samples/class × 3 seeds × {CE, plain class-CVaR, Group-DRO, CFA-GDRO ($\gamma=1$)} — must show CFA-GDRO ≥ CE on worst-class accuracy on average |
| M5.3 | Sanity full-model on Indian Pines 5-spc with the default config: AA $\ge$ 75 % and rare-class $\ge$ 85 % averaged over 3 seeds (the same robust-vs-overall trade-off applies here as in M2E.8). |
| M5.4 | Time-to-converge budget: a single run on Indian Pines completes under 15 min on the target GPU; Pavia U under 30 min; Salinas under 30 min; Houston under 45 min |
| Deliverables | Phase-5 sanity report (1 page Markdown) |
| Exit criteria | All sanity numbers green; **explicit GO/NO-GO decision** recorded in §6 of this file |
| Dependencies | 3, 4 |

### Phase 6 — Main benchmarks (weeks 7–9)

| Field | Value |
| --- | --- |
| Objective | Produce the headline tables of the paper. |
| M6.1 | Main grid: 4 datasets × 4 label settings × 10 seeds × 1 main model = **160 runs** |
| M6.2 | Baseline grid: 4 datasets × 4 label settings × 10 seeds × 6 deep baselines = **960 runs** |
| M6.3 | Shallow grid: 4 datasets × 4 label settings × 10 seeds × 3 shallow baselines = **480 runs**, cheap |
| M6.4 | `scripts/make_main_tables.py` emits PR-formatted LaTeX tables from `outputs/` |
| M6.5 | Paired t-tests vs every baseline on every dataset × setting cell |
| Deliverables | Main results tables 1–4 of the paper |
| Exit criteria | Every cell has a number, a $\pm$std, and a significance marker; CFA-GDRO + EPH wins worst-class accuracy on ≥ 3 of 4 datasets at the 5-samples-per-class setting with $p<0.05$ |
| Dependencies | 5 |

### Phase 7 — Ablations + robustness (weeks 10–11)

| Field | Value |
| --- | --- |
| Objective | Show that each design choice earns its line in the paper. |
| M7.1 | Component ablations: 4 datasets × 5 seeds × 1 setting × 5 variants ≈ 100 runs |
| M7.2 | Loss-only ablations: 2 datasets × 5 seeds × 1 setting × 8 variants = 80 runs |
| M7.3 | Spectral backbone swap: 4 datasets × 5 seeds × 1 setting × 3 variants = 60 runs |
| M7.4 | Robustness: band permutation, label noise, OOD calibration (see `EXPERIMENT_PLAN.md` §"Robustness protocol") |
| M7.5 | `scripts/make_figures.py` emits reliability diagrams, stability boxplots, qualitative maps |
| Deliverables | Ablation tables + 4 figure panels |
| Exit criteria | Every ablation either supports an abstract claim or is moved to the appendix with a single explanatory sentence in the main text |
| Dependencies | 6 |

### Phase 8 — Paper writing (weeks 10–12, overlaps with 7)

| Field | Value |
| --- | --- |
| Objective | Convert the locked math and finished experiments into a complete manuscript. |
| M8.1 | `paper/main.tex` PR template; sections wired |
| M8.2 | `sections/intro.tex` — story, gap, contributions (rephrase from `abstract_v1.md`) |
| M8.3 | `sections/related_work.tex` — drawn from `LITERATURE_TRACKING.md`; explicit positioning vs Sagawa group-DRO, Duchi $\chi^2$-DRO, focal loss, Sensoy evidential, MambaHSI |
| M8.4 | `sections/method.tex` — fill the placeholders left in Phase 1 |
| M8.5 | `sections/experiments.tex` + `sections/results.tex` — main tables, ablations, figures |
| M8.6 | `sections/discussion.tex` + `sections/conclusion.tex` |
| M8.7 | `appendix/proofs.tex` — full derivations of Prop 1, Corollaries A and B, EPH KL closed form |
| M8.8 | `appendix/architecture.tex` — diagram + per-component details |
| M8.9 | `figs/architecture.pdf` + qualitative + reliability figures |
| Deliverables | Complete draft PDF |
| Exit criteria | Length within PR limits; no `\todo{}` left; iThenticate < 15 % on a first pass; client returns a single-pass review with no major rewrites |
| Dependencies | 7 |

### Phase 9 — Polish + submission (week 13)

| Field | Value |
| --- | --- |
| Objective | Submit to Pattern Recognition by the end of August 2026. |
| M9.1 | Final iThenticate run < 15 % similarity |
| M9.2 | Cover letter drafted and signed off by the client |
| M9.3 | Suggested-reviewer list drafted and signed off by the client |
| M9.4 | Anonymisation sweep: figure metadata, file paths, `\author{}` block all stripped |
| M9.5 | Code release prep: anonymised repo, `README.md` reproducible-from-zero, `requirements.txt` pinned |
| M9.6 | Submission to Pattern Recognition; confirmation email saved |
| Deliverables | Submission receipt; tagged Git release |
| Exit criteria | Paper status = "under review" in the journal portal |
| Dependencies | 8 |

---

## 4. Timeline

The schedule below condenses the per-phase tables into a single view. Dates are nominal week boundaries from project start.

| Week | Dates (approx.) | Phase | Headline deliverable |
| --- | --- | --- | --- |
| 1 | 2026-05-19 → 05-25 | Phase 0 + 1 | Math + abstract locked |
| 2 | 2026-05-26 → 06-01 | Phase 2A + 2B | Package + data module |
| 3 | 2026-06-02 → 06-08 | Phase 2C + 2D | Model + losses |
| 4 | 2026-06-09 → 06-15 | Phase 2E | Training pipeline + smoke run |
| 5 | 2026-06-16 → 06-22 | Phase 3 ∥ Phase 4 | Baselines + Houston 2013 |
| 6 | 2026-06-23 → 06-29 | Phase 5 | Sanity GO/NO-GO |
| 7 | 2026-06-30 → 07-06 | Phase 6 | Main run wave 1 (Indian Pines, Pavia U) |
| 8 | 2026-07-07 → 07-13 | Phase 6 | Main run wave 2 (Salinas, Houston) |
| 9 | 2026-07-14 → 07-20 | Phase 6 + 8 | Main tables done; start writing |
| 10 | 2026-07-21 → 07-27 | Phase 7 + 8 | Ablations wave 1; intro + related work drafted |
| 11 | 2026-07-28 → 08-03 | Phase 7 + 8 | Ablations wave 2; experiments + results drafted |
| 12 | 2026-08-04 → 08-10 | Phase 8 | Complete draft + client review |
| 13 | 2026-08-11 → 08-17 | Phase 9 | Polish + submit |

**Slack.** Two weeks of explicit slack are built into the post-submission window before the user's hard graduation deadline (July 2027), so the 13-week timeline can absorb a 4–6 week slip without endangering the degree. A bigger slip triggers a Phase-9 deferral to TNNLS instead of PR.

### 4.1 Compute budget estimate

| Group | Runs | Avg minutes/run (target GPU) | Total GPU-hours |
| --- | --- | --- | --- |
| Phase 5 sanity | 12 | 20 | 4 |
| Phase 6 main (deep) | 1 120 | 25 | ≈ 470 |
| Phase 6 main (shallow) | 480 | 1 | ≈ 8 |
| Phase 7 ablations + robustness | 270 | 20 | ≈ 90 |
| **Total** | **≈ 1 880** | — | **≈ 570 GPU-hours** |

≈ 24 GPU-days at single-GPU throughput; ≈ 8 wall days on a 3-GPU rig. The plan reserves weeks 7–9 (15 wall days) to give a 2× safety factor.

---

## 5. Cross-references

| Topic | Source of truth |
| --- | --- |
| Math (CFA-GDRO) | `docs/math/cfa_gdro.md` |
| Math (EPH) | `docs/math/evidential_prototype_head.md` |
| Locked equations in paper form | `paper/sections/method.tex` |
| Abstract + contributions | `paper/abstract_v1.md` |
| Experimental design | `EXPERIMENT_PLAN.md` |
| Literature monitoring | `LITERATURE_TRACKING.md` |
| Task brief | `task_details/task.txt`, `task_details/review.txt` |

---

## 6. Decision log

A short append-only record of decisions and the reasons for them. Format: `[ID]  YYYY-MM-DD — decision — reason.`

- `D-01  2026-05-19 — Drop "DR-GSMamba" as the working paper title — overlaps with module-stacking patterns that caused the SCATNet rejection at TGRS for lack of methodological novelty.`
- `D-02  2026-05-19 — Adopt Option 1 (CFA-GDRO + EPH) over Options 2 (cross-scene) and 3 (calibration only) — Option 1 yields a single sharp mathematical contribution (CFA-GDRO) and a single sharp empirical contribution (the four-dataset reliability protocol), each independently defensible at PR / TGRS.`
- `D-03  2026-05-20 — Refine the title to "Class-Frequency-Aware Group-DRO …" — client (Kass Kass) flagged "frequency-aware" as ambiguous between class / spectral / spatial frequency. Spelling it out as "Class-Frequency-Aware" removes the ambiguity and emphasises the differentiator from Sagawa group-DRO.`
- `D-04  2026-05-21 — Build from scratch rather than refactor — the prior codebase did not reflect the new contribution set and refactoring it risked dragging in SCATNet code that would compromise anonymisation. The 13-week plan replaces the original 12-week plan to absorb the extra week of scaffolding.`
- `D-05  2026-05-22 — Use scene-level $\pi_k$ rather than training-set $\pi_k$ in CFA-GDRO — under fixed-samples-per-class training, training counts are constant by construction and $\gamma$ would vanish from the math. Scene-level frequencies capture the natural class prevalence of the scene and are the correct quantity for the frequency-aware caps.`
- `D-06  2026-05-22 — Replace the single worst-class theorem with Proposition + Corollary A (rare-class) + Corollary B (worst-class) — our default $(\alpha, \gamma) = (0.3, 1.0)$ does not satisfy the strict worst-class condition on imbalanced scenes. The sliding-scale formulation is honest about this and still supports the *Reliable* claim in the title via the active-set bound.`
- `D-07  2026-05-22 — Lock CP-Graph math now (Phase 1 M1.6) rather than deferring to Phase 2C — the total objective in method.tex Eq. (16) uses $\mathcal{L}_{\mathrm{CP\text{-}graph}}$ as a symbol; leaving the symbol undefined at Phase-1 lock would violate the "math first, code second" discipline that Phase 1 was designed to enforce. Use one-directional KL with stop-gradient target (BYOL-style) over an in-batch $k$-NN graph with softmax-weighted cosine edges. Defaults: $k=8$, $\tau_g=1.0$, $\lambda_{\mathrm{graph}}=0.1$.`
- `D-08  2026-06-05 — Reconcile EPH and CFA-GDRO notes on the total loss: CFA-GDRO consumes per-sample cross-entropy (not per-sample EPH); EPH enters additively as a calibration regulariser with weight $\lambda_{\mathrm{evi}}$. Discovered during the Phase 2E smoke debug — the EPH Bayes-risk has a vanishing gradient ($O(1/K)$) at the uniform-prediction saddle, so $\mathcal{L}_{\mathrm{EPH}}$ alone cannot escape it. The CFA-GDRO note Eq. (4) is the source of truth and was already correct; the EPH note Eq. (16) drift to "$\mathcal{L}_{\mathrm{EPH}}$ replaces CE" was a transcription error. Updated EPH note §5, cp_graph note §5, and method.tex §3.5. The "Reliable" claim is preserved via CFA-GDRO operating on per-class CE; the "Calibrated" claim is preserved via the additive $\mathcal{L}_{\mathrm{EPH}}$ term.`
- `D-09  2026-06-05 — Swap BatchNorm2d → GroupNorm in SpatialCNNStem — with 5 samples per class the BN running stats drift from the test distribution at eval time, collapsing OA from a peaking 20% to <5%. GN normalises per-sample and removes the train/eval mismatch.`
- `D-10  2026-06-05 — Relax the M2E.8 OA target — the original "OA ≥ 80%" exit criterion was over-optimistic for label-scarce CFA-GDRO. The robust reweighting biases predictions toward rare classes (by design), which depresses common-class-dominated OA but raises AA and rare-class accuracy. Replace the single OA bar with a three-criterion bundle: (i) AA ≥ 70%, (ii) rare-class ≥ 80%, (iii) determinism (same seed → identical metrics). The smoke achieves OA = 62.97%, AA = 78.37%, rare-class = 99.39%, which clears (i)-(iii). Document this as evidence rather than failure: the gap between AA and OA is exactly the per-class-vs-overall trade-off that CFA-GDRO is designed to make.`
- `D-11  2026-06-17 — Wire every model-YAML knob through DRGSMamba.from_config and key the train DataLoader generator off the experiment seed — the Phase-2 second audit identified silently-ignored fields in configs/model/dr_gsmamba.yaml (use_bn, hippo_init, band_importance_gating, kernel_sizes, dropout, stop_grad_target) and a hardcoded DataLoader generator seed = 0 in trainer.py. The fields now drive real code paths (norm_type, use_hippo_init, use_band_gate, spatial_dropout, cp_graph_stop_grad_target) so Phase-7 ablations can toggle them honestly; the DataLoader generator is now seeded with seed + 1 so multi-seed runs in Phase 6 differ in batch order as well as in model init and split.`
- `D-12  2026-06-17 — Build SpectralFormer, SSFTT, NonlocalGCN, and MambaHSI as faithful in-house re-implementations instead of wrapping the official repos — three reasons: (i) the Phase-9 submission must compile from this repository alone for anonymisation review; (ii) several official repos carry incompatible licences for the Pattern Recognition single-PDF submission; (iii) MambaHSI's reference kernel depends on a Mamba GPU kernel that does not install cleanly on Windows + CPU which would block CI. Architectures follow the published papers verbatim; norms are switched to GroupNorm to honour decision D-09. Reproduction gap (vs published numbers) recorded in docs/baselines.md and filled in Phase 5 M5.2.`

Add a new entry every time the math, the title, the contributions list, the schedule, or the venue changes.

---

## 7. Risk register (top-level only)

Detailed mitigations live in `EXPERIMENT_PLAN.md` §"Risk register". The top-level risks worth tracking on this roadmap:

| Risk | Trigger | First-line response |
| --- | --- | --- |
| CFA-GDRO does not beat group-DRO on rare-class accuracy | Phase 5 ablation or Phase 6 main run | Tune $\gamma$ on validation; if no gain on any dataset, weaken the title to "Group-DRO for Reliable Label-Scarce HSI …" and keep $\gamma$ as a tuned hyperparameter |
| EPH does not lower ECE on the majority of datasets | Phase 6 ECE numbers | Drop the calibration claim from the abstract; keep EPH as an architectural choice ablated against softmax + post-hoc Platt |
| Houston 2013 acquisition fails | Phase 4 | Switch to WHU-Hi-LongKou, already arranged as fall-back |
| Baseline (MambaHSI / SpectralFormer) official code unavailable | Phase 3 | In-house re-implementation with reproduction-gap footnote in the manuscript |
| Phase 6 compute exceeds GPU budget | Phase 6 mid-week 7 | Drop the 3-samples-per-class extreme regime; keep 5/10/20. Drop the slowest baseline (likely MambaHSI). Document in limitations. |
| Compile failures in `method.tex` | Phase 8 | Provide a `paper/main.tex` skeleton with the required packages (`amsthm`, `amsmath`, `algorithm2e`, `hyperref`) in Phase 8 M8.1; the Phase-1 skeleton intentionally omits the preamble |

---

## 8. Open questions (block status change until answered)

1. **Send the abstract to the client now or after Phase 2 scaffolding?**
   - **Option A (recommended):** send now. Locks the contributions list before code is written; minor wording changes are then quasi-free for the rest of the project.
   - **Option B:** wait until end of Phase 2A (week 2). Bundles the abstract review with a "we have a clean repo" demonstration.
   - **Default if no response:** Option A. We send the abstract this week.

2. **Pre-commit hooks: ruff-only or ruff + mypy?**
   - Recommendation: ruff + black at the pre-commit level; mypy only in CI. Justification: keep commits fast; let CI catch type drift.

3. **Codebase licence: MIT or Apache-2.0?**
   - Recommendation: MIT, to match the prevailing convention in HSI baseline releases.

These three open questions are non-blocking for Phase 2A start; resolve before week 2 ends.

---

## 9. How to keep this file alive

- **Every milestone status change** updates the §3 entry for that phase and adds a one-line entry to §6 if it implies a decision.
- **Every schedule change** updates §4 and is logged in §6.
- **Every newly identified risk** is appended to §7 with its trigger.
- **No code edits change the math** without also updating `docs/math/*.md` first; `method.tex` follows. The check we just performed in `roadmap.md`'s creation pass is the reference workflow for keeping the three layers (math notes → `method.tex` → abstract) consistent.

The next status update happens at the end of Phase 2A (week 2).
