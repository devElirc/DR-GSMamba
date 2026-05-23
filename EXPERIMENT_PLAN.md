# Experiment Plan — CFA-GDRO Paper

This plan operationalises **Option 1** (locked with the client on 2026-05-22).

**Working title.** *Class-Frequency-Aware Group-DRO for Reliable Label-Scarce Hyperspectral Image Classification.*

**Target venue.** Pattern Recognition (Q1 / CCF-B) as primary; IEEE TNNLS as backup. Submission target: end of August 2026. Graduation buffer: 11 months until July 2027.

The math is locked in:

- `docs/math/cfa_gdro.md` (M1.1 + M1.2)
- `docs/math/evidential_prototype_head.md` (M1.3)
- `paper/abstract_v1.md` (M1.4)
- `paper/sections/method.tex` (M1.5)
- `docs/math/cp_graph.md` (M1.6)

This document orchestrates Phase 2 onwards.

---

## Core question

> Can a class-frequency-aware group-DRO objective combined with an evidential prototype head improve the reliability — worst-class accuracy, rare-class accuracy, split stability, and calibration — of hyperspectral image classification under label scarcity, while maintaining competitive overall accuracy?

If the answer is no, the paper is not viable; we will adjust the wording rather than overclaim.

---

## Claims that must be proven experimentally

1. **CFA-GDRO improves worst-class and rare-class accuracy versus plain CE, focal, plain CVaR, and Sagawa-style group-DRO**, on at least three of the four benchmark datasets, with paired-t-test significance over ten seeds.

2. **CFA-GDRO improves split stability**: lower standard deviation, lower coefficient of variation, and higher worst-split accuracy versus the same model trained with plain CE under the same protocol.

3. **EPH lowers expected calibration error** (ECE) versus a softmax head with the same backbone, on at least three of the four datasets.

4. **The frequency-aware cap is not redundant with vanilla group-DRO**: the ablation $\gamma = 0$ (uniform-cap class-CVaR) under-performs $\gamma > 0$ (CFA-GDRO) on rare-class accuracy.

5. **The backbone is interchangeable**: the gains attributed to CFA-GDRO + EPH persist when the spectral encoder is swapped between OP-S4, a Transformer, and a 1D-CNN of comparable parameter counts.

If any claim fails the experimental test, the corresponding wording in the paper will be softened, and the failure will be discussed in the limitations subsection.

---

## Required reporting

| Category | Metrics |
| --- | --- |
| Main accuracy | OA, AA, Cohen's $\kappa$ |
| Per-class | per-class accuracy, per-class F1 |
| Reliability | macro-F1, worst-class accuracy, rare-class accuracy |
| Stability | mean ± std, coefficient of variation, worst-split, all over 10 seeds |
| Calibration | ECE-15, reliability diagrams, mean vacuity |
| Efficiency | parameter count, FLOPs (via fvcore), GPU latency |
| Statistics | paired t-tests against every baseline and every ablation |

---

## Dataset scope

| Dataset | Bands | Classes | Spatial size | Status |
| --- | --- | --- | --- | --- |
| Indian Pines | 200 | 16 | 145 × 145 | Available in `data/indian_pines/` |
| Pavia University | 103 | 9 | 610 × 340 | Available in `data/pavia_university/` |
| Salinas | 204 | 16 | 512 × 217 | Available in `data/salinas/` |
| Houston 2013 | 144 | 15 | 349 × 1905 | **TBD** in Phase 4; fall-back: WHU-Hi-LongKou |

Pavia Centre is *not* used in the main table because (i) it overlaps with Pavia University and (ii) adding it lengthens the experiments without changing the story. It will be reserved as an out-of-distribution dataset for the calibration stress test in Phase 7.

---

## Label-scarcity protocol

Fixed samples per class — not percentage splits — for fair rare-class evaluation:

- 3 samples per class (extreme)
- 5 samples per class (very low)
- 10 samples per class (low)
- 20 samples per class (moderate)

Stratified, seed-controlled, leakage-safe (PCA fit on train only).

Ten random seeds: `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]`.

**Scene-level class frequencies.** Because training counts are constant by construction under this protocol, CFA-GDRO uses *scene-level* frequencies $\pi_k$ computed once per dataset from the full ground-truth map (including labelled-but-unused pixels). This is the source-of-truth definition in `docs/math/cfa_gdro.md` §1 and is also the argument name (`scene_freq`) in the implementation contract.

---

## Baselines

Three families, each addressing a specific reviewer concern.

### Shallow

- SVM (RBF kernel) on raw centre spectra
- Random Forest with class-balanced weights
- $k$-NN with $k = 5$

### Deep — CNN and Transformer

- 3D-CNN (compact spectral-spatial baseline)
- HybridSN (Roy et al. 2020) — 3D + 2D hybrid
- SpectralFormer (Hong et al. 2022) — official code if licence permits
- SSFTT (Sun et al. 2022) — official code

### Deep — Graph and Mamba

- Nonlocal GCN (Mou et al. 2020) — faithful single-scene re-implementation
- MambaHSI (recent) — official repo

Every baseline must reproduce its paper's Indian Pines OA within ± 2 %; otherwise we report the reproduction gap explicitly in the manuscript and footnote our reproduction methodology.

---

## Ablation matrix

Each ablation runs on all four datasets × five seeds × five samples per class (the regime where reliability differences are clearest).

**Component ablations** (does each piece help?)

| Variant | Removes |
| --- | --- |
| Ours (full) | — |
| –CFA-GDRO ($\lambda_{\mathrm{rob}}=0$) | the robust objective entirely |
| –EPH | replaces evidential head with softmax + cross entropy |
| –CP-Graph | replaces in-batch graph with intra-patch attention |
| –OP-S4 | replaces spectral SSM with a Transformer of same params |

**Loss-only ablations** (Indian Pines + Pavia U, five seeds; on the same backbone)

| Variant | Loss |
| --- | --- |
| CE | plain cross entropy |
| Focal | focal loss with $\gamma_F = 2$ |
| CVaR | sample-level CVaR (Rockafellar) |
| Group-DRO | Sagawa worst-class |
| Class-CVaR ($\gamma=0$) | uniform-cap class-CVaR |
| CFA-GDRO ($\gamma{=}0.5$) | weak frequency awareness |
| CFA-GDRO ($\gamma{=}1.0$) | default |
| CFA-GDRO ($\gamma{=}1.5$) | strong frequency awareness |

**Spectral backbone swap** (does the title-less Mamba choice matter?)

| Variant | Backbone |
| --- | --- |
| OP-S4 | bidirectional selective scan on raw bands (default) |
| Transformer | shallow Transformer on raw bands |
| 1D-CNN | three-block 1D-CNN on raw bands |

The most important comparisons are CFA-GDRO vs CE, CFA-GDRO vs Group-DRO, CFA-GDRO ($\gamma{=}1$) vs CFA-GDRO ($\gamma{=}0$), and EPH vs softmax. If these three differences are not statistically significant, the contribution wording will be weakened.

---

## Robustness protocol (Phase 7)

- **Band permutation** on Indian Pines and Pavia U: shuffle band indices at test time. OP-S4 should drop more than Transformer; if not, we discontinue the "order-preserving" framing in the architecture subsection.
- **Label noise**: train with 5 %, 10 %, 20 % label flips on Indian Pines; check that CFA-GDRO degrades less than CE.
- **OOD stress on calibration**: zero out 20 % of bands at test time on Pavia Centre and verify that vacuity rises while EPH probabilities stay calibrated.

---

## Qualitative evidence

Auto-generated for every dataset:

- ground-truth map,
- prediction map,
- error map (predicted ≠ true),
- vacuity map (epistemic),
- aleatoric map.

Plus reliability diagrams per dataset and stability boxplots per metric.

---

## Commands (to be implemented in Phase 2 / 5)

```
python scripts/run_multi_seed.py        --config configs/datasets/indian_pines.yaml
python scripts/run_reliability_protocol.py --config configs/datasets/indian_pines.yaml
python scripts/run_baselines.py         --config configs/datasets/indian_pines.yaml
python scripts/run_ablations.py         --config configs/datasets/indian_pines.yaml
python scripts/run_robustness.py        --config configs/datasets/indian_pines.yaml
python scripts/make_main_tables.py
python scripts/make_figures.py
```

---

## Phase schedule

| Week | Phase | Major deliverable |
| --- | --- | --- |
| 1 | 1 | Math + method.tex + abstract (DONE) |
| 2 | 2A + 2B | Scaffolding + data module |
| 3 | 2C + 2D | Model and loss modules |
| 4 | 2E | Training pipeline, smoke green, IP sanity OA > 80 % |
| 5 | 3 + 4 | All baselines + all datasets runnable |
| 6 | 5 | Unit tests green, first ablation table |
| 7–9 | 6 | Main results tables (start writing in week 9) |
| 10–11 | 7 + 8 | Ablations, robustness, first paper draft |
| 12 | 8 | Complete draft |
| 13 | 9 | Submit to Pattern Recognition |

---

## Risk register

| Risk | Mitigation |
| --- | --- |
| CFA-GDRO does not improve worst-class accuracy | Tune $\gamma$ on validation only; if no gain on any dataset, weaken the title to "Group-DRO for Reliable Label-Scarce HSI Classification" and keep $\gamma$ as a tuned hyperparameter |
| EPH does not lower ECE | Drop the calibration claim from the abstract; keep EPH as an architectural choice ablated against softmax |
| Houston 2013 dataset unavailable | Switch to WHU-Hi-LongKou; both have $\sim 15$ classes and a single-scene split |
| MambaHSI official code does not run | Use a faithful in-house re-implementation, clearly labelled in the manuscript |
| 1 300+ runs in Phase 6 do not fit in GPU budget | Drop the 3-samples-per-class extreme; keep 5/10/20. Drop one slowest baseline (likely MambaHSI) and discuss in limitations |

---

## Literature monitoring (carried over from prior plan)

`LITERATURE_TRACKING.md` continues to be updated whenever the client shares new HSI papers, especially Minchao Ye's updates. Before final benchmark runs, freeze a baseline list and justify every included or excluded recent method in the manuscript.

Most papers in `related_paper/` are cross-scene work. They are cited in related work but **not** used as baselines under the current single-scene protocol. If we later pivot to a cross-scene extension (Option 2 of the planning chat), those papers become baselines.

---

## Exit criteria for this paper

A Pattern Recognition submission is ready when:

1. Every claim above is supported by a table or figure produced from real benchmark runs.
2. Every figure and table in the manuscript is automatically generated from `outputs/` by `scripts/make_main_tables.py` and `scripts/make_figures.py`.
3. Anonymisation pass complete (no SCATNet codebase, no author names in figure metadata).
4. iThenticate plagiarism score under 15 %.
5. Cover letter and suggested-reviewer list signed off by client.
