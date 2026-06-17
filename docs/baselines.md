# Phase 3 baseline catalogue

This document is the single source of truth for the baseline lineup used in
Phase 6 main benchmarks and Phase 7 ablations. Every entry below has:

* a **citation** so reviewers can verify the architecture is faithful;
* a **module path** under `src/hsi_robust/baselines/` so the code can be
  audited;
* a **reproduction-gap field** that will be filled with the Indian-Pines
  best-OA we obtain under the Phase 5 sanity run vs the number the original
  paper reports under matching conditions.

All deep baselines share the same data pipeline as the main model
(`hsi_robust.data.build_split`) and the same training loop
(`hsi_robust.baselines.BaselineTrainer`, plain CE). This ensures
head-to-head numbers in Phase 6 are not contaminated by data or optimiser
drift.

## Shallow baselines (M3.1)

| Name | Reference | Module | Hyperparameters |
| --- | --- | --- | --- |
| `svm`  | Melgani & Bruzzone (2004), *TGRS* | `baselines/shallow.py` | RBF, `C=1`, `gamma=scale` |
| `rf`   | Ham et al. (2005), *TGRS* | `baselines/shallow.py` | 200 trees, `min_samples_leaf=1` |
| `knn`  | Ma et al. (2010) | `baselines/shallow.py` | `k=5`, distance weighting |

All three consume the per-band-standardised raw spectrum (B-dim vector). The
PCA patch is intentionally ignored — sparse local windows do not help shallow
classifiers at 5 samples/class.

### Reproduction gap (filled in Phase 5)

| Dataset | Setting | Paper OA | Ours OA | Gap |
| --- | --- | --- | --- | --- |
| Indian Pines | 5-spc, seed 0 | n/a in original | 38.5 % (`svm`), 43.3 % (`rf`), 37.3 % (`knn`) | n/a |
| Indian Pines | 5-spc, 3-seed mean | — | — | — |
| Pavia U | 5-spc, 3-seed mean | — | — | — |

The original papers did not all benchmark at 5-spc, so the gap is recorded
"n/a" where no comparable number exists. The single-seed smoke above is the
seed-0 run on the real cube; Phase 5 fills the rest.

## Deep baselines (M3.2 – M3.5)

| Name | Reference | Module | Headline numbers (paper) |
| --- | --- | --- | --- |
| `cnn3d`         | Chen et al. (2016) *TGRS*   | `baselines/cnn3d.py`         | 99.07 % OA on IP, 200 spc |
| `hybridsn`      | Roy et al. (2020) *GRSL*    | `baselines/hybridsn.py`      | 99.75 % OA on IP, 30 % train |
| `spectralformer`| Hong et al. (2022) *TGRS*   | `baselines/spectralformer.py`| 81.76 % OA on IP, fixed 10-spc-train |
| `ssftt`         | Sun et al. (2022) *TGRS*    | `baselines/ssftt.py`         | 99.32 % OA on IP, 10 % train |
| `nonlocal_gcn`  | Wan et al. (2020) *TGRS*    | `baselines/nonlocal_gcn.py`  | 95.84 % OA on IP, 100-spc |
| `mambahsi`      | Huang et al. (2024) *TGRS*  | `baselines/mambahsi.py`      | 98.41 % OA on IP, 30-spc |

These numbers are **not** the regime we evaluate. The main paper runs on the
label-scarce regime (3 / 5 / 10 / 20 samples per class, 10 seeds) so the
expected baseline OA will be substantially lower than the headline figures.
The relevant comparison is "baseline at 5-spc vs DR-GSMamba at 5-spc", not
baseline-paper-headline.

### Implementation notes per baseline

* **CNN3D** — three 3D conv blocks (8 → 16 → 32 channels, `(3, 3, 3)` kernels)
  with `GroupNorm` (decision D-09: GN beats BN at 5-spc), adaptive 3D global
  pool, dropout, linear head.
* **HybridSN** — exact published 3D → 2D → MLP hierarchy with kernels
  `(7, 3, 3) → (5, 3, 3) → (3, 3, 3) → 3×3 → FC(256) → FC(128) → classifier`,
  GroupNorm in place of BN.
* **SpectralFormer** — band-group tokeniser (`group_size=7`), sinusoidal
  positional encoding, `depth=4` Transformer with `dim=64`, `heads=4`,
  `mlp_ratio=2`. Classifier on the `[CLS]` token.
* **SSFTT** — 3D-conv → 2D-conv tokeniser, learnable Gaussian-weighted
  semantic tokeniser (`num_tokens=4`), single Transformer block with
  `heads=4`.
* **NonlocalGCN** — flatten patch into `(N, P*P, C_pca)` tokens, `theta-phi-g`
  affinity attention with softmax normalisation, two layers, classifier on
  the central token.
* **MambaHSI** — reuses the in-house **OP-S4** selective-scan block for the
  spectral branch (faithful Mamba-style SSM with band gating) plus the
  same spatial CNN stem as the main model, concat + MLP fusion.

### Reproduction gap (filled in Phase 5)

| Baseline | Dataset | Setting | Paper OA | Ours OA | Gap |
| --- | --- | --- | --- | --- | --- |
| `cnn3d` | Indian Pines | 200-spc | 99.07 % | — | — |
| `hybridsn` | Indian Pines | 30 % train | 99.75 % | — | — |
| `spectralformer` | Indian Pines | 10-spc fixed | 81.76 % | — | — |
| `ssftt` | Indian Pines | 10 % train | 99.32 % | — | — |
| `nonlocal_gcn` | Indian Pines | 100-spc | 95.84 % | — | — |
| `mambahsi` | Indian Pines | 30-spc | 98.41 % | — | — |

The Phase 3 exit criterion (roadmap M3.6) is "match the paper within ± 2 %
OA *or* record the reproduction gap." Where the published setting is
sample-rich (e.g. 30 % train), we record the gap and rely on the 5-spc /
10-spc head-to-head in Phase 6 as the comparison-relevant number.

## Dispatch script

`scripts/run_baselines.py` runs any subset of the above on any dataset at
any seed / `samples_per_class`:

```bash
# Every baseline on Indian Pines, 5 samples/class, seed 0 (CPU).
python scripts/run_baselines.py --dataset indian_pines --baselines all

# A single deep baseline at higher capacity (custom epochs).
python scripts/run_baselines.py --dataset indian_pines --baselines hybridsn \
    --seed 1 --samples-per-class 10 --epochs 200 --device cuda

# Just the shallow trio (no GPU needed).
python scripts/run_baselines.py --dataset indian_pines --baselines shallow
```

Output: `outputs/<dataset>_<baseline>_seed<seed>/final.json` per baseline
plus a top-level `outputs/baselines_summary_<dataset>_seed<seed>_spc<k>.json`.
