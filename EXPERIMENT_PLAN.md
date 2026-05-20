# DR-GSMamba Experiment Plan

This plan turns the client feedback and prior TGRS reviewer comments into concrete checks.

## Core Question

Can graph-state-space modeling with explicit robust optimization improve the reliability of hyperspectral image classification under label scarcity?

## Claims That Must Be Proven

1. Distributionally robust learning is not a title-only claim.
   - Use sample CVaR plus class-level worst-risk optimization.
   - Validate full DR-GSMamba against the same model without the DRO term.
   - Keep the claim only if rare-class accuracy, worst-class accuracy, and split stability improve.

2. Graph-state-space modeling has a clear role.
   - Ablate the spectral state-space branch.
   - Ablate the graph branch.
   - Run Transformer and 1D-CNN spectral replacement variants under the same graph/prototype/DRO framework.

3. Prototype and uncertainty modules support reliability.
   - Ablate prototype learning.
   - Ablate uncertainty weighting.
   - Report calibration metrics if uncertainty is discussed in the paper.

## Required Reporting

- OA, AA, Kappa
- Macro-F1
- Per-class accuracy and per-class F1
- Worst-class accuracy
- Rare-class accuracy
- Mean and standard deviation across random splits
- Coefficient of variation
- Worst-split performance
- Calibration error when uncertainty is included
- Parameter count and latency when efficiency is claimed
- Paired multi-seed comparison against key ablations and baselines

## Dataset Scope

Minimum target:

- Indian Pines
- Pavia University
- Salinas
- Houston 2013 or WHU-Hi

Avoid relying only on two small saturated datasets.

## Baseline Groups

- Classical or shallow baselines
- CNN and Transformer HSI baselines
- Graph and Mamba HSI baselines

Recent Transformer/Mamba methods should be included when official code or fair reimplementation is possible.

The repository includes a shallow baseline runner for SVM, random forest, and kNN. It also includes a neural baseline runner for spectral MLP, 2D-CNN, compact 3D-CNN, HybridSN-style, SpectralFormer-style, SSFTT-style, nonlocal GCN-style, and Mamba/selective-scan spectral models. These strengthen the code package, but final TGRS/Pattern Recognition experiments should still include official implementations where available, especially for the most recent Mamba baselines.

## Qualitative Evidence

Generate classification maps, error maps, and uncertainty maps for every real dataset. Scalar metrics alone are not enough for a strong journal submission because reviewers often expect visual inspection of boundary regions, rare classes, and spatial consistency.

## Literature Monitoring

Keep `LITERATURE_TRACKING.md` updated with recent HSI classification papers, including the HSI-related updates shared by Minchao Ye when available. This tracking should directly inform:

- related-work coverage;
- final baseline selection;
- novelty-risk checks against recent Mamba, graph, Transformer, diffusion, uncertainty, domain-adaptation, and foundation-style HSI methods;
- reviewer-facing justification for why each recent method is included as a baseline, cited only as related work, or excluded due to unavailable code or unfair protocol mismatch.

The current high-priority Minchao Ye update batch includes recent TGRS work on diffusion-enhanced uncertainty attention, meta-learning with feature alignment, adaptive graph modeling with self-training, discriminative Vision Transformers, and Multi-CycleGAN cross-domain mapping. Before final benchmark runs, freeze the baseline list and verify that no important recent HSI method from the tracked updates is silently ignored.

Local PDF review added three important cross-domain/cross-scene references: CD-MFA, CDWOASA, and CDIRF. For the current single-scene label-scarce protocol, these should be cited as related work and novelty-boundary references rather than direct baselines. If the project is extended to cross-scene or heterogeneous transfer, CD-MFA becomes a high-priority deep baseline, while CDWOASA and CDIRF become classical feature-selection baselines.

## Commands

Fast smoke protocol:

```bash
python scripts/run_reliability_protocol.py --config configs/synthetic.yaml --seeds 1
```

Real dataset protocol:

```bash
python scripts/run_reliability_protocol.py --config configs/indian_pines.yaml
python scripts/run_reliability_protocol.py --config configs/pavia_university.yaml
python scripts/run_reliability_protocol.py --config configs/salinas.yaml
```

Houston 2013 requires confirming local file names and `.mat` keys before a full run.

Shallow baselines:

```bash
python scripts/run_shallow_baselines.py --config configs/indian_pines.yaml --seeds 10
```

Deep neural baselines:

```bash
python scripts/run_deep_baselines.py --config configs/indian_pines.yaml --seeds 10 --out-dir outputs/deep_baselines_indian_pines
```

Classification, error, and uncertainty maps:

```bash
python scripts/generate_classification_maps.py --config configs/indian_pines.yaml --metrics outputs/indian_pines/seed_0/metrics.json --out-dir paper/figures/maps_indian_pines
```

Efficiency evidence:

```bash
python scripts/profile_model.py --config configs/indian_pines.yaml
```

Full model vs. without-DRO comparison:

```bash
python scripts/compare_runs.py --full outputs/protocol/ablations/full/summary.json --baseline outputs/protocol/ablations/without_dro/summary.json
```

Spectral-backbone replacement checks:

```bash
python run_experiments.py --config configs/indian_pines.yaml --spectral-backend transformer --output-dir outputs/indian_pines_transformer
python run_experiments.py --config configs/indian_pines.yaml --spectral-backend cnn --output-dir outputs/indian_pines_cnn
```
