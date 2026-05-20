# DR-GSMamba

**Distributionally Robust Graph-State-Space Learning for Label-Scarce Hyperspectral Image Classification**

This project is a full, reproducible starting point for a journal paper on hyperspectral image classification. The method combines a spectral state-space encoder, local spatial CNN features, patch-level graph reasoning, prototype classification, uncertainty-aware training, and a distributionally robust objective aimed at label-scarce and split-sensitive settings.

## Quick Start

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python run_experiments.py --config configs/synthetic.yaml --seeds 1
python scripts/generate_tables.py
python scripts/generate_figures.py
python scripts/generate_classification_maps.py --config configs/synthetic.yaml --metrics outputs/synthetic/seed_0/metrics.json --out-dir paper/figures/maps_synthetic
```

For Indian Pines, this repository is configured to use local dataset files in `data/indian_pines`.

```bash
python run_experiments.py --config configs/indian_pines.yaml
```

To run the reviewer-facing reliability protocol, use:

```bash
python scripts/run_reliability_protocol.py --config configs/indian_pines.yaml
```

For a fast smoke test of the protocol:

```bash
python scripts/run_reliability_protocol.py --config configs/synthetic.yaml --seeds 1
```

Reviewer-facing support scripts:

```bash
python scripts/run_shallow_baselines.py --config configs/indian_pines.yaml --seeds 10
python scripts/run_deep_baselines.py --config configs/indian_pines.yaml --seeds 10 --out-dir outputs/deep_baselines_indian_pines
python scripts/profile_model.py --config configs/indian_pines.yaml
python scripts/compare_runs.py --full outputs/protocol/ablations/full/summary.json --baseline outputs/protocol/ablations/without_dro/summary.json
```

Controlled spectral-backbone ablations:

```bash
python run_experiments.py --config configs/indian_pines.yaml --spectral-backend transformer --output-dir outputs/indian_pines_transformer
python run_experiments.py --config configs/indian_pines.yaml --spectral-backend cnn --output-dir outputs/indian_pines_cnn
```

## Paper-Level Novelty

The intended claim is not only higher overall accuracy. The project is designed to support a stronger journal story:

- accuracy under limited labeled samples;
- lower variance across random splits;
- better rare-class Macro-F1;
- uncertainty estimates for mixed and boundary pixels;
- efficient spectral modeling without quadratic Transformer attention.

## Core Modules

- `models/dr_gsmamba.py`: graph-state-space model with ablation switches for spectral, graph, and prototype branches.
- Spectral branch choices: `mamba` default, legacy `ssm`, `transformer`, and `cnn` for controlled replacement ablations.
- `losses/objectives.py`: cross entropy, sample CVaR, EMA-tracked class-level DRO, prototype compactness, uncertainty weighting, graph smoothness.
- `datasets/hsi_dataset.py`: `.mat` HSI loading, leakage-safe train-fitted PCA, stratified splits, patch extraction.
- `run_experiments.py`: multi-seed reproducibility runner.
- `scripts/generate_tables.py`: exports LaTeX-ready tables.
- `scripts/generate_figures.py`: exports paper figures.
- `scripts/run_reliability_protocol.py`: runs low-label and component-ablation protocols.
- `scripts/run_shallow_baselines.py`: runs SVM, random forest, and kNN shallow baselines.
- `scripts/run_deep_baselines.py`: runs neural baselines including spectral MLP, 2D-CNN, 3D-CNN, HybridSN-style, SpectralFormer-style, SSFTT-style, and nonlocal GCN-style models under the same splits.
- `scripts/profile_model.py`: reports parameter count and inference latency.
- `scripts/compare_runs.py`: compares paired multi-seed runs for significance-style evidence.
- `scripts/generate_classification_maps.py`: exports ground-truth, prediction, error, and uncertainty maps from saved metrics.
- `LITERATURE_TRACKING.md`: tracks recent HSI papers, including updates shared by Minchao Ye, to keep related work and baseline choices current.

## Recommended Experimental Protocol

Run each dataset with at least 10 seeds and report mean, standard deviation, coefficient of variation, and worst split. Compare against 3D-CNN, HybridSN, SpectralFormer/SSFTT, a graph baseline, and one recent Mamba baseline. Include ablations for the spectral branch, spectral-backbone replacements, graph branch, robust loss, prototype head, and uncertainty weighting.

The central validation question is:

> Can graph-state-space modeling with explicit robust optimization improve HSI reliability under label scarcity?

The DRO claim should be retained only if the full model improves rare-class accuracy, worst-class accuracy, and split stability over the same model without the DRO term. Otherwise, the manuscript should weaken the wording and avoid overclaiming distributional robustness. The default spectral branch now uses a dependency-free PyTorch selective-scan implementation inspired by Mamba. It is more faithful than the legacy SSM mixer, but final efficiency claims should still distinguish it from fused CUDA Mamba kernels.

## Reviewer-Risk Checklist

- Novelty: present the method as reliability-oriented robust learning, not as module stacking.
- Related work: connect Transformer, Mamba, graph, and large-model HSI literature directly to the proposed components.
- Comparisons: include recent Transformer, graph, and Mamba baselines where code or fair reimplementation is possible.
- Literature tracking: update `LITERATURE_TRACKING.md` whenever new HSI papers are shared, especially Minchao Ye's updates, and promote important papers into the baseline or related-work list.
- Baseline strength: report shallow baselines plus neural CNN, Transformer-style, graph-style, and Mamba/selective-scan baselines; add recent official Mamba baselines before final submission when practical.
- Qualitative evidence: include classification, error, and uncertainty maps, not only scalar OA/AA/Kappa tables.
- Ablations: validate every named component, especially full model vs. without DRO, plus spectral SSM vs. Transformer vs. CNN replacements.
- Datasets: use Indian Pines, Pavia University, Salinas, and Houston 2013 or WHU-Hi when available.
- Efficiency: report parameters, FLOPs, and runtime against relevant baselines if claiming efficiency.
- Statistics: use paired multi-seed comparisons for the full model against key ablations and baselines.
