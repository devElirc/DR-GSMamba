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
```

For Indian Pines, this repository is already configured to reuse the dataset files found in `../SCATNet_Codebase_update/data/indian_pines`.

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
python scripts/profile_model.py --config configs/indian_pines.yaml
python scripts/compare_runs.py --full outputs/protocol/ablations/full/summary.json --baseline outputs/protocol/ablations/without_dro/summary.json
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
- `losses/objectives.py`: cross entropy, sample CVaR, class-level DRO, prototype compactness, uncertainty weighting, graph smoothness.
- `datasets/hsi_dataset.py`: `.mat` HSI loading, PCA, stratified splits, patch extraction.
- `run_experiments.py`: multi-seed reproducibility runner.
- `scripts/generate_tables.py`: exports LaTeX-ready tables.
- `scripts/generate_figures.py`: exports paper figures.
- `scripts/run_reliability_protocol.py`: runs low-label and component-ablation protocols.
- `scripts/run_shallow_baselines.py`: runs SVM, random forest, and kNN shallow baselines.
- `scripts/profile_model.py`: reports parameter count and inference latency.
- `scripts/compare_runs.py`: compares paired multi-seed runs for significance-style evidence.

## Recommended Experimental Protocol

Run each dataset with at least 10 seeds and report mean, standard deviation, coefficient of variation, and worst split. Compare against 3D-CNN, HybridSN, SpectralFormer/SSFTT, a graph baseline, and one recent Mamba baseline. Include ablations for the spectral branch, graph branch, robust loss, prototype head, and uncertainty weighting.

The central validation question is:

> Can graph-state-space modeling with explicit robust optimization improve HSI reliability under label scarcity?

The DRO claim should be retained only if the full model improves rare-class accuracy, worst-class accuracy, and split stability over the same model without the DRO term. Otherwise, the manuscript should weaken the wording and avoid overclaiming distributional robustness.

## Reviewer-Risk Checklist

- Novelty: present the method as reliability-oriented robust learning, not as module stacking.
- Related work: connect Transformer, Mamba, graph, and large-model HSI literature directly to the proposed components.
- Comparisons: include recent Transformer, graph, and Mamba baselines where code or fair reimplementation is possible.
- Ablations: validate every named component, especially full model vs. without DRO.
- Datasets: use Indian Pines, Pavia University, Salinas, and Houston 2013 or WHU-Hi when available.
- Efficiency: report parameters, FLOPs, and runtime against relevant baselines if claiming efficiency.
- Statistics: use paired multi-seed comparisons for the full model against key ablations and baselines.
