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

## Recommended Experimental Protocol

Run each dataset with at least 10 seeds and report mean, standard deviation, coefficient of variation, and worst split. Compare against 3D-CNN, HybridSN, SpectralFormer/SSFTT, a graph baseline, and one recent Mamba baseline. Include ablations for the spectral branch, graph branch, robust loss, prototype head, and uncertainty weighting.
