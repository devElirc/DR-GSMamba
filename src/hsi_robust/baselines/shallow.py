"""Shallow baselines for the Phase-3 comparison: SVM, RandomForest, kNN.

All three share a uniform interface:

* :class:`ShallowBaselineModel` wraps a scikit-learn classifier and exposes
  ``fit(train_dataset)`` and ``predict(eval_dataset) -> (preds, probs)``.
* :func:`make_shallow_model` is the factory consumed by
  :func:`hsi_robust.baselines.run_shallow_baseline`.

The feature vector is the per-pixel standardised spectrum returned by the
data pipeline (``raw_spectrum``). The spatial patch is discarded by all three
shallow baselines: introducing it does not help SVMs on label-scarce HSI in
the published literature and bloats the kernel matrix beyond what makes
sense at 5 samples/class.

References
----------
* SVM-RBF: Melgani & Bruzzone 2004, "Classification of hyperspectral remote
  sensing images with support vector machines", TGRS.
* Random Forest: Ham et al. 2005, "Investigation of the random forest framework
  for classification of hyperspectral data", TGRS.
* k-NN: Ma et al. 2010, "Local manifold learning-based k-NN for HSI."
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from hsi_robust.data import HSIDataset


def _flatten_spectra(dataset: HSIDataset) -> tuple[np.ndarray, np.ndarray]:
    """Materialise the spectral feature matrix and label vector of one split."""
    feats = np.zeros((len(dataset), 0), dtype=np.float32)
    if len(dataset) == 0:
        return feats, np.zeros((0,), dtype=np.int64)
    samples: list[np.ndarray] = []
    labels: list[int] = []
    for idx in range(len(dataset)):
        spectrum, _patch, label = dataset[idx]
        samples.append(spectrum.detach().cpu().numpy().astype(np.float32))
        labels.append(int(label))
    return np.stack(samples, axis=0), np.asarray(labels, dtype=np.int64)


class ShallowBaselineModel:
    """Thin wrapper over a scikit-learn estimator.

    Parameters
    ----------
    name:
        ``"svm"``, ``"rf"``, or ``"knn"``.
    config:
        Hyperparameters merged into the scikit-learn estimator constructor.
        See :func:`make_shallow_model` for the per-name defaults.
    seed:
        Random seed for the estimator (only ``svm`` ignores this).
    """

    def __init__(self, *, name: str, config: dict[str, Any], seed: int = 0) -> None:
        self.name = name
        self.seed = int(seed)
        self.config = dict(config)
        self.estimator = self._build()

    # ------------------------------------------------------------------ #
    def _build(self):
        if self.name == "svm":
            params = dict(
                C=1.0,
                kernel="rbf",
                gamma="scale",
                probability=True,
                random_state=self.seed,
                cache_size=512,
            )
            params.update(self.config)
            return SVC(**params)
        if self.name == "rf":
            params = dict(
                n_estimators=200,
                max_depth=None,
                min_samples_leaf=1,
                n_jobs=1,
                random_state=self.seed,
            )
            params.update(self.config)
            return RandomForestClassifier(**params)
        if self.name == "knn":
            params = dict(
                n_neighbors=5,
                weights="distance",
                algorithm="auto",
                n_jobs=1,
            )
            params.update(self.config)
            # k-NN has no random state -- ``seed`` is intentionally unused.
            return KNeighborsClassifier(**params)
        raise ValueError(f"unknown shallow baseline '{self.name}'")

    # ------------------------------------------------------------------ #
    def fit(self, train_dataset: HSIDataset) -> ShallowBaselineModel:
        x, y = _flatten_spectra(train_dataset)
        if x.shape[0] == 0:
            raise ValueError("training set is empty")
        self.estimator.fit(x, y)
        return self

    def predict(self, eval_dataset: HSIDataset) -> tuple[np.ndarray, np.ndarray]:
        x, _ = _flatten_spectra(eval_dataset)
        preds = self.estimator.predict(x).astype(np.int64)
        if hasattr(self.estimator, "predict_proba"):
            probs = self.estimator.predict_proba(x).astype(np.float32)
        else:
            # Fall back to one-hot when the estimator has no probability output.
            num_classes = int(self.estimator.classes_.max() + 1)
            probs = np.eye(num_classes, dtype=np.float32)[preds]
        return preds, probs


def make_shallow_model(
    name: str, config: dict[str, Any] | None = None, *, seed: int = 0
) -> ShallowBaselineModel:
    """Construct a :class:`ShallowBaselineModel` for ``name``."""
    return ShallowBaselineModel(name=name, config=dict(config or {}), seed=seed)


def shallow_baseline_names() -> tuple[str, ...]:
    return ("svm", "rf", "knn")


# A helper torch wrapper so the shallow baselines integrate with the same
# ``BaselineTrainer.evaluate_predictions`` interface as the deep baselines
# at the script level.
def predictions_to_torch(preds: np.ndarray, probs: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.from_numpy(preds), torch.from_numpy(probs)
