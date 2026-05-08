from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, f1_score


def expected_calibration_error(y_true, y_prob, bins: int = 10) -> float:
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    if y_prob.size == 0:
        return 0.0
    confidence = y_prob.max(axis=1)
    prediction = y_prob.argmax(axis=1)
    correct = (prediction == y_true).astype(float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (confidence > lo) & (confidence <= hi)
        if mask.any():
            ece += mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())
    return float(ece)


def classification_metrics(y_true, y_pred, num_classes: int, y_prob=None) -> dict:
    labels = list(range(num_classes))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    support = cm.sum(axis=1)
    per_class = cm.diagonal() / np.maximum(cm.sum(axis=1), 1)
    per_class_f1 = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    rare_mask = support <= np.median(support[support > 0]) if np.any(support > 0) else np.ones_like(support, dtype=bool)
    metrics = {
        "oa": float(accuracy_score(y_true, y_pred)),
        "aa": float(np.mean(per_class)),
        "kappa": float(cohen_kappa_score(y_true, y_pred, labels=labels)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "worst_class_accuracy": float(np.min(per_class)),
        "rare_class_accuracy": float(np.mean(per_class[rare_mask])) if rare_mask.any() else float(np.mean(per_class)),
        "per_class_accuracy": per_class.tolist(),
        "per_class_f1": per_class_f1.tolist(),
        "class_support": support.astype(int).tolist(),
        "confusion_matrix": cm.tolist(),
    }
    if y_prob is not None:
        metrics["ece"] = expected_calibration_error(y_true, y_prob)
    return metrics
