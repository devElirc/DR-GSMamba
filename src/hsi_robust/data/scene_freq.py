"""Scene-level class-frequency :math:`\\pi_k` and ground-truth flattening.

This module is the implementation contract of ``docs/math/cfa_gdro.md`` Section 1.
The source-of-truth definition is:

    pi_k = N_k^scene / sum_j N_j^scene

where ``N_k^scene`` counts labelled pixels of class ``k`` *in the full ground-truth
map*, including labelled-but-unused pixels. The "ignore" label (typically ``0``)
is excluded.

Because we run a fixed-samples-per-class protocol, the training-set frequencies
are uniform by construction and would make :math:`\\gamma` vanish from the CFA-GDRO
math; the scene-level definition above is the correct quantity.
"""

from __future__ import annotations

import numpy as np
import torch


def flatten_labeled_pixels(
    gt: np.ndarray,
    *,
    num_classes: int,
    ignore_label: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Flatten the ground-truth map into a 1D label array + (i, j) positions.

    The output label array is **0-indexed** in the range ``[0, num_classes - 1]``.
    If the input ``gt`` uses the 1-indexed HSI convention (labels in
    ``{1, ..., num_classes}`` with ``0`` as ignore), the function subtracts 1 from
    every non-ignore value. If ``gt`` already uses a 0-indexed convention (labels
    in ``{0, ..., num_classes - 1}`` with ``ignore_label != 0``), the labels pass
    through unchanged.

    Parameters
    ----------
    gt:
        Integer array of shape ``(H, W)`` with raw scene labels.
    num_classes:
        Number of labelled classes ``K``.
    ignore_label:
        Label value that marks "no class". Defaults to ``0``.

    Returns
    -------
    (labels_flat, positions):
        labels_flat has shape ``(N_labelled,)`` with dtype ``int64`` and values
        in ``[0, num_classes - 1]``;
        positions has shape ``(N_labelled, 2)`` with dtype ``int64`` and
        ``(i, j)`` row-major coordinates.
    """
    if gt.ndim != 2:
        raise ValueError(f"expected 2D gt; got shape {gt.shape}")

    mask = gt != ignore_label
    pos_i, pos_j = np.where(mask)
    positions = np.stack([pos_i, pos_j], axis=1).astype(np.int64)
    labels = gt[mask].astype(np.int64)

    unique = np.unique(labels)
    if unique.size == 0:
        raise ValueError(
            f"no labelled pixels found (every pixel has gt == {ignore_label})"
        )
    if unique.min() >= 1 and unique.max() <= num_classes:
        # 1-indexed scene convention; remap to 0-indexed.
        labels = labels - 1
    elif unique.min() >= 0 and unique.max() <= num_classes - 1:
        # 0-indexed convention; nothing to do.
        pass
    else:
        raise ValueError(
            f"gt labels {sorted(unique.tolist())} do not fit either "
            f"[1, {num_classes}] (1-indexed) or [0, {num_classes - 1}] (0-indexed)"
        )

    return labels, positions


def compute_scene_freq(
    labels_flat: np.ndarray | torch.Tensor,
    num_classes: int,
) -> torch.Tensor:
    """Scene-level class frequencies :math:`\\pi_k` for CFA-GDRO.

    Parameters
    ----------
    labels_flat:
        1D array of all labelled pixels in the scene, 0-indexed. This is the
        output of :func:`flatten_labeled_pixels`.
    num_classes:
        Number of classes ``K``. Required so the output is ``(K,)`` even when
        some class is empty in the scene (a degenerate but checked case below).

    Returns
    -------
    pi:
        ``torch.float64`` tensor of shape ``(K,)``, summing to 1.

    Raises
    ------
    ValueError:
        If a class has zero labelled pixels in the scene (would make
        :math:`\\pi_k^{-\\gamma}` infinite for that class) or if the labels are
        outside ``[0, K - 1]``.
    """
    if isinstance(labels_flat, torch.Tensor):
        labels_np = labels_flat.detach().cpu().numpy()
    else:
        labels_np = np.asarray(labels_flat)
    labels_np = labels_np.astype(np.int64, copy=False)

    if labels_np.ndim != 1:
        raise ValueError(f"expected 1D labels; got shape {labels_np.shape}")
    if labels_np.size == 0:
        raise ValueError("labels_flat is empty")
    lo, hi = int(labels_np.min()), int(labels_np.max())
    if lo < 0 or hi >= num_classes:
        raise ValueError(
            f"labels out of range [0, {num_classes - 1}]: min={lo}, max={hi}"
        )

    counts = np.bincount(labels_np, minlength=num_classes).astype(np.float64)
    empty_classes = np.where(counts == 0)[0]
    if empty_classes.size > 0:
        raise ValueError(
            f"classes {empty_classes.tolist()} have zero labelled pixels in the "
            "scene; CFA-GDRO requires pi_k > 0 for every class"
        )

    total = float(counts.sum())
    pi = counts / total
    return torch.from_numpy(pi)
