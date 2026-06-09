"""Stratified fixed-samples-per-class sampler.

Produces a train / test split such that the training set has exactly
``samples_per_class`` pixels per class whenever the scene contains enough
labelled pixels for that class. For classes with fewer labelled pixels than
``samples_per_class`` (e.g. class 9 of Indian Pines at ``samples_per_class=20``),
all available pixels go to training and the class has zero test pixels; a
``UserWarning`` is emitted because the class then cannot be evaluated.

Determinism contract:
    * Same ``seed`` and same ``labels_flat`` --> identical ``train_idx`` and
      ``test_idx`` arrays (bitwise identical, including their order).
    * Different ``seed`` --> different draw inside each class, but the
      per-class counts are still exactly the same.

The RNG used is :func:`numpy.random.default_rng`, which has a fixed bit
generator (PCG64) across NumPy versions.
"""

from __future__ import annotations

import warnings

import numpy as np


def stratified_fixed_per_class_split(
    labels_flat: np.ndarray,
    *,
    samples_per_class: int,
    seed: int,
    num_classes: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Stratified fixed-samples-per-class split.

    Parameters
    ----------
    labels_flat:
        1D 0-indexed labels of shape ``(N_labelled,)``; output of
        :func:`hsi_robust.data.scene_freq.flatten_labeled_pixels`.
    samples_per_class:
        Number of pixels to draw per class for training (3, 5, 10, or 20 in our
        experiments).
    seed:
        Integer seed passed to :func:`numpy.random.default_rng`. Same seed +
        same ``labels_flat`` produces identical splits.
    num_classes:
        Number of classes ``K``. If ``None``, inferred from ``labels_flat.max() + 1``.

    Returns
    -------
    (train_idx, test_idx):
        Both arrays are sorted by class index, then by intra-class shuffled
        order, with dtype ``int64``. Indices reference into ``labels_flat``
        (and thus into the position array returned by ``flatten_labeled_pixels``).

    Raises
    ------
    ValueError:
        If ``samples_per_class`` is non-positive.
    """
    if samples_per_class <= 0:
        raise ValueError(f"samples_per_class must be > 0; got {samples_per_class}")

    labels_flat = np.ascontiguousarray(labels_flat).astype(np.int64, copy=False)
    if labels_flat.ndim != 1:
        raise ValueError(f"expected 1D labels; got shape {labels_flat.shape}")

    if num_classes is None:
        num_classes = int(labels_flat.max()) + 1
    if num_classes <= 0:
        raise ValueError(f"num_classes must be > 0; got {num_classes}")

    rng = np.random.default_rng(seed)

    train_chunks: list[np.ndarray] = []
    test_chunks: list[np.ndarray] = []
    starved_classes: list[tuple[int, int]] = []

    for k in range(num_classes):
        class_idx = np.where(labels_flat == k)[0]
        if class_idx.size == 0:
            # Empty classes are caught by compute_scene_freq; for the sampler we
            # silently skip so this function can be called on tiny synthetic data.
            continue
        # Permute in place under the per-call RNG.
        permuted = rng.permutation(class_idx)
        if permuted.size <= samples_per_class:
            train_chunks.append(permuted)
            starved_classes.append((k, int(permuted.size)))
        else:
            train_chunks.append(permuted[:samples_per_class])
            test_chunks.append(permuted[samples_per_class:])

    if starved_classes:
        msg = ", ".join(f"class {k} (n={n})" for k, n in starved_classes)
        warnings.warn(
            f"samples_per_class={samples_per_class} >= scene count for "
            f"{len(starved_classes)} class(es) [{msg}]; these classes have 0 test "
            "samples and cannot be evaluated.",
            UserWarning,
            stacklevel=2,
        )

    train_idx = (
        np.concatenate(train_chunks).astype(np.int64)
        if train_chunks
        else np.empty(0, dtype=np.int64)
    )
    test_idx = (
        np.concatenate(test_chunks).astype(np.int64)
        if test_chunks
        else np.empty(0, dtype=np.int64)
    )
    return train_idx, test_idx
