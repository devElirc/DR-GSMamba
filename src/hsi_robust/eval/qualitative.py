"""Qualitative outputs: classification, error, vacuity, and aleatoric maps.

Each function returns a NumPy array shaped like the original ground-truth map
``(H, W)``, with ``NaN`` (for float arrays) or ``-1`` (for the integer
classification map) at positions whose label was 0 in the original ground
truth (unlabelled pixels).

These are *raw arrays* -- no plotting. A downstream notebook can use any
colormap (e.g. ``matplotlib``) to render the maps.
"""

from __future__ import annotations

import numpy as np


def _scatter_into_map(
    flat_values: np.ndarray,
    flat_pixel_coords: np.ndarray,
    spatial_shape: tuple[int, int],
    fill: float | int = float("nan"),
    dtype: np.dtype | type = np.float32,
) -> np.ndarray:
    h, w = spatial_shape
    out = np.full((h, w), fill_value=fill, dtype=dtype)
    rows = flat_pixel_coords[:, 0]
    cols = flat_pixel_coords[:, 1]
    out[rows, cols] = flat_values
    return out


def classification_map(
    pred: np.ndarray,
    pixel_coords: np.ndarray,
    spatial_shape: tuple[int, int],
) -> np.ndarray:
    """Classification map (-1 at unlabelled background)."""
    return _scatter_into_map(
        pred.astype(np.int64), pixel_coords, spatial_shape, fill=-1, dtype=np.int64
    )


def error_map(
    pred: np.ndarray,
    target: np.ndarray,
    pixel_coords: np.ndarray,
    spatial_shape: tuple[int, int],
) -> np.ndarray:
    """Binary error map (1 = incorrect, 0 = correct, NaN = background)."""
    err = (pred != target).astype(np.float32)
    return _scatter_into_map(err, pixel_coords, spatial_shape, fill=float("nan"), dtype=np.float32)


def vacuity_map(
    vacuity: np.ndarray,
    pixel_coords: np.ndarray,
    spatial_shape: tuple[int, int],
) -> np.ndarray:
    """Vacuity ``K/S`` map (NaN at background)."""
    return _scatter_into_map(
        vacuity.astype(np.float32),
        pixel_coords,
        spatial_shape,
        fill=float("nan"),
        dtype=np.float32,
    )


def aleatoric_map(
    aleatoric: np.ndarray,
    pixel_coords: np.ndarray,
    spatial_shape: tuple[int, int],
) -> np.ndarray:
    """Aleatoric ``sum_k p_k(1-p_k)`` map (NaN at background)."""
    return _scatter_into_map(
        aleatoric.astype(np.float32),
        pixel_coords,
        spatial_shape,
        fill=float("nan"),
        dtype=np.float32,
    )
