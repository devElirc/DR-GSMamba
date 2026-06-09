"""Raw cube + ground-truth loaders for HSI datasets.

Supported file formats:
    * ``.mat`` (MATLAB v5 / v7), loaded with :func:`scipy.io.loadmat`.
    * ``.npy``, loaded with :func:`numpy.load`.

The loader takes a dataset config (see ``configs/datasets/*.yaml``) and returns
a tuple ``(cube, gt)`` with:

    cube : float32 array of shape ``(H, W, B)``
    gt   : int64 array of shape ``(H, W)`` -- raw scene labels, including the
           ignore label (typically ``0``).

Shape and dtype sanity checks are performed against the config so that a typo
in the YAML is caught at load time rather than several functions later.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import loadmat


def _read_mat_or_npy(path: Path, key: str | None) -> np.ndarray:
    """Read a single array from a ``.mat`` or ``.npy`` file."""
    suffix = path.suffix.lower()
    if suffix == ".mat":
        if key is None:
            raise ValueError(f"loading {path}: a `key` is required for .mat files")
        bundle = loadmat(str(path))
        if key not in bundle:
            available = [k for k in bundle if not k.startswith("__")]
            raise KeyError(f"key '{key}' not found in {path}; available keys: {available}")
        return np.asarray(bundle[key])
    if suffix == ".npy":
        return np.load(path)
    raise ValueError(f"unsupported file format '{suffix}' for {path}")


def load_hsi_cube(config: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Load the HSI cube and ground truth named by ``config``.

    Parameters
    ----------
    config:
        A dataset config (e.g. parsed from ``configs/datasets/indian_pines.yaml``)
        with the following required keys:

        * ``data_dir`` (str)            : directory containing the cube and gt files
        * ``cube_file`` (str)           : filename of the cube file inside ``data_dir``
        * ``gt_file`` (str)             : filename of the ground-truth file
        * ``cube_key`` (str, optional)  : .mat field name for the cube (required if .mat)
        * ``gt_key`` (str, optional)    : .mat field name for the gt (required if .mat)
        * ``num_bands`` (int)           : expected number of spectral bands
        * ``num_classes`` (int)         : number of *labelled* classes (excludes ignore)
        * ``spatial_size`` (list[int])  : ``[H, W]`` expected spatial shape
        * ``ignore_label`` (int)        : label value that means "no class"

    Returns
    -------
    (cube, gt):
        cube has dtype ``float32`` and shape ``(H, W, num_bands)``;
        gt has dtype ``int64`` and shape ``(H, W)``. The gt values are *not*
        relabelled here -- ``flatten_labeled_pixels`` handles 1-indexed to
        0-indexed remapping further downstream.
    """
    data_dir = Path(config["data_dir"])
    cube_path = data_dir / config["cube_file"]
    gt_path = data_dir / config["gt_file"]

    cube = _read_mat_or_npy(cube_path, config.get("cube_key")).astype(np.float32)
    gt = _read_mat_or_npy(gt_path, config.get("gt_key")).astype(np.int64)

    # Shape sanity checks against the config.
    if cube.ndim != 3:
        raise ValueError(f"expected 3D cube (H, W, B); got shape {cube.shape}")
    if gt.ndim != 2:
        raise ValueError(f"expected 2D gt (H, W); got shape {gt.shape}")
    expected_h, expected_w = config["spatial_size"]
    if (cube.shape[0], cube.shape[1]) != (expected_h, expected_w):
        raise ValueError(
            f"cube spatial shape {cube.shape[:2]} does not match config "
            f"spatial_size {config['spatial_size']}"
        )
    if gt.shape != (expected_h, expected_w):
        raise ValueError(
            f"gt spatial shape {gt.shape} does not match config spatial_size "
            f"{config['spatial_size']}"
        )
    if cube.shape[2] != config["num_bands"]:
        raise ValueError(
            f"cube has {cube.shape[2]} bands but config['num_bands'] = {config['num_bands']}"
        )

    # Label-set sanity: every non-ignore value must be in [1, num_classes] for the
    # standard 1-indexed scene convention or in [0, num_classes - 1] for a 0-indexed
    # convention. We accept either by inspection.
    unique = sorted(np.unique(gt).tolist())
    ignore = int(config["ignore_label"])
    non_ignore = [v for v in unique if v != ignore]
    k = int(config["num_classes"])
    if not non_ignore:
        raise ValueError(f"gt has no labelled pixels (only ignore={ignore})")
    if min(non_ignore) >= 1 and max(non_ignore) <= k:
        # 1-indexed convention (the usual HSI .mat scheme).
        pass
    elif min(non_ignore) >= 0 and max(non_ignore) <= k - 1 and ignore != 0:
        # 0-indexed convention; only valid if the ignore label is something else.
        pass
    else:
        raise ValueError(
            f"gt has unexpected label values {non_ignore}; expected either "
            f"[1, {k}] (1-indexed) or [0, {k - 1}] (0-indexed) with ignore={ignore}"
        )

    return cube, gt
