from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import scipy.io as sio
import torch
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader, Dataset


@dataclass
class HSIData:
    cube: np.ndarray
    labels: np.ndarray
    num_classes: int


def _first_mat_array(path: Path) -> np.ndarray:
    mat = sio.loadmat(path)
    for key, value in mat.items():
        if not key.startswith("__") and isinstance(value, np.ndarray):
            return value
    raise ValueError(f"No ndarray found in {path}")


def load_hsi(cfg: Dict, seed: int) -> HSIData:
    dataset = cfg["dataset"]
    if dataset["name"] == "synthetic":
        return make_synthetic_hsi(dataset["synthetic"], seed)

    root = Path(dataset["root"])
    data_path = root / dataset["data_file"]
    gt_path = root / dataset["gt_file"]
    data_mat = sio.loadmat(data_path)
    gt_mat = sio.loadmat(gt_path)
    cube = data_mat.get(dataset.get("data_key")) if dataset.get("data_key") else None
    labels = gt_mat.get(dataset.get("gt_key")) if dataset.get("gt_key") else None
    cube = cube if cube is not None else _first_mat_array(data_path)
    labels = labels if labels is not None else _first_mat_array(gt_path)
    labels = labels.astype(np.int64)
    num_classes = int(labels.max())
    return HSIData(cube=cube.astype(np.float32), labels=labels, num_classes=num_classes)


def make_synthetic_hsi(params: Dict, seed: int) -> HSIData:
    rng = np.random.default_rng(seed)
    h, w, bands, classes = params["height"], params["width"], params["bands"], params["classes"]
    yy, xx = np.mgrid[:h, :w]
    labels = np.zeros((h, w), dtype=np.int64)
    centers = rng.uniform([10, 10], [h - 10, w - 10], size=(classes, 2))
    for idx, (cy, cx) in enumerate(centers, start=1):
        radius = rng.uniform(9, 18)
        mask = (yy - cy) ** 2 + (xx - cx) ** 2 <= radius**2
        labels[mask] = idx
    labels[labels == 0] = rng.integers(1, classes + 1, size=(labels == 0).sum())

    wavelengths = np.linspace(0, 1, bands)
    signatures = []
    for cls in range(classes):
        peaks = rng.uniform(0.1, 0.9, size=3)
        width = rng.uniform(0.03, 0.12)
        sig = sum(np.exp(-((wavelengths - p) ** 2) / width) for p in peaks)
        signatures.append(sig / sig.max())
    signatures = np.asarray(signatures, dtype=np.float32)

    cube = signatures[labels - 1]
    cube += 0.08 * rng.normal(size=cube.shape).astype(np.float32)
    cube += 0.15 * np.sin(xx[..., None] / 9.0) * np.cos(yy[..., None] / 11.0)
    return HSIData(cube=cube.astype(np.float32), labels=labels, num_classes=classes)


def normalize_and_reduce(cube: np.ndarray, fit_indices: np.ndarray, components: int, seed: int) -> np.ndarray:
    """Normalize and reduce an HSI cube using training pixels only.

    Fitting PCA on every labeled pixel leaks validation/test distribution into
    preprocessing. For reviewer-facing experiments, the train split must define
    the scaler and PCA projection, then the learned transform is applied to all
    pixels.
    """
    h, w, c = cube.shape
    flat = cube.reshape(-1, c)
    train_flat = flat[fit_indices]
    mean = train_flat.mean(axis=0, keepdims=True)
    std = train_flat.std(axis=0, keepdims=True) + 1e-6
    flat = (flat - mean) / std
    if components and components < c:
        actual_components = min(int(components), c, max(1, len(fit_indices)))
        pca = PCA(n_components=actual_components, random_state=seed, whiten=False)
        pca.fit(flat[fit_indices])
        flat = pca.transform(flat)
    return flat.reshape(h, w, -1).astype(np.float32)


def stratified_indices(labels: np.ndarray, cfg: Dict, seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    y = labels.reshape(-1)
    valid = np.where(y > 0)[0]
    classes = np.unique(y[valid])
    train, val, test = [], [], []
    rng = np.random.default_rng(seed)
    samples_per_class = cfg["dataset"].get("samples_per_class")
    for cls in classes:
        cls_idx = valid[y[valid] == cls]
        rng.shuffle(cls_idx)
        if samples_per_class:
            n_train = min(samples_per_class, max(1, len(cls_idx) // 3))
        else:
            n_train = max(1, int(len(cls_idx) * cfg["dataset"]["train_ratio"]))
        remaining = cls_idx[n_train:]
        n_val = max(1, int(len(cls_idx) * cfg["dataset"]["val_ratio"]))
        train.extend(cls_idx[:n_train])
        val.extend(remaining[:n_val])
        test.extend(remaining[n_val:])
    return np.asarray(train), np.asarray(val), np.asarray(test)


class HSIPatchDataset(Dataset):
    def __init__(self, cube: np.ndarray, labels: np.ndarray, indices: np.ndarray, patch_size: int):
        self.cube = cube
        self.labels = labels
        self.indices = indices
        self.patch_size = patch_size
        self.pad = patch_size // 2
        self.padded = np.pad(cube, ((self.pad, self.pad), (self.pad, self.pad), (0, 0)), mode="reflect")
        self.width = labels.shape[1]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int):
        flat_idx = int(self.indices[item])
        row, col = divmod(flat_idx, self.width)
        r, c = row + self.pad, col + self.pad
        patch = self.padded[r - self.pad : r + self.pad + 1, c - self.pad : c + self.pad + 1]
        spectrum = self.cube[row, col]
        label = self.labels[row, col] - 1
        return {
            "patch": torch.from_numpy(patch.transpose(2, 0, 1)).float(),
            "spectrum": torch.from_numpy(spectrum).float(),
            "label": torch.tensor(label).long(),
        }


def create_dataloaders(cfg: Dict, seed: int):
    data = load_hsi(cfg, seed)
    train_idx, val_idx, test_idx = stratified_indices(data.labels, cfg, seed)
    cube = normalize_and_reduce(data.cube, train_idx, cfg["dataset"]["pca_components"], seed)
    cfg.setdefault("runtime", {})["spectral_dim"] = int(cube.shape[-1])
    patch_size = cfg["dataset"]["patch_size"]
    batch_size = cfg["training"]["batch_size"]
    workers = cfg["training"].get("num_workers", 0)
    loaders = {}
    for split, indices, shuffle in [("train", train_idx, True), ("val", val_idx, False), ("test", test_idx, False)]:
        ds = HSIPatchDataset(cube, data.labels, indices, patch_size)
        loaders[split] = DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=workers)
    return loaders, data.num_classes
