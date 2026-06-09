"""Per-band standardisation, PCA reduction, and patch extraction.

All three transforms are **fit on the train pixels only** so that the test split
remains causally independent of the model -- a precondition for the leak-free
evaluation protocol declared in ``EXPERIMENT_PLAN.md``.

* :class:`PerBandStandardize` fits a per-band z-score on the training spectra.
* :class:`PCAReducer` wraps :class:`sklearn.decomposition.PCA` with the
  reproducibility settings we need (full SVD solver, fixed random state).
* :func:`pad_cube_for_patches` and :func:`extract_patch` implement the
  reflection-padded patch extraction used by the spatial CNN stem.
"""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA


class PerBandStandardize:
    """Per-band z-score normaliser, ``fit`` on training spectra only.

    The fitted mean and std are exposed as ``mean_`` and ``std_`` attributes so
    that leak-freedom can be unit-tested.
    """

    def __init__(self, eps: float = 1e-8) -> None:
        self.eps = float(eps)
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None

    def fit(self, spectra: np.ndarray) -> PerBandStandardize:
        """Fit per-band mean and std on ``spectra`` (shape ``(N, B)``)."""
        if spectra.ndim != 2:
            raise ValueError(f"expected 2D spectra (N, B); got {spectra.shape}")
        self.mean_ = spectra.mean(axis=0).astype(np.float32)
        self.std_ = spectra.std(axis=0).astype(np.float32) + self.eps
        return self

    def transform(self, spectra: np.ndarray) -> np.ndarray:
        """Apply the standardisation in-place-safe to ``spectra``.

        Accepts both 2D ``(N, B)`` and 3D ``(H, W, B)`` arrays.
        """
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("PerBandStandardize must be fit before transform")
        if spectra.ndim == 2:
            return ((spectra - self.mean_) / self.std_).astype(np.float32)
        if spectra.ndim == 3:
            return ((spectra - self.mean_) / self.std_).astype(np.float32)
        raise ValueError(f"expected 2D or 3D input; got {spectra.shape}")

    def fit_transform(self, spectra: np.ndarray) -> np.ndarray:
        """Convenience: ``fit`` then ``transform``."""
        return self.fit(spectra).transform(spectra)


class PCAReducer:
    """Train-only PCA wrapper for the spatial branch.

    The wrapped :class:`sklearn.decomposition.PCA` is constructed with
    ``svd_solver='full'`` and a fixed ``random_state`` so that two invocations
    on the same training spectra produce bit-identical outputs.
    """

    def __init__(self, n_components: int, random_state: int = 0) -> None:
        self.n_components = int(n_components)
        self.random_state = int(random_state)
        self.pca: PCA | None = None

    def fit(self, spectra: np.ndarray) -> PCAReducer:
        """Fit PCA on the 2D training spectra ``(N_train, B)``."""
        if spectra.ndim != 2:
            raise ValueError(f"expected 2D spectra (N, B); got {spectra.shape}")
        if spectra.shape[0] < self.n_components:
            raise ValueError(
                f"PCA requires N >= n_components; got N={spectra.shape[0]} and "
                f"n_components={self.n_components}"
            )
        self.pca = PCA(
            n_components=self.n_components,
            svd_solver="full",
            random_state=self.random_state,
        )
        self.pca.fit(spectra)
        return self

    def transform(self, spectra: np.ndarray) -> np.ndarray:
        """Apply PCA. Accepts ``(N, B)`` or ``(H, W, B)`` input.

        Output shape is ``(N, n_components)`` or ``(H, W, n_components)``.
        """
        if self.pca is None:
            raise RuntimeError("PCAReducer must be fit before transform")
        if spectra.ndim == 2:
            return self.pca.transform(spectra).astype(np.float32)
        if spectra.ndim == 3:
            h, w, b = spectra.shape
            reshaped = spectra.reshape(h * w, b)
            transformed = self.pca.transform(reshaped).astype(np.float32)
            return transformed.reshape(h, w, self.n_components)
        raise ValueError(f"expected 2D or 3D input; got {spectra.shape}")

    @property
    def mean_(self) -> np.ndarray:
        """The mean vector fitted by the underlying sklearn PCA."""
        if self.pca is None:
            raise RuntimeError("PCAReducer must be fit before reading mean_")
        return np.asarray(self.pca.mean_)


def pad_cube_for_patches(cube: np.ndarray, patch_size: int) -> np.ndarray:
    """Reflection-pad the spatial dimensions for centre-aligned patch extraction.

    Output shape is ``(H + 2 * half, W + 2 * half, B)`` where ``half = patch_size // 2``.
    """
    if cube.ndim != 3:
        raise ValueError(f"expected 3D cube (H, W, B); got {cube.shape}")
    if patch_size <= 0 or patch_size % 2 == 0:
        raise ValueError(f"patch_size must be positive and odd; got {patch_size}")
    half = patch_size // 2
    return np.pad(cube, ((half, half), (half, half), (0, 0)), mode="reflect")


def extract_patch(
    padded_cube: np.ndarray,
    i: int,
    j: int,
    patch_size: int,
) -> np.ndarray:
    """Extract a ``patch_size x patch_size x B`` patch.

    ``(i, j)`` are coordinates in the **original (unpadded)** cube. The padded
    cube is expected to be the output of :func:`pad_cube_for_patches`.
    """
    return padded_cube[i : i + patch_size, j : j + patch_size, :]
