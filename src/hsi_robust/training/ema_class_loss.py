"""EMA-stabilised per-class loss tracker.

Implements §4.4 of ``docs/math/cfa_gdro.md`` exactly:

* For each class ``k``, maintain a running estimate ``hat_ell_k``;
* On every batch, update ``hat_ell_k = eta * hat_ell_k + (1 - eta) * bar_ell_k_batch``
  **only if class ``k`` appears in the batch**;
* Otherwise leave ``hat_ell_k`` unchanged.

The first time a class is seen, the EMA is initialised to the batch mean
(equivalent to setting the very first ``hat_ell_k = bar_ell_k``); this avoids
the ``eta * 0 + (1 - eta) * bar_ell_k`` cold-start bias.
"""

from __future__ import annotations

import torch


class EMAClassLoss:
    """Stateful EMA over per-class loss estimates.

    Parameters
    ----------
    num_classes:
        ``K``.
    momentum:
        ``eta`` from §4.4; default ``0.9`` per the math note.
    device, dtype:
        Forwarded to internal buffers.
    """

    def __init__(
        self,
        num_classes: int,
        *,
        momentum: float = 0.9,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        if num_classes <= 0:
            raise ValueError("num_classes must be positive")
        if not (0.0 <= momentum < 1.0):
            raise ValueError(f"momentum must lie in [0, 1); got {momentum}")
        self.num_classes = int(num_classes)
        self.momentum = float(momentum)
        self._losses = torch.zeros(num_classes, device=device, dtype=dtype)
        self._seen = torch.zeros(num_classes, device=device, dtype=torch.bool)

    @property
    def losses(self) -> torch.Tensor:
        """Current EMA estimate, shape ``(K,)``. Use ``.clone()`` if mutating."""
        return self._losses

    @property
    def seen(self) -> torch.Tensor:
        """Boolean ``(K,)`` mask of classes ever observed."""
        return self._seen

    def reset(self) -> None:
        self._losses.zero_()
        self._seen.zero_()

    @torch.no_grad()
    def update(
        self, per_sample_losses: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        """Update the EMA with one batch.

        Parameters
        ----------
        per_sample_losses:
            ``(N,)`` per-sample losses (detached or not — gradients are
            discarded by ``@torch.no_grad`` either way).
        labels:
            ``(N,)`` integer labels in ``[0, K)``.

        Returns
        -------
        ``(K,)`` per-class batch means *as computed inside the update* (useful
        for logging). Classes absent from the batch are returned as ``0``.
        """
        if per_sample_losses.ndim != 1 or labels.shape != per_sample_losses.shape:
            raise ValueError("expected 1-D losses and labels of equal shape")
        sums = torch.zeros_like(self._losses)
        sums = sums.scatter_add(0, labels, per_sample_losses.to(self._losses.dtype))
        counts = torch.zeros(self.num_classes, device=self._losses.device, dtype=torch.long)
        counts = counts.scatter_add(0, labels, torch.ones_like(labels, dtype=torch.long))
        batch_seen = counts > 0
        batch_mean = sums / counts.clamp(min=1).to(self._losses.dtype)
        # Update only present classes.
        fresh = batch_seen & ~self._seen  # first time we see this class
        warm = batch_seen & self._seen
        self._losses = torch.where(
            fresh,
            batch_mean,
            torch.where(
                warm,
                self.momentum * self._losses + (1.0 - self.momentum) * batch_mean,
                self._losses,
            ),
        )
        self._seen = self._seen | batch_seen
        return batch_mean

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {"losses": self._losses.clone(), "seen": self._seen.clone()}

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        self._losses.copy_(state["losses"])
        self._seen.copy_(state["seen"])
