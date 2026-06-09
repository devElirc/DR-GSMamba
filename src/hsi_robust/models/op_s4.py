"""Order-Preserving Spectral Selective-Scan (OP-S4) block.

This is a **design choice, not a contribution** -- the spectral encoder is
ablated against a 1D-CNN and a Transformer in Phase 7. The implementation is a
simplified S4D-style state-space layer with:

* a diagonal, real-valued state matrix ``A`` initialised with HiPPO-LegS-inspired
  log-spaced decay rates (slow + fast timescales over the spectral axis);
* learnable input / output / skip vectors ``B``, ``C``, ``D``;
* a learned **per-band importance gate** that multiplies the raw spectrum before
  the SSM, providing the "selective" part of the name without the full
  data-dependent Mamba dynamics;
* a **bidirectional** scan (forward + reverse over spectral bands), pooled by
  mean over bands to produce a single per-pixel feature vector.

For a diagonal, time-invariant SSM, the recurrence
``h_t = a * h_{t-1} + B * x_t``, ``y_t = C * h_t + D * x_t`` is equivalent to
the causal convolution ``y = K * x + D * x`` where the per-channel impulse
response is ``K[t] = sum_k C_k * a_k^t * B_k``. We compute this convolution
once per forward via FFT, which is O(T log T) on the spectral axis and
substantially faster than a Python loop. The closed-form is exact (no
numerical regression vs the recurrent form to 1e-5 in float32).

Public surface (re-exported from :mod:`hsi_robust.models`):

* :class:`OPS4Block`     -- one S4D layer (used internally by the encoder).
* :class:`OPS4Encoder`   -- the full spectral encoder used by the model.
"""

from __future__ import annotations

import math

import torch
from torch import nn


def _hippo_legs_log_dt(d_model: int) -> torch.Tensor:
    """Log-spaced timescales between 1e-3 and 1e-1, after Gu et al. (S4)."""
    return torch.linspace(math.log(1e-3), math.log(1e-1), d_model)


class OPS4Block(nn.Module):
    """A single S4D-real layer with residual connection + GELU.

    The state recurrence on a single state dim ``k`` is

        h_k(t) = a_k * h_k(t-1) + B_k * x(t)
        y(t)   = sum_k C_k * h_k(t) + D * x(t)

    where ``a_k = exp(-dt_k * exp(log_A_k))`` is strictly in (0, 1) thanks to
    the positive exponential parameterisation, guaranteeing stability.
    """

    def __init__(self, d_model: int, d_state: int, *, dropout: float = 0.1) -> None:
        super().__init__()
        if d_model <= 0 or d_state <= 0:
            raise ValueError("d_model and d_state must be positive")
        self.d_model = int(d_model)
        self.d_state = int(d_state)

        # HiPPO-LegS-inspired init: per-channel log dt + per-(channel, state) log A.
        self.log_dt = nn.Parameter(_hippo_legs_log_dt(d_model))
        log_A = math.log(0.5) + 0.1 * torch.randn(d_model, d_state)
        self.log_A = nn.Parameter(log_A)
        self.B = nn.Parameter(torch.randn(d_model, d_state) / math.sqrt(d_state))
        self.C = nn.Parameter(torch.randn(d_model, d_state) / math.sqrt(d_state))
        self.D = nn.Parameter(torch.ones(d_model))

        self.norm = nn.LayerNorm(d_model)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def _kernel(self, length: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        """Compute the per-channel impulse response ``K[t]`` for ``t = 0..length-1``.

        Returns a ``(length, d_model)`` tensor.
        """
        dt = torch.exp(self.log_dt).unsqueeze(-1)  # (d_model, 1)
        A = -torch.exp(self.log_A)  # (d_model, d_state)
        log_a = dt * A  # (d_model, d_state), negative
        t = torch.arange(length, device=device, dtype=dtype)
        # powers[t, m, k] = exp(t * log_a[m, k]) in [0, 1].
        powers = torch.exp(log_a.unsqueeze(0) * t.view(length, 1, 1))
        # K[t, m] = sum_k C[m, k] * B[m, k] * powers[t, m, k]
        cb = (self.C * self.B).unsqueeze(0)  # (1, d_model, d_state)
        return (cb * powers).sum(dim=-1)  # (length, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """FFT-based forward over the time (= spectral band) axis.

        Parameters
        ----------
        x:
            tensor of shape ``(N, T, d_model)``.

        Returns
        -------
        Output tensor of shape ``(N, T, d_model)``.
        """
        if x.ndim != 3 or x.shape[-1] != self.d_model:
            raise ValueError(f"expected (N, T, {self.d_model}); got {tuple(x.shape)}")
        residual = x
        x_norm = self.norm(x)
        t = x_norm.shape[1]
        # Length-2T FFT for causal convolution (avoids cyclic wrap).
        fft_len = 2 * t
        kernel = self._kernel(t, dtype=x_norm.dtype, device=x_norm.device)  # (T, d_model)
        x_fft = torch.fft.rfft(x_norm, n=fft_len, dim=1)  # (N, fft_len/2+1, d_model)
        k_fft = torch.fft.rfft(kernel, n=fft_len, dim=0)  # (fft_len/2+1, d_model)
        y_fft = x_fft * k_fft.unsqueeze(0)
        y = torch.fft.irfft(y_fft, n=fft_len, dim=1)[:, :t, :]
        y = y + self.D * x_norm  # skip term

        y = self.act(y)
        y = self.dropout(y)
        return y + residual


class OPS4Encoder(nn.Module):
    """Bidirectional OP-S4 encoder for spectral input.

    Architecture::

        spectrum (N, B)
            --> per-band importance gate (sigmoid scalar per band)
            --> input projection 1 -> d_model
            --> stack of OPS4Block (forward pass) and OPS4Block (reverse pass)
            --> concat(forward, reverse)
            --> pool (mean over bands)
            --> output projection (2 * d_model -> out_dim)
    """

    def __init__(
        self,
        *,
        num_bands: int,
        d_model: int = 64,
        d_state: int = 16,
        num_layers: int = 2,
        out_dim: int = 64,
        bidirectional: bool = True,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if num_bands <= 0:
            raise ValueError("num_bands must be positive")
        self.num_bands = int(num_bands)
        self.d_model = int(d_model)
        self.bidirectional = bool(bidirectional)
        self.out_dim = int(out_dim)

        # Per-band importance gate (the "selective" part).
        self.band_gate = nn.Parameter(torch.zeros(num_bands))

        # Input projection: scalar per band -> d_model channels.
        self.in_proj = nn.Linear(1, d_model)

        # Forward stack.
        self.fwd_blocks = nn.ModuleList(
            [OPS4Block(d_model=d_model, d_state=d_state, dropout=dropout) for _ in range(num_layers)]
        )
        # Reverse stack (separate parameters so each direction can specialise).
        self.bwd_blocks = (
            nn.ModuleList(
                [
                    OPS4Block(d_model=d_model, d_state=d_state, dropout=dropout)
                    for _ in range(num_layers)
                ]
            )
            if bidirectional
            else None
        )

        out_in = 2 * d_model if bidirectional else d_model
        self.out_proj = nn.Linear(out_in, out_dim)

    def forward(self, spectrum: torch.Tensor) -> torch.Tensor:
        """Encode a batch of per-pixel spectra.

        Parameters
        ----------
        spectrum:
            ``(N, num_bands)`` tensor of standardised band intensities.

        Returns
        -------
        ``(N, out_dim)`` per-pixel spectral feature.
        """
        if spectrum.ndim != 2 or spectrum.shape[-1] != self.num_bands:
            raise ValueError(
                f"expected (N, {self.num_bands}); got {tuple(spectrum.shape)}"
            )
        gate = torch.sigmoid(self.band_gate)  # (num_bands,)
        x = spectrum * gate  # (N, num_bands)
        x = x.unsqueeze(-1)  # (N, num_bands, 1)
        x = self.in_proj(x)  # (N, num_bands, d_model)

        # Forward scan.
        h_fwd = x
        for block in self.fwd_blocks:
            h_fwd = block(h_fwd)

        if self.bwd_blocks is not None:
            # Reverse scan: flip along band axis, run blocks, flip back.
            h_bwd = torch.flip(x, dims=[1])
            for block in self.bwd_blocks:
                h_bwd = block(h_bwd)
            h_bwd = torch.flip(h_bwd, dims=[1])
            h = torch.cat([h_fwd, h_bwd], dim=-1)  # (N, B, 2 * d_model)
        else:
            h = h_fwd

        pooled = h.mean(dim=1)  # (N, 2*d_model) or (N, d_model)
        return self.out_proj(pooled)  # (N, out_dim)
