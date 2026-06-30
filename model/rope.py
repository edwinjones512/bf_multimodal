"""Rotary Position Embeddings (RoPE) with optional NTK-style length scaling."""

from __future__ import annotations

import torch
import torch.nn as nn


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply RoPE to query and key tensors.

    Args:
        q, k: (B, n_heads, T, head_dim)
        cos, sin: (1, 1, T, head_dim) broadcastable
    """
    q_out = (q * cos) + (rotate_half(q) * sin)
    k_out = (k * cos) + (rotate_half(k) * sin)
    return q_out, k_out


class RotaryEmbedding(nn.Module):
    """Precomputed RoPE cos/sin tables with dynamic length extension."""

    def __init__(
        self,
        head_dim: int,
        max_seq_len: int = 2048,
        theta: float = 10000.0,
        scale: float = 1.0,
    ):
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError(f"RoPE requires even head_dim, got {head_dim}")

        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.theta = theta
        self.scale = scale
        self.register_buffer("cos_cached", torch.empty(0), persistent=False)
        self.register_buffer("sin_cached", torch.empty(0), persistent=False)
        self._build_cache(max_seq_len)

    def set_scale(self, scale: float) -> None:
        """NTK-style inference scaling: scale > 1 extends usable context length."""
        if scale != self.scale:
            self.scale = scale
            self._build_cache(self.max_seq_len)

    def _build_cache(self, seq_len: int, device: torch.device | None = None, dtype: torch.dtype | None = None) -> None:
        device = device or self.cos_cached.device if self.cos_cached.numel() else torch.device("cpu")
        dtype = dtype or (self.cos_cached.dtype if self.cos_cached.numel() else torch.float32)

        effective_theta = self.theta * (self.scale ** (self.head_dim / (self.head_dim - 2)))
        positions = torch.arange(seq_len, device=device, dtype=dtype)
        inv_freq = 1.0 / (
            effective_theta ** (torch.arange(0, self.head_dim, 2, device=device, dtype=dtype) / self.head_dim)
        )
        freqs = torch.outer(positions, inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.cos_cached = emb.cos()[None, None, :, :]
        self.sin_cached = emb.sin()[None, None, :, :]
        self.max_seq_len = seq_len

    def get_cos_sin(
        self,
        seq_len: int,
        offset: int = 0,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if seq_len + offset > self.max_seq_len:
            self._build_cache(seq_len + offset, device=device, dtype=dtype)

        cos = self.cos_cached[:, :, offset : offset + seq_len, :]
        sin = self.sin_cached[:, :, offset : offset + seq_len, :]

        if device is not None:
            cos = cos.to(device=device)
            sin = sin.to(device=device)
        if dtype is not None:
            cos = cos.to(dtype=dtype)
            sin = sin.to(dtype=dtype)
        return cos, sin