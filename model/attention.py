import torch
import torch.nn as nn
import torch.nn.functional as F

from .rope import RotaryEmbedding, apply_rotary_emb


def sliding_window_mask(seq_len: int, window_size: int, device: torch.device) -> torch.Tensor:
    """
    Build a causal sliding-window attention mask.

    Token at position i may attend to positions j where:
        max(0, i - window_size + 1) <= j <= i

    Returns:
        (seq_len, seq_len) bool tensor (True = may attend)
    """
    if window_size <= 0 or window_size >= seq_len:
        return torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool))

    row = torch.arange(seq_len, device=device).unsqueeze(1)
    col = torch.arange(seq_len, device=device).unsqueeze(0)
    causal = col <= row
    in_window = (row - col) < window_size
    return causal & in_window


class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1,
        max_seq_len: int = 2048,
        rope_theta: float = 10000.0,
        rope_scale: float = 1.0,
        window_size: int = 0,
    ):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.dropout = dropout
        self.window_size = window_size

        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.rope = RotaryEmbedding(
            self.head_dim,
            max_seq_len=max_seq_len,
            theta=rope_theta,
            scale=rope_scale,
        )

    def _attention_mask(
        self,
        seq_len: int,
        key_padding: torch.Tensor | None,
        device: torch.device,
    ) -> torch.Tensor | None:
        """Return SDPA bool mask (True = attend) or None for full causal."""
        use_window = self.window_size > 0 and self.window_size < seq_len

        if use_window:
            attn_mask = sliding_window_mask(seq_len, self.window_size, device)
            attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, T, T)
        elif key_padding is not None:
            causal = torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool))
            attn_mask = causal.unsqueeze(0).unsqueeze(0)
        else:
            return None

        if key_padding is not None:
            key_pad = key_padding.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, T)
            attn_mask = attn_mask & key_pad

        return attn_mask

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        pos_offset: int = 0,
    ) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        cos, sin = self.rope.get_cos_sin(T, offset=pos_offset, device=x.device, dtype=x.dtype)
        q, k = apply_rotary_emb(q, k, cos, sin)

        dropout_p = self.dropout if self.training else 0.0
        attn_mask = self._attention_mask(T, mask, x.device)

        if attn_mask is not None:
            if attn_mask.shape[0] == 1 and B > 1:
                attn_mask = attn_mask.expand(B, -1, -1, -1)

            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=dropout_p,
                is_causal=False,
            )
        else:
            out = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=dropout_p,
                is_causal=True,
            )

        out = out.transpose(1, 2).reshape(B, T, C)
        return self.out_proj(out)