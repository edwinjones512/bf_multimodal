import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import CausalSelfAttention


def _sample_next_token(
    logits: torch.Tensor,
    temperature: float,
    top_k: int,
) -> torch.Tensor:
    """Sample one token; fall back to greedy argmax when logits are non-finite."""
    logits = logits / max(temperature, 1e-5)
    if not torch.isfinite(logits).all():
        safe = torch.nan_to_num(logits, nan=0.0, posinf=1e4, neginf=-1e4)
        return safe.argmax(dim=-1, keepdim=True)

    if top_k > 0:
        k = min(top_k, logits.size(-1))
        v, _ = torch.topk(logits, k)
        logits = logits.clone()
        logits[logits < v[:, [-1]]] = float("-inf")

    probs = F.softmax(logits, dim=-1)
    if not torch.isfinite(probs).all() or (probs < 0).any():
        return logits.argmax(dim=-1, keepdim=True)

    return torch.multinomial(probs, num_samples=1)


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        max_seq_len: int = 2048,
        rope_theta: float = 10000.0,
        rope_scale: float = 1.0,
        window_size: int = 0,
    ):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(
            d_model, n_heads, dropout,
            max_seq_len=max_seq_len,
            rope_theta=rope_theta,
            rope_scale=rope_scale,
            window_size=window_size,
        )
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ff, dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        pos_offset: int = 0,
    ) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), mask, pos_offset=pos_offset)
        x = x + self.ff(self.ln2(x))
        return x


class DecoderTransformer(nn.Module):
    """Decoder-only transformer with RoPE and Flash Attention (via SDPA)."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 6,
        d_ff: int = 2048,
        max_seq_len: int = 1024,
        dropout: float = 0.1,
        pad_token_id: int = 0,
        rope_theta: float = 10000.0,
        rope_scale: float = 1.0,
        inference_max_seq_len: int | None = None,
        sliding_window: int = 0,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.inference_max_seq_len = inference_max_seq_len or max_seq_len * 4
        self.sliding_window = sliding_window
        self.pad_token_id = pad_token_id
        self.rope_scale = rope_scale

        rope_cache_len = max(max_seq_len, self.inference_max_seq_len)

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model, n_heads, d_ff, dropout,
                max_seq_len=rope_cache_len,
                rope_theta=rope_theta,
                rope_scale=rope_scale,
                window_size=sliding_window,
            )
            for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight

    def set_rope_scale(self, scale: float) -> None:
        """Apply NTK-style RoPE scaling for longer inference context."""
        self.rope_scale = scale
        for block in self.blocks:
            block.attn.rope.set_scale(scale)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        pos_offset: int = 0,
    ) -> torch.Tensor:
        if inputs_embeds is None:
            x = self.token_emb(input_ids)
        else:
            x = inputs_embeds

        x = self.drop(x)

        mask = attention_mask.bool() if attention_mask is not None else None

        for block in self.blocks:
            x = block(x, mask, pos_offset=pos_offset)

        x = self.ln_f(x)
        return self.lm_head(x)

    def forward_hidden(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        pos_offset: int = 0,
    ) -> torch.Tensor:
        """Return final hidden states before the language-model head."""
        if inputs_embeds is None:
            x = self.token_emb(input_ids)
        else:
            x = inputs_embeds

        x = self.drop(x)
        mask = attention_mask.bool() if attention_mask is not None else None

        for block in self.blocks:
            x = block(x, mask, pos_offset=pos_offset)

        return self.ln_f(x)

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 128,
        temperature: float = 0.8,
        top_k: int = 50,
        eos_token_id: int | None = None,
        inputs_embeds: torch.Tensor | None = None,
        rope_scale: float | None = None,
    ) -> torch.Tensor:
        self.eval()
        if rope_scale is not None:
            self.set_rope_scale(rope_scale)

        ctx_limit = self.inference_max_seq_len
        generated = input_ids

        if inputs_embeds is not None:
            embeds = inputs_embeds
            for _ in range(max_new_tokens):
                if embeds.shape[1] > ctx_limit:
                    embeds = embeds[:, -ctx_limit:]
                    generated = generated[:, -ctx_limit:]

                logits = self.forward(generated, inputs_embeds=embeds)
                next_token = _sample_next_token(logits[:, -1, :], temperature, top_k)
                generated = torch.cat([generated, next_token], dim=1)

                next_embed = self.token_emb(next_token)
                embeds = torch.cat([embeds, next_embed], dim=1)

                if eos_token_id is not None and (next_token == eos_token_id).all():
                    break
            return generated

        for _ in range(max_new_tokens):
            ctx = generated[:, -ctx_limit:]
            logits = self.forward(ctx)
            next_token = _sample_next_token(logits[:, -1, :], temperature, top_k)
            generated = torch.cat([generated, next_token], dim=1)

            if eos_token_id is not None and (next_token == eos_token_id).all():
                break

        return generated