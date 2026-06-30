import torch

from config import ModelConfig
from model import MultimodalModel
from model.attention import CausalSelfAttention, sliding_window_mask


def test_sliding_window_mask_shape_and_band():
    mask = sliding_window_mask(8, window_size=3, device=torch.device("cpu"))
    assert mask.shape == (8, 8)
    # position 5 may attend to 3,4,5 (window=3)
    assert mask[5, 2].item() is False
    assert mask[5, 3].item() is True
    assert mask[5, 4].item() is True
    assert mask[5, 5].item() is True


def test_window_zero_is_full_causal():
    mask = sliding_window_mask(8, window_size=0, device=torch.device("cpu"))
    assert mask.sum() == 36  # 8*9/2 lower triangular


def test_sliding_window_forward_long_sequence():
    cfg = ModelConfig(
        d_model=128, n_heads=4, n_layers=2, d_ff=256,
        max_seq_len=128, vocab_size=1000, sliding_window=16,
    )
    model = MultimodalModel(cfg)
    ids = torch.randint(7, 50, (2, 64))
    logits = model(text_ids=ids)["logits"]
    assert logits.shape[0] == 2


def test_windowed_differs_from_full_attention():
    d_model, n_heads = 64, 4
    x = torch.randn(1, 32, d_model)

    attn_full = CausalSelfAttention(d_model, n_heads, window_size=0)
    attn_window = CausalSelfAttention(d_model, n_heads, window_size=8)

    out_full = attn_full(x)
    out_window = attn_window(x)
    assert not torch.allclose(out_full, out_window)