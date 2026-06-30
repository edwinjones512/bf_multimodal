import torch

from config import ModelConfig, SPECIAL_TOKENS
from model import MultimodalModel
from model.rope import RotaryEmbedding, apply_rotary_emb


def test_rope_changes_qk_not_v():
    rope = RotaryEmbedding(head_dim=64, max_seq_len=128)
    q = torch.randn(2, 4, 16, 64)
    k = torch.randn(2, 4, 16, 64)
    cos, sin = rope.get_cos_sin(16)
    q2, k2 = apply_rotary_emb(q, k, cos, sin)
    assert not torch.allclose(q, q2)
    assert not torch.allclose(k, k2)


def test_rope_offset_differs():
    rope = RotaryEmbedding(head_dim=64, max_seq_len=128)
    q = torch.randn(1, 2, 8, 64)
    k = q.clone()
    cos0, sin0 = rope.get_cos_sin(8, offset=0)
    cos4, sin4 = rope.get_cos_sin(8, offset=4)
    _, k_pos0 = apply_rotary_emb(q, k, cos0, sin0)
    _, k_pos4 = apply_rotary_emb(q, k, cos4, sin4)
    assert not torch.allclose(k_pos0, k_pos4)


def test_model_forward_without_pos_emb():
    cfg = ModelConfig(d_model=128, n_heads=4, n_layers=2, d_ff=256, max_seq_len=64, vocab_size=1000)
    model = MultimodalModel(cfg)
    assert not hasattr(model.backbone, "pos_emb")

    ids = torch.randint(7, 50, (2, 32))
    labels = ids.clone()
    out = model(text_ids=ids, labels=labels)
    assert out["loss"].item() > 0
    assert out["logits"].shape[0] == 2
    assert out["logits"].shape[2] == cfg.vocab_size


def test_rope_scale_extends_cache():
    rope = RotaryEmbedding(head_dim=64, max_seq_len=64, scale=1.0)
    rope.set_scale(2.0)
    cos, sin = rope.get_cos_sin(128)
    assert cos.shape[2] == 128


def test_flash_attention_path_runs():
    cfg = ModelConfig(d_model=128, n_heads=4, n_layers=2, d_ff=256, max_seq_len=64, vocab_size=1000)
    model = MultimodalModel(cfg)
    ids = torch.randint(7, 50, (2, 32))
    logits = model(text_ids=ids)["logits"]
    assert logits.shape[0] == 2