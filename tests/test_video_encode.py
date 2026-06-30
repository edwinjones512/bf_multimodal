import torch

from config import ModelConfig
from model import MultimodalModel


def test_video_micro_batch_encoding():
    cfg = ModelConfig(
        d_model=64,
        n_heads=4,
        n_layers=2,
        d_ff=128,
        max_seq_len=256,
        vocab_size=256,
        image_size=112,
        patch_size=16,
        n_video_frames=8,
        video_train_frames=4,
        video_encode_micro_batch=1,
    )
    model = MultimodalModel(cfg)
    model.train()
    text_ids = torch.randint(9, 50, (3, 8))
    video = torch.randn(3, cfg.video_train_frames, 3, cfg.image_size, cfg.image_size)

    input_ids, embeds = model.build_inputs(text_ids, video_frames=video)
    assert embeds.shape[0] == 3
    assert (input_ids[:, 0] == 5).all()  # <video> token


def test_video_encoder_gradient_checkpointing():
    cfg = ModelConfig(d_model=64, n_heads=4, n_layers=2, d_ff=128, image_size=112, n_video_frames=4)
    model = MultimodalModel(cfg)
    model.train()
    frames = torch.randn(2, 4, 3, cfg.image_size, cfg.image_size)
    tokens = model._encode_videos(frames)
    assert tokens.shape[0] == 2
    loss = tokens.sum()
    loss.backward()