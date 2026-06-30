import torch
import torch.nn as nn

from .vision_encoder import VisionEncoder


class VideoEncoder(nn.Module):
    """
    Encode videos by sampling frames, encoding each with the vision encoder,
    then applying temporal attention to produce per-frame tokens.
    """

    def __init__(
        self,
        d_model: int = 512,
        image_size: int = 224,
        patch_size: int = 16,
        n_frames: int = 8,
        n_heads: int = 8,
    ):
        super().__init__()
        self.n_frames = n_frames
        self.frame_encoder = VisionEncoder(d_model, image_size, patch_size, n_layers=1, n_heads=n_heads)

        self.temporal_attn = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=d_model * 4,
                batch_first=True,
                activation="gelu",
            ),
            num_layers=2,
        )
        self.frame_pool = nn.Linear(d_model, d_model)

    def _encode_frames(self, frames: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = frames.shape
        flat = frames.view(B * T, C, H, W)
        encoded = self.frame_encoder(flat)
        pooled = encoded[:, 0, :]
        pooled = pooled.view(B, T, -1)
        pooled = self.frame_pool(pooled)
        return self.temporal_attn(pooled)

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Args:
            frames: (B, T, 3, H, W) sampled video frames
        Returns:
            (B, T, d_model) one token per frame
        """
        if self.training:
            return torch.utils.checkpoint.checkpoint(
                self._encode_frames,
                frames,
                use_reentrant=False,
            )
        return self._encode_frames(frames)

    @property
    def n_tokens(self) -> int:
        return self.n_frames