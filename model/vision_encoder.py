import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    """Split an image into patches and project to d_model (ViT-style)."""

    def __init__(self, image_size: int = 224, patch_size: int = 16, d_model: int = 512):
        super().__init__()
        self.n_patches = (image_size // patch_size) ** 2
        self.proj = nn.Conv2d(3, d_model, kernel_size=patch_size, stride=patch_size)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # images: (B, 3, H, W) -> (B, n_patches, d_model)
        x = self.proj(images)
        return x.flatten(2).transpose(1, 2)


class VisionEncoder(nn.Module):
    """Encode images into a sequence of tokens for the transformer backbone."""

    def __init__(
        self,
        d_model: int = 512,
        image_size: int = 224,
        patch_size: int = 16,
        n_layers: int = 2,
        n_heads: int = 8,
    ):
        super().__init__()
        self.patch_embed = PatchEmbedding(image_size, patch_size, d_model)
        self.n_patches = self.patch_embed.n_patches

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        B = images.shape[0]
        patches = self.patch_embed(images)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, patches], dim=1)
        return self.encoder(x)

    @property
    def n_tokens(self) -> int:
        return self.n_patches + 1