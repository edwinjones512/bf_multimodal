import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .audio_encoder import LogMelSpectrogram, _build_mel_filterbank


class MelVocoder(nn.Module):
    """Convert log-mel spectrograms to waveforms with Griffin-Lim."""

    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 400,
        hop_length: int = 160,
        n_mels: int = 80,
        n_iter: int = 16,
    ):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_iter = n_iter
        self.register_buffer("window", torch.hann_window(n_fft))
        mel_fb = _build_mel_filterbank(sample_rate, n_fft, n_mels)
        self.register_buffer("mel_pinv", torch.linalg.pinv(mel_fb))

    def _mel_to_magnitude(self, log_mel: torch.Tensor) -> torch.Tensor:
        mel = torch.exp(log_mel.clamp(max=8.0))
        return torch.einsum("fn,bmt->bft", self.mel_pinv, mel).clamp(min=0)

    def _griffin_lim(self, magnitudes: torch.Tensor) -> torch.Tensor:
        angles = torch.rand_like(magnitudes) * 2 * math.pi
        for _ in range(self.n_iter):
            complex_spec = torch.polar(magnitudes, angles)
            waveform = torch.istft(
                complex_spec,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                window=self.window,
            )
            spec = torch.stft(
                waveform,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                window=self.window,
                return_complex=True,
            )
            angles = spec.angle()
        complex_spec = torch.polar(magnitudes, angles)
        return torch.istft(
            complex_spec,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.window,
        )

    def forward(self, log_mel: torch.Tensor) -> torch.Tensor:
        magnitudes = self._mel_to_magnitude(log_mel)
        return self._griffin_lim(magnitudes)


class AudioDecoder(nn.Module):
    """Decode speech latents into waveforms for speech output."""

    def __init__(
        self,
        d_model: int = 512,
        sample_rate: int = 16000,
        n_mels: int = 80,
        n_fft: int = 400,
        hop_length: int = 160,
        max_samples: int = 80_000,
        n_tokens: int = 32,
        n_heads: int = 8,
        n_layers: int = 2,
    ):
        super().__init__()
        self.max_samples = max_samples
        self.n_tokens = n_tokens
        self.target_mel_frames = max_samples // hop_length + 1

        grid_h = max(1, int(math.sqrt(n_tokens)))
        grid_w = max(1, math.ceil(n_tokens / grid_h))
        self.grid_shape = (grid_h, grid_w)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            batch_first=True,
            activation="gelu",
        )
        self.decoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.to_mel = nn.Sequential(
            nn.ConvTranspose2d(d_model, 32, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),
            nn.AdaptiveAvgPool2d((n_mels, self.target_mel_frames)),
        )
        self.vocoder = MelVocoder(sample_rate, n_fft, hop_length, n_mels)

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        """
        Args:
            latents: (B, n_tokens, d_model) speech token sequence
        Returns:
            (B, samples) mono waveform
        """
        x = self.decoder(latents)
        B, T, D = x.shape
        gh, gw = self.grid_shape

        if T < gh * gw:
            x = F.pad(x, (0, 0, 0, gh * gw - T))
        elif T > gh * gw:
            x = x[:, : gh * gw, :]

        x = x.transpose(1, 2).reshape(B, D, gh, gw)
        log_mel = self.to_mel(x).squeeze(1)
        waveform = self.vocoder(log_mel)

        if waveform.shape[1] > self.max_samples:
            waveform = waveform[:, : self.max_samples]
        elif waveform.shape[1] < self.max_samples:
            waveform = F.pad(waveform, (0, self.max_samples - waveform.shape[1]))
        return waveform