import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _build_mel_filterbank(
    sample_rate: int,
    n_fft: int,
    n_mels: int,
    f_min: float = 0.0,
    f_max: float | None = None,
) -> torch.Tensor:
    """HTK mel filterbank (n_mels, n_fft // 2 + 1)."""
    if f_max is None:
        f_max = sample_rate / 2

    def hz_to_mel(hz: float) -> float:
        return 2595.0 * math.log10(1.0 + hz / 700.0)

    def mel_to_hz(mel: float) -> float:
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    n_freqs = n_fft // 2 + 1
    fft_freqs = torch.linspace(0, sample_rate / 2, n_freqs)
    mel_min = hz_to_mel(f_min)
    mel_max = hz_to_mel(f_max)
    mel_points = torch.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = torch.tensor([mel_to_hz(m) for m in mel_points.tolist()])
    bins = torch.floor((n_fft + 1) * hz_points / sample_rate).long()

    filterbank = torch.zeros(n_mels, n_freqs)
    for i in range(n_mels):
        left, center, right = bins[i], bins[i + 1], bins[i + 2]
        if center == left:
            center += 1
        if right == center:
            right += 1
        for j in range(left, center):
            if 0 <= j < n_freqs and center > left:
                filterbank[i, j] = (j - left) / (center - left)
        for j in range(center, right):
            if 0 <= j < n_freqs and right > center:
                filterbank[i, j] = (right - j) / (right - center)

    return filterbank


class LogMelSpectrogram(nn.Module):
    """Waveform (B, samples) -> log-mel spectrogram (B, n_mels, time)."""

    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 400,
        hop_length: int = 160,
        n_mels: int = 80,
    ):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.register_buffer("window", torch.hann_window(n_fft))
        self.register_buffer("mel_fb", _build_mel_filterbank(sample_rate, n_fft, n_mels))

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.dim() == 3:
            waveform = waveform.squeeze(1)
        waveform = waveform.float()

        spec = torch.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.window,
            return_complex=True,
        )
        power = spec.abs() ** 2
        mel = torch.einsum("mn,bnt->bmt", self.mel_fb, power)
        return torch.log(mel.clamp(min=1e-10))


class AudioEncoder(nn.Module):
    """Encode audio waveforms into a fixed-length token sequence for the transformer backbone."""

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

        self.mel = LogMelSpectrogram(sample_rate, n_fft, hop_length, n_mels)
        grid_h = max(1, int(math.sqrt(n_tokens)))
        grid_w = max(1, math.ceil(n_tokens / grid_h))
        self.pool_shape = (grid_h, grid_w)

        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(32, d_model, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(self.pool_shape),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

    def _prepare_waveform(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.dim() == 3:
            waveform = waveform.squeeze(1)

        n = waveform.shape[1]
        if n < self.max_samples:
            waveform = F.pad(waveform, (0, self.max_samples - n))
        else:
            waveform = waveform[:, : self.max_samples]
        return waveform

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Args:
            waveform: (B, samples) or (B, 1, samples) mono audio
        Returns:
            (B, n_tokens, d_model)
        """
        waveform = self._prepare_waveform(waveform).clamp(-1.0, 1.0)
        mel = self.mel(waveform).unsqueeze(1)
        features = self.conv(mel)
        B, D, H, W = features.shape
        tokens = features.permute(0, 2, 3, 1).reshape(B, H * W, D)
        if tokens.shape[1] > self.n_tokens:
            tokens = tokens[:, : self.n_tokens, :]
        elif tokens.shape[1] < self.n_tokens:
            pad = self.n_tokens - tokens.shape[1]
            tokens = F.pad(tokens, (0, 0, 0, pad))
        return self.encoder(tokens)