"""Load and preprocess images, videos, and audio for multimodal prompts."""

import math
import struct
import wave
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms


def _image_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def load_image(path: str | Path, image_size: int = 224) -> torch.Tensor:
    """Load a single image as (1, 3, H, W) tensor."""
    img = Image.open(path).convert("RGB")
    tensor = _image_transform(image_size)(img)
    return tensor.unsqueeze(0)


def load_images(paths: list[str | Path], image_size: int = 224) -> torch.Tensor:
    """Load multiple images as (B, 3, H, W) tensor."""
    transform = _image_transform(image_size)
    tensors = [transform(Image.open(p).convert("RGB")) for p in paths]
    return torch.stack(tensors)


def load_video_frames(
    path: str | Path,
    n_frames: int = 8,
    image_size: int = 224,
) -> torch.Tensor:
    """
    Sample n_frames evenly from a video file.

    Returns:
        (1, n_frames, 3, H, W) tensor
    """
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        raise ValueError(f"Video has no frames: {path}")

    indices = np.linspace(0, total - 1, n_frames, dtype=int)
    transform = _image_transform(image_size)
    frames: list[torch.Tensor] = []

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret:
            continue
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(frame_rgb)
        frames.append(transform(pil))

    cap.release()

    if not frames:
        raise ValueError(f"Could not read frames from: {path}")

    while len(frames) < n_frames:
        frames.append(frames[-1])

    stacked = torch.stack(frames[:n_frames])
    return stacked.unsqueeze(0)


def _load_wav_mono(path: Path, sample_rate: int) -> torch.Tensor:
    with wave.open(str(path), "rb") as wf:
        channels = wf.getnchannels()
        src_rate = wf.getframerate()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)

    if wf.getsampwidth() == 2:
        samples = np.array(struct.unpack(f"<{n_frames * channels}h", raw), dtype=np.float32) / 32768.0
    elif wf.getsampwidth() == 4:
        samples = np.array(struct.unpack(f"<{n_frames * channels}i", raw), dtype=np.float32) / 2147483648.0
    else:
        raise ValueError(f"Unsupported WAV sample width: {wf.getsampwidth()} bytes")

    waveform = torch.from_numpy(samples)
    if channels > 1:
        waveform = waveform.view(-1, channels).mean(dim=1)

    if src_rate != sample_rate:
        src_len = waveform.shape[0]
        dst_len = max(1, int(src_len * sample_rate / src_rate))
        waveform = torch.nn.functional.interpolate(
            waveform.view(1, 1, -1),
            size=dst_len,
            mode="linear",
            align_corners=False,
        ).squeeze()

    return waveform


def make_sine_wave(
    frequency: float = 440.0,
    duration_sec: float = 1.0,
    sample_rate: int = 16000,
    amplitude: float = 0.5,
) -> torch.Tensor:
    """Generate a mono sine wave tensor (samples,)."""
    n = int(sample_rate * duration_sec)
    t = torch.arange(n, dtype=torch.float32) / sample_rate
    return amplitude * torch.sin(2 * math.pi * frequency * t)


def load_audio(
    path: str | Path,
    sample_rate: int = 16000,
    max_duration_sec: float = 5.0,
) -> torch.Tensor:
    """
    Load mono audio as (1, samples), padded or truncated to max_duration_sec.

    Supports WAV natively. Other formats are loaded via torchaudio when installed.
    """
    path = Path(path)
    max_samples = int(sample_rate * max_duration_sec)

    if path.suffix.lower() == ".wav":
        waveform = _load_wav_mono(path, sample_rate)
    else:
        try:
            import torchaudio
        except ImportError as exc:
            raise ValueError(
                f"Non-WAV audio ({path.suffix}) requires torchaudio. Install with: pip install torchaudio"
            ) from exc

        waveform, src_rate = torchaudio.load(str(path))
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0)
        else:
            waveform = waveform.squeeze(0)
        if src_rate != sample_rate:
            waveform = torchaudio.functional.resample(waveform, src_rate, sample_rate)

    if waveform.shape[0] > max_samples:
        waveform = waveform[:max_samples]
    elif waveform.shape[0] < max_samples:
        waveform = torch.nn.functional.pad(waveform, (0, max_samples - waveform.shape[0]))

    return waveform.unsqueeze(0)


def save_audio(
    path: str | Path,
    waveform: torch.Tensor,
    sample_rate: int = 16000,
) -> None:
    """Save mono waveform tensor (samples,) or (1, samples) as a WAV file."""
    path = Path(path)
    if waveform.dim() == 2:
        waveform = waveform.squeeze(0)
    samples = (waveform.detach().cpu().clamp(-1, 1) * 32767).short().tolist()
    with wave.open(str(path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(struct.pack(f"<{len(samples)}h", *samples))


def load_audio_batch(
    paths: list[str | Path],
    sample_rate: int = 16000,
    max_duration_sec: float = 5.0,
) -> torch.Tensor:
    """Load multiple audio clips as (B, samples)."""
    clips = [load_audio(p, sample_rate, max_duration_sec).squeeze(0) for p in paths]
    return torch.stack(clips)