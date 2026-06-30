"""Convert audiobook audio files to model-ready WAV using bundled ffmpeg."""

from __future__ import annotations

import subprocess
import wave
from pathlib import Path

import torch


def ffmpeg_executable() -> str:
    try:
        import imageio_ffmpeg
    except ImportError as exc:
        raise RuntimeError(
            "MP3 conversion requires imageio-ffmpeg. Install with: pip install imageio-ffmpeg"
        ) from exc
    return imageio_ffmpeg.get_ffmpeg_exe()


def convert_to_wav(
    input_path: Path,
    output_path: Path,
    sample_rate: int = 16000,
) -> None:
    """Convert an audio file to 16 kHz mono WAV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg_executable(),
        "-y",
        "-i",
        str(input_path),
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        str(output_path),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def load_wav_mono(path: Path) -> tuple[torch.Tensor, int]:
    """Load a mono WAV file as (samples,) float tensor and sample rate."""
    with wave.open(str(path), "rb") as wf:
        sample_rate = wf.getframerate()
        channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)

    if sample_width == 2:
        import numpy as np

        samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    else:
        raise ValueError(f"Unsupported WAV sample width: {sample_width} bytes")

    waveform = torch.from_numpy(samples)
    if channels > 1:
        waveform = waveform.view(-1, channels).mean(dim=1)
    return waveform, sample_rate


def chunk_waveform_with_offsets(
    waveform: torch.Tensor,
    sample_rate: int,
    chunk_sec: float,
    overlap_sec: float = 0.5,
) -> list[dict]:
    """Split a waveform into chunks with start/end timestamps in seconds."""
    chunk_samples = max(1, int(sample_rate * chunk_sec))
    overlap_samples = max(0, int(sample_rate * overlap_sec))

    if waveform.numel() <= chunk_samples:
        return [{
            "waveform": waveform,
            "start_sec": 0.0,
            "end_sec": waveform.numel() / sample_rate,
        }]

    step = max(1, chunk_samples - overlap_samples)
    windows: list[dict] = []
    start = 0
    while start < waveform.numel():
        end = min(start + chunk_samples, waveform.numel())
        chunk = waveform[start:end]
        if chunk.numel() < chunk_samples // 4:
            break
        if chunk.numel() < chunk_samples:
            chunk = torch.nn.functional.pad(chunk, (0, chunk_samples - chunk.numel()))
        windows.append({
            "waveform": chunk,
            "start_sec": start / sample_rate,
            "end_sec": (start + chunk_samples) / sample_rate,
        })
        if end >= waveform.numel():
            break
        start += step
    return windows


def chunk_waveform(
    waveform: torch.Tensor,
    sample_rate: int,
    chunk_sec: float,
    overlap_sec: float = 0.5,
) -> list[torch.Tensor]:
    """Split a waveform into fixed-length chunks with optional overlap."""
    return [
        window["waveform"]
        for window in chunk_waveform_with_offsets(waveform, sample_rate, chunk_sec, overlap_sec)
    ]