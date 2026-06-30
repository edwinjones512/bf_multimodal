"""Word-level speech transcription utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def words_to_text(words: list[dict[str, Any]]) -> str:
    return " ".join(w["word"] for w in words if w.get("word")).strip()


def align_words_to_chunk(
    words: list[dict[str, Any]],
    chunk_start_sec: float,
    chunk_end_sec: float,
) -> list[dict[str, Any]]:
    """Select words overlapping a chunk and shift timestamps to chunk-relative seconds."""
    aligned: list[dict[str, Any]] = []
    chunk_dur = max(chunk_end_sec - chunk_start_sec, 1e-6)

    for word in words:
        token = (word.get("word") or "").strip()
        if not token:
            continue

        start = float(word["start"])
        end = float(word["end"])
        overlap = min(end, chunk_end_sec) - max(start, chunk_start_sec)
        if overlap <= 0:
            continue

        word_dur = max(end - start, 1e-6)
        if overlap / word_dur < 0.25 and overlap < 0.05:
            continue

        rel_start = max(0.0, start - chunk_start_sec)
        rel_end = min(chunk_dur, end - chunk_start_sec)
        if rel_end <= rel_start:
            continue

        aligned.append({
            "word": token,
            "start": round(rel_start, 3),
            "end": round(rel_end, 3),
        })

    return aligned


def load_transcript_cache(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def save_transcript_cache(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


_WHISPER_MODELS: dict[tuple[str, str], Any] = {}


def _load_whisper_model(model_name: str, device: str):
    key = (model_name, device)
    if key not in _WHISPER_MODELS:
        import whisper
        _WHISPER_MODELS[key] = whisper.load_model(model_name, device=device)
    return _WHISPER_MODELS[key]


def transcribe_words(
    audio_path: Path,
    *,
    model_name: str = "base",
    language: str = "en",
    device: str | None = None,
) -> tuple[list[dict[str, Any]], str]:
    """
    Transcribe audio with Whisper and return word-level timestamps.

    Timestamps are seconds from the start of the audio file.
    """
    if device is None:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = _load_whisper_model(model_name, device)
    result = model.transcribe(
        str(audio_path),
        language=language,
        word_timestamps=True,
        verbose=False,
    )

    words: list[dict[str, Any]] = []
    for segment in result.get("segments", []):
        for word in segment.get("words") or []:
            token = (word.get("word") or "").strip()
            if not token:
                continue
            words.append({
                "word": token,
                "start": round(float(word["start"]), 3),
                "end": round(float(word["end"]), 3),
            })

    full_text = (result.get("text") or words_to_text(words)).strip()
    return words, full_text


def get_or_create_section_transcript(
    wav_path: Path,
    cache_path: Path,
    *,
    model_name: str = "base",
    language: str = "en",
    device: str | None = None,
    force: bool = False,
) -> tuple[list[dict[str, Any]], str]:
    if not force:
        cached = load_transcript_cache(cache_path)
        if cached and cached.get("words"):
            return cached["words"], cached.get("text", words_to_text(cached["words"]))

    words, text = transcribe_words(
        wav_path,
        model_name=model_name,
        language=language,
        device=device,
    )
    save_transcript_cache(cache_path, {
        "audio_path": wav_path.as_posix(),
        "text": text,
        "words": words,
    })
    return words, text