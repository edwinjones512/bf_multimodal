"""Helpers for detecting and counting multimodal corpus records."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any


def record_modalities(record: dict[str, Any]) -> set[str]:
    """Return modality tags present on a corpus record."""
    explicit = record.get("modalities")
    if explicit:
        return set(explicit)

    found: set[str] = set()
    if record.get("audio_path"):
        found.add("audio")
    if record.get("image_path"):
        found.add("image")
    if record.get("video_path"):
        found.add("video")
    if record.get("text"):
        found.add("text")
    return found


def has_media(record: dict[str, Any]) -> bool:
    return bool(record_modalities(record) & {"audio", "image", "video"})


def modality_bucket(record: dict[str, Any]) -> str:
    """Classify a record into a training bucket for the mixed sampler."""
    mods = record_modalities(record)
    if "video" in mods:
        return "video"
    if "image" in mods:
        return "image"
    if "audio" in mods:
        return "audio"
    return "text"


def corpus_modality_counts(records: Iterable[dict[str, Any]]) -> dict[str, int]:
    counts = {"text": 0, "audio": 0, "image": 0, "video": 0, "multimodal_any": 0}
    for record in records:
        bucket = modality_bucket(record)
        counts[bucket] += 1
        if has_media(record):
            counts["multimodal_any"] += 1
    return counts