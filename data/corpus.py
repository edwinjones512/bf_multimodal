"""Unified training corpus schema shared by Wikipedia and Gutenberg."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator


CORPUS_FIELDS = (
    "source", "id", "title", "language", "authors", "chunk", "text",
    "audio_path", "image_path", "video_path", "caption", "modalities", "words",
)


def make_record(
    source: str,
    text: str,
    title: str,
    *,
    record_id: str | None = None,
    language: str | None = None,
    authors: str | None = None,
    chunk: int = 0,
    audio_path: str | None = None,
    image_path: str | None = None,
    video_path: str | None = None,
    caption: str | None = None,
    modalities: list[str] | None = None,
    words: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    record = {
        "source": source,
        "id": record_id,
        "title": title,
        "language": language,
        "authors": authors,
        "chunk": chunk,
        "text": text,
    }
    if audio_path:
        record["audio_path"] = audio_path
    if image_path:
        record["image_path"] = image_path
    if video_path:
        record["video_path"] = video_path
    if caption:
        record["caption"] = caption
    if modalities:
        record["modalities"] = modalities
    if words:
        record["words"] = words
    return record


def write_record(handle, record: dict[str, Any]) -> None:
    handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def iter_jsonl(path: Path, *, skip_invalid: bool = False) -> Iterator[dict[str, Any]]:
    with open(path, encoding="utf-8", errors="replace") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line or "\x00" in line:
                if skip_invalid:
                    continue
                if not line:
                    continue
                raise ValueError(f"{path}:{line_no}: line contains null bytes")
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                if skip_invalid:
                    continue
                raise json.JSONDecodeError(
                    f"{path}:{line_no}: {e.msg}", e.doc, e.pos
                ) from e


def normalize_legacy_record(record: dict[str, Any]) -> dict[str, Any]:
    """Upgrade old {title, text} records to the unified schema."""
    if "source" in record:
        return record
    return make_record(
        source="wikipedia",
        text=record["text"],
        title=record.get("title", "untitled"),
        language="en",
    )