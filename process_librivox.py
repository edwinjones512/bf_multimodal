#!/usr/bin/env python3
"""
Process downloaded LibriVox MP3 sections into JSONL with speech chunks.

Converts MP3 -> 16 kHz mono WAV, transcribes with Whisper word timestamps,
splits into fixed-length chunks, and writes records with audio_path + words.

Usage:
    python process_librivox.py
    python process_librivox.py --max-books 10
    python process_librivox.py --no-transcribe
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from tqdm import tqdm

from config import DataConfig, LIBRIVOX_WHISPER_MODEL, ModelConfig
from data.corpus import make_record, write_record
from utils.audio_convert import chunk_waveform_with_offsets, convert_to_wav, load_wav_mono
from utils.transcribe import (
    align_words_to_chunk,
    get_or_create_section_transcript,
    words_to_text,
)


def load_manifest(path: Path) -> list[dict]:
    books = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                books.append(json.loads(line))
    return books


def section_filename(book_id: str, section_number: str) -> str:
    return f"{book_id}_{section_number.zfill(4)}.mp3"


def build_chunk_text(book: dict, section: dict, chunk_idx: int) -> str:
    book_title = book.get("title", "untitled")
    section_title = section.get("title", "section")
    authors = book.get("authors") or "unknown author"
    if chunk_idx == 0:
        return f"{book_title} by {authors}. {section_title}."
    return f"{book_title}: {section_title} (part {chunk_idx + 1})."


def transcript_cache_path(transcript_dir: Path, book_id: str, section_number: str) -> Path:
    return transcript_dir / book_id / f"{book_id}_{section_number.zfill(4)}.json"


def process_section(
    book: dict,
    section: dict,
    raw_dir: Path,
    wav_dir: Path,
    chunk_dir: Path,
    transcript_dir: Path,
    sample_rate: int,
    chunk_sec: float,
    overlap_sec: float,
    *,
    transcribe: bool = True,
    whisper_model: str = LIBRIVOX_WHISPER_MODEL,
    whisper_device: str | None = None,
    force_transcribe: bool = False,
) -> list[dict]:
    book_id = book["id"]
    section_number = str(section["section_number"])
    mp3_path = raw_dir / book_id / section_filename(book_id, section_number)
    if not mp3_path.exists():
        return []

    wav_path = wav_dir / book_id / f"{book_id}_{section_number.zfill(4)}.wav"
    if not wav_path.exists() or wav_path.stat().st_size == 0:
        convert_to_wav(mp3_path, wav_path, sample_rate=sample_rate)

    section_words: list[dict] = []
    if transcribe:
        cache_path = transcript_cache_path(transcript_dir, book_id, section_number)
        section_words, _ = get_or_create_section_transcript(
            wav_path,
            cache_path,
            model_name=whisper_model,
            language="en",
            device=whisper_device,
            force=force_transcribe,
        )

    waveform, sr = load_wav_mono(wav_path)
    if sr != sample_rate:
        new_len = max(1, int(waveform.numel() * sample_rate / sr))
        waveform = torch.nn.functional.interpolate(
            waveform.view(1, 1, -1),
            size=new_len,
            mode="linear",
            align_corners=False,
        ).squeeze()

    windows = chunk_waveform_with_offsets(waveform, sample_rate, chunk_sec, overlap_sec)
    records: list[dict] = []
    section_id = section.get("id") or section_number

    for chunk_idx, window in enumerate(windows):
        chunk = window["waveform"]
        chunk_name = f"{book_id}_{section_number.zfill(4)}_{chunk_idx:04d}.wav"
        chunk_path = chunk_dir / book_id / chunk_name
        chunk_path.parent.mkdir(parents=True, exist_ok=True)
        save_wav(chunk_path, chunk, sample_rate)

        chunk_words = align_words_to_chunk(
            section_words,
            window["start_sec"],
            window["end_sec"],
        ) if section_words else []

        if chunk_words:
            text = words_to_text(chunk_words)
        else:
            text = build_chunk_text(book, section, chunk_idx)

        rel_audio = chunk_path.as_posix()
        records.append(make_record(
            source="librivox",
            text=text,
            title=book.get("title", "untitled"),
            record_id=f"{book_id}:{section_id}:{chunk_idx}",
            language="en",
            authors=book.get("authors"),
            chunk=chunk_idx,
            audio_path=rel_audio,
            modalities=["audio", "text"],
            words=chunk_words or None,
        ))

    return records


def save_wav(path: Path, waveform: torch.Tensor, sample_rate: int) -> None:
    import struct
    import wave

    samples = (waveform.detach().cpu().clamp(-1, 1) * 32767).short().tolist()
    with wave.open(str(path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(struct.pack(f"<{len(samples)}h", *samples))


def main() -> None:
    model_cfg = ModelConfig()
    parser = argparse.ArgumentParser(description="Process LibriVox MP3s to JSONL")
    parser.add_argument("--manifest", type=Path, default=DataConfig().librivox_dir / "books.jsonl")
    parser.add_argument("--raw-dir", type=Path, default=DataConfig().librivox_dir / "raw")
    parser.add_argument("--wav-dir", type=Path, default=DataConfig().librivox_dir / "wav")
    parser.add_argument("--chunk-dir", type=Path, default=DataConfig().librivox_dir / "chunks")
    parser.add_argument("--transcript-dir", type=Path, default=DataConfig().librivox_dir / "transcripts")
    parser.add_argument("--output", type=Path, default=DataConfig().processed_dir / "librivox.jsonl")
    parser.add_argument("--max-books", type=int, default=None)
    parser.add_argument("--sample-rate", type=int, default=model_cfg.audio_sample_rate)
    parser.add_argument("--chunk-sec", type=float, default=model_cfg.audio_duration_sec)
    parser.add_argument("--overlap-sec", type=float, default=0.5)
    parser.add_argument("--no-transcribe", action="store_true", help="Skip Whisper transcription")
    parser.add_argument("--force-transcribe", action="store_true", help="Re-run Whisper even if cached")
    parser.add_argument("--whisper-model", type=str, default=LIBRIVOX_WHISPER_MODEL)
    parser.add_argument("--whisper-device", choices=["auto", "cuda", "cpu"], default="auto")
    args = parser.parse_args()

    if not args.manifest.exists():
        print(f"No LibriVox manifest found: {args.manifest}")
        print("Run: python download_librivox.py")
        sys.exit(1)

    books = load_manifest(args.manifest)
    if args.max_books:
        books = books[: args.max_books]

    whisper_device = None
    if args.whisper_device != "auto":
        whisper_device = args.whisper_device

    args.output.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with_words = 0

    with open(args.output, "w", encoding="utf-8") as out:
        for book in tqdm(books, desc="Processing books"):
            for section in book.get("sections", []):
                for record in process_section(
                    book,
                    section,
                    args.raw_dir,
                    args.wav_dir,
                    args.chunk_dir,
                    args.transcript_dir,
                    args.sample_rate,
                    args.chunk_sec,
                    args.overlap_sec,
                    transcribe=not args.no_transcribe,
                    whisper_model=args.whisper_model,
                    whisper_device=whisper_device,
                    force_transcribe=args.force_transcribe,
                ):
                    write_record(out, record)
                    written += 1
                    if record.get("words"):
                        with_words += 1

    print(f"Wrote {written:,} records to {args.output}")
    if not args.no_transcribe:
        print(f"  {with_words:,} records include word-level transcripts")
    print("Next: python merge_corpus.py --shuffle")


if __name__ == "__main__":
    main()