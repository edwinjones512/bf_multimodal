#!/usr/bin/env python3
"""
Process downloaded GitHub repo archives into unified JSONL training format.

Extracts Python, JavaScript, TypeScript, Kotlin, and Swift source files from zip archives.

Usage:
    python process_github.py
    python process_github.py --max-repos 10
"""

from __future__ import annotations

import argparse
import re
import sys
import zipfile
from pathlib import Path

from tqdm import tqdm

from config import DataConfig, GITHUB_CODE_EXTENSIONS, GITHUB_SKIP_DIRS
from data.corpus import make_record, write_record

# Reuse chunking logic aligned with Gutenberg processor
def chunk_text(text: str, chunk_size: int = 2000, overlap: int = 200) -> list[str]:
    if len(text) <= chunk_size:
        return [text]
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            boundary = text.rfind("\n", start, end)
            if boundary > start + chunk_size // 2:
                end = boundary + 1
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks


def language_from_extension(ext: str) -> str | None:
    ext = ext.lower()
    if ext in {".py"}:
        return "python"
    if ext in {".js", ".jsx", ".mjs", ".cjs"}:
        return "javascript"
    if ext in {".ts", ".tsx"}:
        return "typescript"
    if ext in {".kt", ".kts"}:
        return "kotlin"
    if ext in {".swift"}:
        return "swift"
    return None


def should_skip_path(path: str) -> bool:
    parts = Path(path).parts
    return any(part in GITHUB_SKIP_DIRS or part.startswith(".") for part in parts[:-1])


def is_probably_text(data: bytes) -> bool:
    if b"\x00" in data[:8000]:
        return False
    try:
        data[:8000].decode("utf-8")
        return True
    except UnicodeDecodeError:
        return False


def parse_repo_name_from_zip(zip_name: str) -> str:
    stem = Path(zip_name).stem
    if "_" in stem:
        owner, repo = stem.split("_", 1)
        return f"{owner}/{repo}"
    return stem


def process_zip(
    zip_path: Path,
    min_chars: int,
    max_file_bytes: int,
    chunk_size: int,
) -> list[dict]:
    records: list[dict] = []
    repo_name = parse_repo_name_from_zip(zip_path.name)

    try:
        zf = zipfile.ZipFile(zip_path, "r")
    except zipfile.BadZipFile:
        return records

    with zf:
        for info in zf.infolist():
            if info.is_dir():
                continue

            inner = info.filename
            if should_skip_path(inner):
                continue

            suffix = Path(inner).suffix.lower()
            if suffix not in GITHUB_CODE_EXTENSIONS:
                continue

            lang = language_from_extension(suffix)
            if not lang:
                continue

            if info.file_size > max_file_bytes:
                continue

            try:
                raw = zf.read(info)
            except (RuntimeError, zipfile.BadZipFile):
                continue

            if not is_probably_text(raw):
                continue

            text = raw.decode("utf-8", errors="replace")
            text = re.sub(r"\r\n?", "\n", text).strip()
            if len(text) < min_chars:
                continue

            rel_path = inner.split("/", 1)[-1] if "/" in inner else inner
            file_id = f"{repo_name}:{rel_path}"

            for i, chunk in enumerate(chunk_text(text, chunk_size=chunk_size)):
                records.append(make_record(
                    source="github",
                    text=chunk,
                    title=repo_name,
                    record_id=file_id,
                    language=lang,
                    chunk=i,
                ))

    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Process GitHub zips to JSONL")
    parser.add_argument("--input-dir", type=Path, default=DataConfig().github_dir / "raw")
    parser.add_argument("--output", type=Path, default=DataConfig().processed_dir / "github.jsonl")
    parser.add_argument("--min-chars", type=int, default=100)
    parser.add_argument("--max-file-bytes", type=int, default=512_000)
    parser.add_argument("--chunk-size", type=int, default=2000)
    parser.add_argument("--max-repos", type=int, default=None)
    args = parser.parse_args()

    if not args.input_dir.exists():
        print(f"No GitHub data found: {args.input_dir}")
        print("Run: python download_github.py")
        sys.exit(1)

    zips = sorted(args.input_dir.glob("*.zip"))
    if args.max_repos:
        zips = zips[: args.max_repos]

    if not zips:
        print(f"No zip archives in {args.input_dir}")
        sys.exit(1)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    written = 0

    with open(args.output, "w", encoding="utf-8") as out:
        for zip_path in tqdm(zips, desc="Processing repos"):
            for record in process_zip(
                zip_path, args.min_chars, args.max_file_bytes, args.chunk_size
            ):
                write_record(out, record)
                written += 1

    print(f"Wrote {written:,} records from {len(zips):,} repos to {args.output}")
    print("Next: python merge_corpus.py --shuffle")


if __name__ == "__main__":
    main()