#!/usr/bin/env python3
"""
Process downloaded Gutenberg ebooks into unified JSONL training format.

Reads from data/gutenberg/raw/*.txt (populated by extract_gutenberg.py).

Usage:
    python download_gutenberg.py         # Zenodo subset (~134 MB default)
    python extract_gutenberg.py          # zip -> raw/*.txt
    python process_gutenberg.py
    python process_gutenberg.py --from-archive   # auto-extract if raw/ is empty
    python process_gutenberg.py --max-books 1000
"""

from __future__ import annotations

import argparse
import csv
import io
import re
import sys
from pathlib import Path

from tqdm import tqdm

from config import DataConfig, GUTENBERG_CATALOG_URL
from data.corpus import make_record, write_record
from utils.gutenberg_archive import count_raw_txt, ensure_gutenberg_raw

START_RE = re.compile(
    r"\*\*\*\s*START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*",
    re.IGNORECASE | re.DOTALL,
)
END_RE = re.compile(
    r"\*\*\*\s*END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*",
    re.IGNORECASE | re.DOTALL,
)


def strip_gutenberg_boilerplate(text: str) -> str:
    start = START_RE.search(text)
    if start:
        text = text[start.end():]
    end = END_RE.search(text)
    if end:
        text = text[: end.start()]
    return re.sub(r"\s+", " ", text).strip()


def chunk_text(text: str, chunk_size: int = 2000, overlap: int = 200) -> list[str]:
    if len(text) <= chunk_size:
        return [text]

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            boundary = text.rfind(". ", start, end)
            if boundary > start + chunk_size // 2:
                end = boundary + 1
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)

    return chunks


def load_catalog_metadata(catalog_path: Path) -> dict[str, dict[str, str]]:
    if not catalog_path.exists():
        import requests
        resp = requests.get(GUTENBERG_CATALOG_URL, timeout=120)
        resp.raise_for_status()
        catalog_path.parent.mkdir(parents=True, exist_ok=True)
        catalog_path.write_bytes(resp.content)

    meta: dict[str, dict[str, str]] = {}
    reader = csv.DictReader(io.StringIO(catalog_path.read_text(encoding="utf-8")))
    for row in reader:
        if row.get("Type") != "Text":
            continue
        book_id = row["Text#"].strip()
        meta[book_id] = {
            "title": row.get("Title", "").strip(),
            "language": row.get("Language", "").strip(),
            "authors": row.get("Authors", "").strip(),
        }
    return meta


def process_file(
    book_id: str,
    text: str,
    meta: dict[str, dict[str, str]],
    min_chars: int,
    chunk_size: int,
) -> list[dict]:
    info = meta.get(book_id, {})
    cleaned = strip_gutenberg_boilerplate(text)
    if len(cleaned) < min_chars:
        return []

    title = info.get("title") or f"Gutenberg #{book_id}"
    chunks = chunk_text(cleaned, chunk_size=chunk_size)
    return [
        make_record(
            source="gutenberg",
            text=chunk,
            title=title,
            record_id=book_id,
            language=info.get("language") or None,
            authors=info.get("authors") or None,
            chunk=i,
        )
        for i, chunk in enumerate(chunks)
    ]


def process_raw_dir(
    raw_dir: Path,
    out_handle,
    meta: dict[str, dict[str, str]],
    min_chars: int,
    chunk_size: int,
    max_books: int | None,
) -> int:
    files = sorted(raw_dir.glob("*.txt"))
    if max_books:
        files = files[:max_books]

    written = 0
    for path in tqdm(files, desc="Gutenberg raw"):
        book_id = path.stem
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for record in process_file(book_id, text, meta, min_chars, chunk_size):
            write_record(out_handle, record)
            written += 1
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Process Gutenberg ebooks to JSONL")
    parser.add_argument("--gutenberg-dir", type=Path, default=DataConfig().gutenberg_dir)
    parser.add_argument("--output", type=Path, default=DataConfig().processed_dir / "gutenberg.jsonl")
    parser.add_argument(
        "--from-archive",
        action="store_true",
        help="Extract Zenodo subset zip to raw/ first if needed",
    )
    parser.add_argument("--from-bulk", action="store_true", help=argparse.SUPPRESS)  # legacy alias
    parser.add_argument("--min-chars", type=int, default=200)
    parser.add_argument("--chunk-size", type=int, default=2000)
    parser.add_argument("--max-books", type=int, default=None)
    parser.add_argument("--force-extract", action="store_true", help="Re-run extraction before processing")
    args = parser.parse_args()

    catalog_path = args.gutenberg_dir / "pg_catalog.csv"
    meta = load_catalog_metadata(catalog_path)

    from_archive = args.from_archive or args.from_bulk
    raw_dir = args.gutenberg_dir / "raw"

    if count_raw_txt(raw_dir) == 0:
        from utils.gutenberg_archive import find_gutenberg_archive

        if find_gutenberg_archive(args.gutenberg_dir) or from_archive:
            if find_gutenberg_archive(args.gutenberg_dir) is None:
                print("No Zenodo subset zip found.")
                print("Run: python download_gutenberg.py")
                sys.exit(1)
            print("raw/ is empty — extracting Zenodo subset first...")
            try:
                raw_dir = ensure_gutenberg_raw(
                    args.gutenberg_dir,
                    force=args.force_extract,
                    max_files=args.max_books,
                )
            except (ValueError, RuntimeError) as exc:
                print(exc)
                sys.exit(1)
        else:
            print("No Gutenberg raw/*.txt files found.")
            print("Run: python download_gutenberg.py && python extract_gutenberg.py")
            print("  or: python process_gutenberg.py --from-archive")
            sys.exit(1)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as out:
        written = process_raw_dir(
            raw_dir, out, meta, args.min_chars, args.chunk_size, args.max_books
        )

    print(f"Wrote {written:,} records to {args.output}")
    print("Next: python merge_corpus.py")


if __name__ == "__main__":
    main()