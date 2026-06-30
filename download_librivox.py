#!/usr/bin/env python3
"""
Download English audiobooks from LibriVox.

Uses the LibriVox API to list books, then downloads section MP3 files from
Archive.org. Supports resume by skipping existing files.

Usage:
    python download_librivox.py
    python download_librivox.py --max-books 50
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import requests
from tqdm import tqdm

from config import LIBRIVOX_API_BASE, LIBRIVOX_DEFAULT_LANGUAGE, DataConfig

PAGE_SIZE = 50
REQUEST_DELAY = 0.5


def strip_html(text: str) -> str:
    return re.sub(r"<[^>]+>", "", text or "").strip()


def is_english(language: str | None) -> bool:
    lang = (language or "").strip().lower()
    return lang == "english" or lang.startswith("english")


def format_authors(authors: list[dict] | None) -> str:
    if not authors:
        return ""
    names = []
    for author in authors:
        first = (author.get("first_name") or "").strip()
        last = (author.get("last_name") or "").strip()
        name = " ".join(part for part in (first, last) if part)
        if name:
            names.append(name)
    return "; ".join(names)


def fetch_english_books(
    session: requests.Session,
    max_books: int,
    language: str,
) -> list[dict]:
    books: list[dict] = []
    offset = 0

    while len(books) < max_books:
        params = {
            "limit": PAGE_SIZE,
            "offset": offset,
            "extended": 1,
            "format": "json",
            "language": language,
        }
        resp = session.get(LIBRIVOX_API_BASE, params=params, timeout=60)
        resp.raise_for_status()
        items = resp.json().get("books", [])
        if not items:
            break

        for item in items:
            if not is_english(item.get("language")):
                continue
            sections = item.get("sections") or []
            books.append({
                "id": str(item["id"]),
                "title": item.get("title", "untitled"),
                "description": strip_html(item.get("description", "")),
                "language": item.get("language", language),
                "authors": format_authors(item.get("authors")),
                "num_sections": len(sections),
                "url_librivox": item.get("url_librivox", ""),
                "url_text_source": item.get("url_text_source", ""),
                "sections": [
                    {
                        "id": str(section.get("id", "")),
                        "section_number": str(section.get("section_number", "")),
                        "title": section.get("title", "").strip(),
                        "listen_url": section.get("listen_url", ""),
                        "playtime": section.get("playtime", ""),
                    }
                    for section in sections
                    if section.get("listen_url")
                ],
            })
            if len(books) >= max_books:
                break

        if len(items) < PAGE_SIZE:
            break
        offset += PAGE_SIZE
        time.sleep(REQUEST_DELAY)

    return books


def section_filename(book_id: str, section_number: str) -> str:
    safe_section = section_number.zfill(4)
    return f"{book_id}_{safe_section}.mp3"


def download_section(
    session: requests.Session,
    url: str,
    dest: Path,
) -> bool:
    if dest.exists() and dest.stat().st_size > 1000:
        return False

    dest.parent.mkdir(parents=True, exist_ok=True)
    with session.get(url, stream=True, timeout=120) as resp:
        resp.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1024 * 256):
                if chunk:
                    f.write(chunk)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Download English LibriVox audiobooks")
    parser.add_argument("--output-dir", type=Path, default=DataConfig().librivox_dir / "raw")
    parser.add_argument("--manifest", type=Path, default=DataConfig().librivox_dir / "books.jsonl")
    parser.add_argument("--max-books", type=int, default=100, help="Maximum English books to download")
    parser.add_argument(
        "--language",
        type=str,
        default=LIBRIVOX_DEFAULT_LANGUAGE,
        help="LibriVox language filter (default: English)",
    )
    parser.add_argument(
        "--max-sections",
        type=int,
        default=None,
        help="Optional cap on sections per book",
    )
    args = parser.parse_args()

    session = requests.Session()
    session.headers["User-Agent"] = "transformer-corpus/1.0 (educational ML project)"

    print("LibriVox audiobook download")
    print(f"  Language  : {args.language}")
    print(f"  Max books : {args.max_books}")
    print(f"  Output    : {args.output_dir.resolve()}")
    print()

    print("Fetching catalog...")
    books = fetch_english_books(session, args.max_books, args.language)
    print(f"  Found {len(books):,} English books")

    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    with open(args.manifest, "w", encoding="utf-8") as f:
        for book in books:
            f.write(json.dumps(book, ensure_ascii=False) + "\n")

    downloaded = 0
    skipped = 0
    failed = 0

    for book in tqdm(books, desc="Downloading books"):
        book_id = book["id"]
        sections = book["sections"]
        if args.max_sections:
            sections = sections[: args.max_sections]

        for section in sections:
            url = section["listen_url"]
            dest = args.output_dir / book_id / section_filename(book_id, section["section_number"])
            try:
                if download_section(session, url, dest):
                    downloaded += 1
                else:
                    skipped += 1
            except requests.RequestException:
                failed += 1
            time.sleep(0.2)

    print(f"\nDone: {downloaded:,} downloaded, {skipped:,} skipped, {failed:,} failed")
    print(f"Manifest: {args.manifest}")
    print("Next: python process_librivox.py")


if __name__ == "__main__":
    main()