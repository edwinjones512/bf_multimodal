#!/usr/bin/env python3
"""
Reset processed corpus outputs so download/process jobs can run cleanly.

Keeps raw downloads by default (resume-friendly). Use --include-downloads to
also remove partial/finished raw archives.

Usage:
    python reset_processing.py
    python reset_processing.py --include-downloads
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from config import DataConfig, GUTENBERG_LEGACY_BULK_FILES, GUTENBERG_ZENODO_SUBSETS

PROCESSED_FILES = (
    "wikipedia.jsonl",
    "gutenberg.jsonl",
    "github.jsonl",
    "librivox.jsonl",
    "corpus.jsonl",
    "wiki_qa.jsonl",
)

LIBRIVOX_INTERMEDIATE_DIRS = ("wav", "chunks", "transcripts")


def remove_path(path: Path) -> bool:
    if not path.exists():
        return False
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Reset processed training data")
    parser.add_argument(
        "--include-downloads",
        action="store_true",
        help="Also delete raw downloads (GitHub zips, LibriVox MP3s, Gutenberg archive, Wikipedia dump)",
    )
    args = parser.parse_args()

    cfg = DataConfig()
    removed: list[str] = []

    for name in PROCESSED_FILES:
        path = cfg.processed_dir / name
        if remove_path(path):
            removed.append(str(path))

    if remove_path(cfg.tokenizer_path):
        removed.append(str(cfg.tokenizer_path))

    for dirname in LIBRIVOX_INTERMEDIATE_DIRS:
        path = cfg.librivox_dir / dirname
        if remove_path(path):
            removed.append(str(path))

    if args.include_downloads:
        download_targets = [
            cfg.github_dir / "raw",
            cfg.github_dir / "repos.jsonl",
            cfg.librivox_dir / "raw",
            cfg.librivox_dir / "books.jsonl",
            cfg.gutenberg_dir / "raw",
            cfg.gutenberg_dir / "pg_catalog.csv",
            cfg.gutenberg_dir / "zenodo_subset.txt",
            *[cfg.gutenberg_dir / name for name in GUTENBERG_LEGACY_BULK_FILES],
            *[cfg.gutenberg_dir / str(info["filename"]) for info in GUTENBERG_ZENODO_SUBSETS.values()],
            *[cfg.gutenberg_dir / f"{info['filename']}.part" for info in GUTENBERG_ZENODO_SUBSETS.values()],
            cfg.wiki_dump_dir,
        ]
        for path in download_targets:
            if remove_path(path):
                removed.append(str(path))

    if not removed:
        print("Nothing to reset — processed outputs already clean.")
        return

    print("Removed:")
    for path in removed:
        print(f"  {path}")
    print("\nRe-run processing once:")
    print("  python process_all.py")
    print("  python process_all.py --whisper-device cuda   # Colab GPU for LibriVox")


if __name__ == "__main__":
    main()