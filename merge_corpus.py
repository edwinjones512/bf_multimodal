#!/usr/bin/env python3
"""
Merge Wikipedia, Gutenberg, GitHub, LibriVox, COCO, and MSR-VTT into corpus.jsonl.

Usage:
    python merge_corpus.py
    python merge_corpus.py --shuffle
    python merge_corpus.py --sources librivox,coco
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

from config import DataConfig
from data.corpus import iter_jsonl, normalize_legacy_record, write_record
from data.modality import corpus_modality_counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge all sources into corpus.jsonl")
    parser.add_argument("--wikipedia", type=Path, default=DataConfig().processed_dir / "wikipedia.jsonl")
    parser.add_argument("--gutenberg", type=Path, default=DataConfig().processed_dir / "gutenberg.jsonl")
    parser.add_argument("--github", type=Path, default=DataConfig().processed_dir / "github.jsonl")
    parser.add_argument("--librivox", type=Path, default=DataConfig().processed_dir / "librivox.jsonl")
    parser.add_argument("--coco", type=Path, default=DataConfig().processed_dir / "coco.jsonl")
    parser.add_argument("--msrvtt", type=Path, default=DataConfig().processed_dir / "msrvtt.jsonl")
    parser.add_argument("--output", type=Path, default=DataConfig().corpus_path)
    parser.add_argument("--shuffle", action="store_true", help="Shuffle merged records")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--sources",
        type=str,
        default=None,
        help="Comma-separated subset of sources to merge (e.g. librivox,coco)",
    )
    args = parser.parse_args()

    all_sources: list[tuple[str, Path]] = [
        ("wikipedia", args.wikipedia),
        ("gutenberg", args.gutenberg),
        ("github", args.github),
        ("librivox", args.librivox),
        ("coco", args.coco),
        ("msrvtt", args.msrvtt),
    ]
    if args.sources:
        wanted = {s.strip() for s in args.sources.split(",")}
        all_sources = [(n, p) for n, p in all_sources if n in wanted]

    sources: list[tuple[str, Path]] = []
    for name, path in all_sources:
        if path.exists() and path.stat().st_size > 0:
            sources.append((name, path))

    if not sources:
        print("No processed data found. Run:")
        print("  python process_wikipedia.py")
        print("  python process_gutenberg.py")
        print("  python process_github.py")
        print("  python process_librivox.py")
        print("  python process_coco.py")
        print("  python process_msrvtt.py")
        sys.exit(1)

    records = []
    counts: dict[str, int] = {}

    for name, path in sources:
        count = 0
        for record in iter_jsonl(path):
            records.append(normalize_legacy_record(record))
            count += 1
        counts[name] = count
        print(f"  {name}: {count:,} records from {path}")

    if args.shuffle:
        rng = random.Random(args.seed)
        rng.shuffle(records)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as out:
        for record in records:
            write_record(out, record)

    mod_counts = corpus_modality_counts(records)
    print(f"Merged {len(records):,} records -> {args.output}")
    for name, count in counts.items():
        pct = 100 * count / len(records)
        print(f"  {name}: {count:,} ({pct:.1f}%)")
    print("Modality breakdown:")
    for key in ("text", "audio", "image", "video", "multimodal_any"):
        print(f"  {key}: {mod_counts[key]:,}")
    print("Next: python train_tokenizer.py")


if __name__ == "__main__":
    main()