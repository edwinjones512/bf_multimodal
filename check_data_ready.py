#!/usr/bin/env python3
"""Print download/processing status and exit 0 only when ready to train."""

from __future__ import annotations

import sys
from pathlib import Path

from config import (
    DataConfig,
    GUTENBERG_LEGACY_BULK_FILES,
    GUTENBERG_ZENODO_SUBSETS,
    find_wikipedia_dump,
    has_gutenberg_archive,
    read_gutenberg_subset,
)
from utils.gutenberg_archive import count_raw_txt, validate_gutenberg_zip
from data.corpus import iter_jsonl
from data.modality import corpus_modality_counts


def main() -> None:
    cfg = DataConfig()
    print("Data pipeline status\n")

    wiki = find_wikipedia_dump(cfg.wiki_dump_dir)
    print(f"  Wikipedia dump     : {wiki.name if wiki else 'missing'}")

    legacy_bulk = [name for name in GUTENBERG_LEGACY_BULK_FILES if (cfg.gutenberg_dir / name).exists()]
    if legacy_bulk:
        print(f"  Gutenberg legacy   : stale bulk file(s) {legacy_bulk} — run remove_legacy_bulk_artifacts or re-run download cell")

    raw_n = count_raw_txt(cfg.gutenberg_dir / "raw")
    subset = read_gutenberg_subset(cfg.gutenberg_dir)
    if subset:
        info = GUTENBERG_ZENODO_SUBSETS[subset]
        archive = cfg.gutenberg_dir / str(info["filename"])
        partial = archive.with_suffix(archive.suffix + ".part")
        if archive.exists():
            ok, msg = validate_gutenberg_zip(archive, subset, verify_crc=False)
            status = "OK" if ok else f"incomplete — {msg}"
            print(
                f"  Gutenberg subset   : {subset} / {archive.name} "
                f"({archive.stat().st_size / 1e6:.0f} MB, {status})"
            )
        elif partial.exists():
            expected_mb = int(info["bytes"]) / 1e6
            print(
                f"  Gutenberg subset   : downloading {subset} "
                f"({partial.stat().st_size / 1e6:.0f} / {expected_mb:.0f} MB)"
            )
        else:
            print(f"  Gutenberg subset   : {subset} configured but zip missing")
    else:
        print("  Gutenberg subset   : missing (run download_gutenberg.py)")
    print(f"  Gutenberg raw txt  : {raw_n:,} files")

    github_raw = cfg.github_dir / "raw"
    n_zips = len(list(github_raw.glob("*.zip"))) if github_raw.is_dir() else 0
    print(f"  GitHub zips        : {n_zips}")

    librivox_manifest = cfg.librivox_dir / "books.jsonl"
    print(f"  LibriVox manifest  : {'yes' if librivox_manifest.exists() else 'missing'}")
    print(f"  COCO captions      : {'yes' if (cfg.coco_dir / 'annotations' / 'captions_train2017.json').exists() else 'missing'}")
    msrvtt_ann = (
        (cfg.msrvtt_dir / "msrvtt_data.json").exists()
        or any((cfg.msrvtt_dir / name).exists() for name in ("msrvtt_train_7k.json", "msrvtt_train_9k.json", "msrvtt_test_1k.json"))
    )
    n_msrvtt_videos = len(list(cfg.msrvtt_dir.glob("video*.mp4"))) if cfg.msrvtt_dir.is_dir() else 0
    print(f"  MSR-VTT annotations: {'yes' if msrvtt_ann else 'missing (run download_msrvtt.py)'}")
    print(f"  MSR-VTT videos      : {n_msrvtt_videos:,} mp4 files")

    print()
    for name, path in (
        ("wikipedia.jsonl", cfg.processed_dir / "wikipedia.jsonl"),
        ("gutenberg.jsonl", cfg.processed_dir / "gutenberg.jsonl"),
        ("github.jsonl", cfg.processed_dir / "github.jsonl"),
        ("librivox.jsonl", cfg.processed_dir / "librivox.jsonl"),
        ("coco.jsonl", cfg.processed_dir / "coco.jsonl"),
        ("msrvtt.jsonl", cfg.processed_dir / "msrvtt.jsonl"),
        ("corpus.jsonl", cfg.corpus_path),
        ("tokenizer.json", cfg.tokenizer_path),
        ("wiki_qa.jsonl", cfg.processed_dir / "wiki_qa.jsonl"),
    ):
        if path.exists() and path.stat().st_size > 0:
            print(f"  {name:18s} OK ({path.stat().st_size / 1e6:.1f} MB)")
        else:
            print(f"  {name:18s} —")

    if cfg.corpus_path.exists() and cfg.corpus_path.stat().st_size > 0:
        counts = corpus_modality_counts(iter_jsonl(cfg.corpus_path))
        print()
        print("Corpus modality breakdown:")
        for key in ("text", "audio", "image", "video", "multimodal_any"):
            print(f"  {key:16s} {counts[key]:,}")

    print()
    if cfg.corpus_path.exists() and cfg.tokenizer_path.exists():
        print("Ready to train.")
        print("  python train.py --device cuda --multimodal-fraction 0.25")
        return

    print("Not ready to train. Run download cells, then:")
    print("  python process_all.py --whisper-device cuda --whisper-model base --max-librivox-books 25")
    sys.exit(1)


if __name__ == "__main__":
    main()