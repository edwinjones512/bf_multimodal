#!/usr/bin/env python3
"""
Run all data processing steps once: per-source JSONL, merge, wiki QA, tokenizer.

Skips any step whose output already exists. Use --force to rebuild everything.

Usage:
    python process_all.py
    python process_all.py --whisper-device cuda
    python process_all.py --force
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from config import DataConfig, find_wikipedia_dump, has_gutenberg_archive


def needs_output(path: Path, force: bool) -> bool:
    return force or not path.exists() or path.stat().st_size == 0


def run_step(label: str, cmd: list[str], *, optional: bool = False) -> bool:
    print(f"\n=== {label} ===")
    print(" ".join(cmd))
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        if optional:
            print(f"Warning: {label} failed (exit {result.returncode}) — continuing")
            return False
        print(f"Failed: {label} (exit {result.returncode})")
        sys.exit(result.returncode)
    return True


def has_wikipedia_raw(cfg: DataConfig) -> bool:
    return find_wikipedia_dump(cfg.wiki_dump_dir) is not None


def has_gutenberg_bulk(cfg: DataConfig) -> bool:
    return has_gutenberg_archive(cfg.gutenberg_dir)


def has_gutenberg_raw(cfg: DataConfig) -> bool:
    raw = cfg.gutenberg_dir / "raw"
    return raw.is_dir() and any(raw.iterdir())


def has_github_raw(cfg: DataConfig) -> bool:
    raw = cfg.github_dir / "raw"
    return raw.is_dir() and bool(list(raw.glob("*.zip")))


def has_librivox_raw(cfg: DataConfig) -> bool:
    return (cfg.librivox_dir / "books.jsonl").exists()


def is_complete(cfg: DataConfig) -> bool:
    return cfg.corpus_path.exists() and cfg.tokenizer_path.exists()


def main() -> None:
    parser = argparse.ArgumentParser(description="Process all training data once")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild all outputs even if they already exist",
    )
    parser.add_argument(
        "--whisper-model",
        type=str,
        default=None,
        help="Whisper model for LibriVox (default from config)",
    )
    parser.add_argument(
        "--whisper-device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="Whisper device for LibriVox transcription",
    )
    parser.add_argument("--max-librivox-books", type=int, default=None)
    parser.add_argument(
        "--max-wiki-questions",
        type=int,
        default=10_000,
        help="Cap Wikipedia QA pairs for eval (default: 10000)",
    )
    parser.add_argument(
        "--skip-librivox",
        action="store_true",
        help="Skip LibriVox processing (e.g. when Whisper is unavailable)",
    )
    args = parser.parse_args()

    cfg = DataConfig()
    py = sys.executable

    if is_complete(cfg) and not args.force:
        print(f"Corpus and tokenizer already built:")
        print(f"  {cfg.corpus_path}")
        print(f"  {cfg.tokenizer_path}")
        print("Use --force to reprocess.")
        return

    needs = lambda path: needs_output(path, args.force)

    if needs(cfg.processed_dir / "wikipedia.jsonl"):
        if has_wikipedia_raw(cfg):
            run_step("Wikipedia", [py, "process_wikipedia.py"])
        else:
            print("Skipping Wikipedia (no dump — run download_wikipedia.py)")

    if needs(cfg.processed_dir / "gutenberg.jsonl"):
        if has_gutenberg_bulk(cfg):
            run_step("Gutenberg extract", [py, "extract_gutenberg.py"], optional=True)
            run_step("Gutenberg", [py, "process_gutenberg.py"], optional=True)
        elif has_gutenberg_raw(cfg):
            run_step("Gutenberg", [py, "process_gutenberg.py"])
        else:
            print("Skipping Gutenberg (no archive — run download_gutenberg.py)")

    if needs(cfg.processed_dir / "github.jsonl"):
        if has_github_raw(cfg):
            run_step("GitHub", [py, "process_github.py"])
        else:
            print("Skipping GitHub (no zips — run download_github.py)")

    if needs(cfg.processed_dir / "coco.jsonl"):
        captions = cfg.coco_dir / "annotations" / "captions_train2017.json"
        if captions.exists():
            run_step("COCO", [py, "process_coco.py"], optional=True)
        else:
            print("Skipping COCO (no captions — run download_coco.py)")

    if needs(cfg.processed_dir / "msrvtt.jsonl"):
        ann = cfg.msrvtt_dir / "msrvtt_data.json"
        demo_video = Path("test_video.mp4")
        if ann.exists():
            run_step("MSR-VTT", [py, "process_msrvtt.py"], optional=True)
        elif demo_video.exists():
            run_step("MSR-VTT demo", [py, "process_msrvtt.py", "--demo"], optional=True)
        else:
            print("Skipping MSR-VTT (run download_msrvtt.py or add test_video.mp4)")

    if needs(cfg.processed_dir / "librivox.jsonl"):
        if args.skip_librivox:
            print("Skipping LibriVox (--skip-librivox)")
        elif has_librivox_raw(cfg):
            cmd = [py, "process_librivox.py"]
            if args.whisper_device != "auto":
                cmd.extend(["--whisper-device", args.whisper_device])
            if args.whisper_model:
                cmd.extend(["--whisper-model", args.whisper_model])
            if args.max_librivox_books:
                cmd.extend(["--max-books", str(args.max_librivox_books)])
            if not run_step("LibriVox", cmd, optional=True):
                librivox_out = cfg.processed_dir / "librivox.jsonl"
                if librivox_out.exists() and librivox_out.stat().st_size == 0:
                    librivox_out.unlink()
        else:
            print("Skipping LibriVox (no manifest — run download_librivox.py)")

    source_files = [
        cfg.processed_dir / "wikipedia.jsonl",
        cfg.processed_dir / "gutenberg.jsonl",
        cfg.processed_dir / "github.jsonl",
        cfg.processed_dir / "librivox.jsonl",
        cfg.processed_dir / "coco.jsonl",
        cfg.processed_dir / "msrvtt.jsonl",
    ]
    if not any(p.exists() and p.stat().st_size > 0 for p in source_files):
        print("\nNo processed source files found. Download data first, then re-run.")
        sys.exit(1)

    if needs(cfg.corpus_path):
        run_step("Merge corpus", [py, "merge_corpus.py", "--shuffle"])
    else:
        print(f"\nCorpus already present: {cfg.corpus_path}")

    wiki_qa = cfg.processed_dir / "wiki_qa.jsonl"
    wikipedia_out = cfg.processed_dir / "wikipedia.jsonl"
    if needs(wiki_qa):
        if wikipedia_out.exists() and wikipedia_out.stat().st_size > 0:
            wiki_cmd = [py, "-m", "eval.build_wiki_qa"]
            if args.max_wiki_questions:
                wiki_cmd.extend(["--max-questions", str(args.max_wiki_questions)])
            run_step("Wiki QA eval set", wiki_cmd, optional=True)
        else:
            print("Skipping Wiki QA (no Wikipedia data)")
    else:
        print(f"Wiki QA already present: {wiki_qa}")

    if needs(cfg.tokenizer_path):
        run_step("Tokenizer", [py, "train_tokenizer.py"])
    else:
        print(f"Tokenizer already present: {cfg.tokenizer_path}")

    print("\nProcessing complete.")


if __name__ == "__main__":
    main()