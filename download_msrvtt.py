#!/usr/bin/env python3
"""
Download MSR-VTT annotations and optionally video files.

The Oxford VGG msrvtt.zip mirror is offline (404). This script uses the
friedrichor/MSR-VTT dataset on Hugging Face instead.

Usage:
    python download_msrvtt.py
    python download_msrvtt.py --split train_9k
    python download_msrvtt.py --with-videos   # ~2.2 GB MSRVTT_Videos.zip
"""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

import requests

from config import (
    DataConfig,
    MSRVTT_ANNOTATION_FILES,
    MSRVTT_DEFAULT_SPLIT,
    MSRVTT_HF_BASE,
    MSRVTT_VIDEOS_ZIP,
)


def hf_url(filename: str) -> str:
    return f"{MSRVTT_HF_BASE}/{filename}"


def download_file(url: str, dest: Path, chunk_size: int = 1 << 20) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"Already present: {dest}")
        return

    print(f"Downloading {url}")
    print(f"  -> {dest}")
    with requests.get(url, stream=True, timeout=120) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("content-length") or 0)
        downloaded = 0
        with open(dest, "wb") as f:
            for chunk in resp.iter_content(chunk_size=chunk_size):
                if not chunk:
                    continue
                f.write(chunk)
                downloaded += len(chunk)
                if total and downloaded % (50 * chunk_size) < chunk_size:
                    pct = 100 * downloaded / total
                    print(f"  {pct:.1f}% ({downloaded / 1e9:.2f} GB)", end="\r")
    if total:
        print()
    print(f"Saved {dest.stat().st_size / 1e6:.1f} MB")


def extract_videos_zip(zip_path: Path, dest_dir: Path) -> int:
    dest_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    with zipfile.ZipFile(zip_path, "r") as zf:
        for info in zf.infolist():
            if not info.filename.lower().endswith(".mp4"):
                continue
            # Flatten any subdirectories so process_msrvtt finds {video_id}.mp4
            target = dest_dir / Path(info.filename).name
            if target.exists() and target.stat().st_size > 0:
                count += 1
                continue
            target.write_bytes(zf.read(info))
            count += 1
    return count


def main() -> None:
    cfg = DataConfig()
    parser = argparse.ArgumentParser(description="Download MSR-VTT from Hugging Face")
    parser.add_argument("--msrvtt-dir", type=Path, default=cfg.msrvtt_dir)
    parser.add_argument(
        "--split",
        choices=tuple(MSRVTT_ANNOTATION_FILES.keys()),
        default=MSRVTT_DEFAULT_SPLIT,
        help="Annotation split to download (default: train_7k)",
    )
    parser.add_argument(
        "--with-videos",
        action="store_true",
        help=f"Also download and extract {MSRVTT_VIDEOS_ZIP} (~2.2 GB)",
    )
    parser.add_argument(
        "--annotations-only",
        action="store_true",
        help="Download captions JSON only (default unless --with-videos is passed)",
    )
    args = parser.parse_args()

    args.msrvtt_dir.mkdir(parents=True, exist_ok=True)

    ann_name = MSRVTT_ANNOTATION_FILES[args.split]
    ann_dest = args.msrvtt_dir / ann_name
    download_file(hf_url(ann_name), ann_dest)

    canonical = args.msrvtt_dir / "msrvtt_data.json"
    if not canonical.exists() or canonical.stat().st_mtime < ann_dest.stat().st_mtime:
        canonical.write_text(ann_dest.read_text(encoding="utf-8"), encoding="utf-8")

    print(f"Annotations: {canonical} ({canonical.stat().st_size / 1e6:.1f} MB)")

    if args.with_videos:
        zip_dest = args.msrvtt_dir / MSRVTT_VIDEOS_ZIP
        download_file(hf_url(MSRVTT_VIDEOS_ZIP), zip_dest)
        n_videos = extract_videos_zip(zip_dest, args.msrvtt_dir)
        print(f"Extracted {n_videos:,} .mp4 files to {args.msrvtt_dir}")
    else:
        print("Videos not downloaded (annotations only).")
        print("  Full set: python download_msrvtt.py --with-videos")
        print("  Quick test: python process_msrvtt.py --demo")

    print("Next: python process_msrvtt.py")


if __name__ == "__main__":
    main()