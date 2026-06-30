#!/usr/bin/env python3
"""
Build MSR-VTT video-caption JSONL for multimodal training.

Usage:
    python process_msrvtt.py
    python process_msrvtt.py --demo   # use test_video.mp4 for all entries
    python process_msrvtt.py --max-records 2000
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from config import DataConfig
from data.corpus import make_record, write_record

VIDEO_PROMPT = "Describe this video."


def load_msrvtt_entries(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict) and "sentences" in data:
        return data["sentences"]
    if isinstance(data, list):
        return data
    raise ValueError(f"Unrecognized MSR-VTT JSON structure in {path}")


def video_id_from_entry(entry: dict) -> str:
    if "video_id" in entry:
        return str(entry["video_id"]).replace(".mp4", "")
    if "video" in entry:
        return str(entry["video"]).replace(".mp4", "")
    if "id" in entry:
        return str(entry["id"])
    return "unknown"


def video_filename_from_entry(entry: dict) -> str:
    if entry.get("video"):
        return str(entry["video"])
    vid = video_id_from_entry(entry)
    return f"{vid}.mp4" if vid != "unknown" else ""


def resolve_video_path(videos_dir: Path, entry: dict) -> Path | None:
    fname = video_filename_from_entry(entry)
    if not fname:
        return None
    direct = videos_dir / fname
    if direct.exists():
        return direct
    matches = list(videos_dir.rglob(fname))
    return matches[0] if matches else None


def caption_from_entry(entry: dict) -> str:
    for key in ("caption", "sentence", "text"):
        if key in entry and entry[key]:
            return str(entry[key]).strip()
    return ""


def main() -> None:
    cfg = DataConfig()
    parser = argparse.ArgumentParser(description="Process MSR-VTT to JSONL")
    parser.add_argument(
        "--annotations",
        type=Path,
        default=cfg.msrvtt_dir / "msrvtt_data.json",
    )
    parser.add_argument("--videos-dir", type=Path, default=cfg.msrvtt_dir)
    parser.add_argument("--output", type=Path, default=cfg.processed_dir / "msrvtt.jsonl")
    parser.add_argument("--demo", action="store_true", help="Use test_video.mp4 for every record")
    parser.add_argument("--max-records", type=int, default=None)
    args = parser.parse_args()

    if args.demo:
        demo_src = Path("test_video.mp4")
        if not demo_src.exists():
            print("test_video.mp4 not found for --demo")
            sys.exit(1)
        args.videos_dir.mkdir(parents=True, exist_ok=True)
        demo_dest = args.videos_dir / "demo.mp4"
        if not demo_dest.exists():
            shutil.copy(demo_src, demo_dest)
        demo_path = demo_dest.as_posix()
    else:
        if not args.annotations.exists():
            print(f"Annotations not found: {args.annotations}")
            print("Run: python download_msrvtt.py  OR  python process_msrvtt.py --demo")
            sys.exit(1)
        entries = load_msrvtt_entries(args.annotations)
        demo_path = None

    written = 0
    skipped = 0
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with open(args.output, "w", encoding="utf-8") as out:
        if args.demo:
            entries = [{"video_id": "demo", "caption": "A sample video scene."}] * 50
        else:
            entries = load_msrvtt_entries(args.annotations)

        for entry in entries:
            if args.max_records and written >= args.max_records:
                break
            vid = video_id_from_entry(entry)
            caption = caption_from_entry(entry)
            if not caption:
                skipped += 1
                continue

            if demo_path:
                video_path = demo_path
            else:
                resolved = resolve_video_path(args.videos_dir, entry)
                if resolved is None:
                    skipped += 1
                    continue
                video_path = resolved.as_posix()

            text = f"{VIDEO_PROMPT} {caption}"
            write_record(
                out,
                make_record(
                    source="msrvtt",
                    text=text,
                    title=vid,
                    record_id=f"{vid}:{written}",
                    language="en",
                    video_path=video_path,
                    caption=caption,
                    modalities=["video", "text"],
                ),
            )
            written += 1

    print(f"Wrote {written:,} records to {args.output} (skipped {skipped:,})")
    print("Next: python merge_corpus.py --shuffle")


if __name__ == "__main__":
    main()