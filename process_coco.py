#!/usr/bin/env python3
"""
Build COCO image-caption JSONL for multimodal training.

Usage:
    python process_coco.py
    python process_coco.py --max-records 5000
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from config import DataConfig
from data.corpus import make_record, write_record

IMAGE_PROMPT = "Describe this image."


def main() -> None:
    cfg = DataConfig()
    parser = argparse.ArgumentParser(description="Process COCO captions to JSONL")
    parser.add_argument(
        "--captions",
        type=Path,
        default=cfg.coco_dir / "annotations" / "captions_train2017.json",
    )
    parser.add_argument("--images-dir", type=Path, default=cfg.coco_dir / "train2017")
    parser.add_argument("--output", type=Path, default=cfg.processed_dir / "coco.jsonl")
    parser.add_argument("--max-records", type=int, default=None)
    args = parser.parse_args()

    if not args.captions.exists():
        print(f"Captions not found: {args.captions}")
        print("Run: python download_coco.py")
        sys.exit(1)

    with open(args.captions, encoding="utf-8") as f:
        data = json.load(f)

    image_by_id = {img["id"]: img for img in data["images"]}
    written = 0
    skipped = 0
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with open(args.output, "w", encoding="utf-8") as out:
        for ann in data["annotations"]:
            if args.max_records and written >= args.max_records:
                break
            image = image_by_id.get(ann["image_id"])
            if not image:
                continue
            file_name = image["file_name"]
            image_path = args.images_dir / file_name
            if not image_path.exists():
                skipped += 1
                continue
            caption = ann["caption"].strip()
            text = f"{IMAGE_PROMPT} {caption}"
            write_record(
                out,
                make_record(
                    source="coco",
                    text=text,
                    title=file_name,
                    record_id=str(ann["id"]),
                    language="en",
                    image_path=image_path.as_posix(),
                    caption=caption,
                    modalities=["image", "text"],
                ),
            )
            written += 1

    print(f"Wrote {written:,} records to {args.output} (skipped {skipped:,} missing images)")
    print("Next: python merge_corpus.py --shuffle")


if __name__ == "__main__":
    main()