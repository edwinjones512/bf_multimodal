#!/usr/bin/env python3
"""
Download COCO 2017 captions and optionally a subset of training images.

Usage:
    python download_coco.py
    python download_coco.py --max-images 1000
"""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

import requests

from config import DataConfig

ANNOTATIONS_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
TRAIN_IMAGES_URL = "http://images.cocodataset.org/zips/train2017.zip"


def download_file(url: str, dest: Path, chunk_size: int = 1 << 20) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"Already present: {dest}")
        return
    print(f"Downloading {url} -> {dest}")
    with requests.get(url, stream=True, timeout=120) as resp:
        resp.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in resp.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)


def extract_zip(zip_path: Path, dest_dir: Path) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest_dir)


def main() -> None:
    cfg = DataConfig()
    parser = argparse.ArgumentParser(description="Download COCO captions and images")
    parser.add_argument("--coco-dir", type=Path, default=cfg.coco_dir)
    parser.add_argument("--max-images", type=int, default=None, help="Limit train images to download")
    parser.add_argument("--skip-images", action="store_true", help="Only download annotations")
    args = parser.parse_args()

    args.coco_dir.mkdir(parents=True, exist_ok=True)
    ann_zip = args.coco_dir / "annotations_trainval2017.zip"
    download_file(ANNOTATIONS_URL, ann_zip)
    extract_zip(ann_zip, args.coco_dir)

    captions_path = args.coco_dir / "annotations" / "captions_train2017.json"
    if not captions_path.exists():
        raise FileNotFoundError(f"Expected captions at {captions_path}")

    if args.skip_images:
        print(f"Annotations ready at {captions_path}")
        print("Next: python process_coco.py (images optional if paths missing)")
        return

    if args.max_images:
        with open(captions_path, encoding="utf-8") as f:
            data = json.load(f)
        image_ids = sorted({img["id"] for img in data["images"]})[: args.max_images]
        images_dir = args.coco_dir / "train2017"
        images_dir.mkdir(parents=True, exist_ok=True)
        missing = [iid for iid in image_ids if not (images_dir / f"{iid:012d}.jpg").exists()]
        print(f"Need {len(missing)} images (of {len(image_ids)} requested)")
        for iid in missing:
            fname = f"{iid:012d}.jpg"
            url = f"http://images.cocodataset.org/train2017/{fname}"
            download_file(url, images_dir / fname)
    else:
        img_zip = args.coco_dir / "train2017.zip"
        download_file(TRAIN_IMAGES_URL, img_zip)
        extract_zip(img_zip, args.coco_dir)

    print(f"COCO ready under {args.coco_dir}")
    print("Next: python process_coco.py")


if __name__ == "__main__":
    main()