#!/usr/bin/env python3
"""Wait for Zenodo Gutenberg download, then run the full training pipeline."""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

from config import DataConfig, GUTENBERG_DEFAULT_SUBSET, GUTENBERG_ZENODO_SUBSETS, gutenberg_zenodo_path
from utils.gutenberg_archive import validate_gutenberg_zip

POLL_SECONDS = 30


def download_complete(gutenberg_dir: Path, subset: str = GUTENBERG_DEFAULT_SUBSET) -> bool:
    archive = gutenberg_zenodo_path(gutenberg_dir, subset)
    if not archive.exists():
        return False
    ok, _ = validate_gutenberg_zip(archive, subset, verify_crc=False)
    return ok


def wait_for_download(gutenberg_dir: Path, subset: str = GUTENBERG_DEFAULT_SUBSET) -> None:
    archive = gutenberg_zenodo_path(gutenberg_dir, subset)
    partial = archive.with_suffix(archive.suffix + ".part")
    expected = int(GUTENBERG_ZENODO_SUBSETS[subset]["bytes"])
    print(f"Waiting for Gutenberg download: {archive.name} ({subset})")
    last_size = -1
    while not download_complete(gutenberg_dir, subset):
        size = partial.stat().st_size if partial.exists() else 0
        if size != last_size:
            pct = 100 * size / expected
            print(f"  {size / 1e6:.1f} MB / {expected / 1e6:.0f} MB ({pct:.1f}%)")
            last_size = size
        time.sleep(POLL_SECONDS)
    print(f"Download complete: {archive}")


def run_step(cmd: list[str]) -> None:
    print(f"\n>>> {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def main() -> None:
    cfg = DataConfig()
    if not download_complete(cfg.gutenberg_dir):
        wait_for_download(cfg.gutenberg_dir)

    run_step([sys.executable, "process_all.py"])
    run_step([
        sys.executable, "train.py", "--device", "cuda",
        "--epochs", "3", "--num-workers", "0",
    ])

    print("\nPipeline complete.")


if __name__ == "__main__":
    main()