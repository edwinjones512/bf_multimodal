#!/usr/bin/env python3
"""Sync local Colab checkpoints to mounted Google Drive."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from config import PROJECT_ROOT
from utils.drive_sync import (
    colab_local_checkpoint_dir,
    is_colab,
    seed_local_from_drive,
    sync_directory,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sync checkpoint files from Colab local storage to Google Drive"
    )
    parser.add_argument(
        "--local-dir",
        type=Path,
        default=None,
        help="Local checkpoint directory (default: /content/transformer_local/checkpoints on Colab)",
    )
    parser.add_argument(
        "--drive-dir",
        type=Path,
        default=PROJECT_ROOT / "checkpoints",
        help="Mounted Drive checkpoint directory",
    )
    parser.add_argument(
        "--restore",
        action="store_true",
        help="Copy newer Drive files into local storage instead of syncing out",
    )
    parser.add_argument(
        "--watch",
        type=int,
        default=0,
        metavar="SECONDS",
        help="Repeat sync every N seconds (0 = run once)",
    )
    args = parser.parse_args()

    local_dir = args.local_dir or colab_local_checkpoint_dir()
    drive_dir = args.drive_dir

    if not is_colab() and args.local_dir is None:
        print("Not running in Colab; use --local-dir to specify a source directory.")

    print(f"Local : {local_dir}")
    print(f"Drive : {drive_dir}")

    while True:
        if args.restore:
            seed_local_from_drive(local_dir, drive_dir)
        else:
            sync_directory(local_dir, drive_dir)
        if args.watch <= 0:
            break
        time.sleep(args.watch)


if __name__ == "__main__":
    main()