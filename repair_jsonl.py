#!/usr/bin/env python3
"""Remove invalid lines from a JSONL file."""
import argparse
import json
import sys
from pathlib import Path


def repair(path: Path, output: Path | None = None) -> int:
    out_path = output or path
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    removed = 0
    kept = 0
    with open(path, encoding="utf-8", errors="replace") as src, open(
        tmp, "w", encoding="utf-8"
    ) as dst:
        for i, line in enumerate(src, 1):
            stripped = line.strip()
            if not stripped or "\x00" in stripped:
                removed += 1
                print(f"  drop line {i}: empty or null bytes", flush=True)
                continue
            try:
                json.loads(stripped)
            except json.JSONDecodeError as e:
                removed += 1
                print(f"  drop line {i}: {e}", flush=True)
                continue
            dst.write(stripped + "\n")
            kept += 1
    tmp.replace(out_path)
    print(f"Repaired {path}: kept {kept:,}, removed {removed:,} -> {out_path}")
    return removed


def main() -> None:
    parser = argparse.ArgumentParser(description="Repair corrupt JSONL files")
    parser.add_argument("path", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    removed = repair(args.path, args.output)
    sys.exit(1 if removed else 0)


if __name__ == "__main__":
    main()