#!/usr/bin/env python3
"""
Download Project Gutenberg ebooks for training.

Default method downloads a Zenodo subset (~134 MB) instead of the 10 GB bulk archive:
https://zenodo.org/records/3360392

Methods:
  zenodo  - Zenodo subset zip with plain .txt files (default)
  catalog - Download individually from pg_catalog.csv

Usage:
    python download_gutenberg.py
    python download_gutenberg.py --subset 184mb
    python download_gutenberg.py --subset 1gb
    python download_gutenberg.py --method catalog --max-books 1000
"""

from __future__ import annotations

import argparse
import csv
import io
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from tqdm import tqdm

from config import (
    GUTENBERG_CATALOG_URL,
    GUTENBERG_DEFAULT_SUBSET,
    GUTENBERG_FILES_BASE,
    GUTENBERG_ZENODO_SUBSETS,
    DataConfig,
    gutenberg_zenodo_url,
    gutenberg_zenodo_path,
)
from utils.gutenberg_archive import (
    remove_legacy_bulk_artifacts,
    validate_gutenberg_zip,
    write_subset_marker,
)


def download_file(
    url: str,
    dest: Path,
    chunk_size: int = 8 * 1024 * 1024,
    max_retries: int = 100,
) -> None:
    """Stream-download with resume support and automatic retry on connection drops."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    temp = dest.with_suffix(dest.suffix + ".part")
    session = requests.Session()
    session.headers["User-Agent"] = "transformer-corpus/1.0 (educational ML project)"

    bar: tqdm | None = None
    attempt = 0

    while attempt < max_retries:
        headers: dict[str, str] = {}
        mode = "wb"
        initial = 0

        if temp.exists():
            initial = temp.stat().st_size
            headers["Range"] = f"bytes={initial}-"
            mode = "ab"
            if attempt == 0:
                print(f"Resuming {dest.name} from {initial / 1e6:.1f} MB")

        try:
            with session.get(url, headers=headers, stream=True, timeout=(30, 120)) as resp:
                if resp.status_code == 416:
                    temp.rename(dest)
                    if bar:
                        bar.close()
                    return
                resp.raise_for_status()

                total = initial
                if resp.headers.get("Content-Length"):
                    total += int(resp.headers["Content-Length"])
                elif resp.headers.get("Content-Range"):
                    total_str = resp.headers["Content-Range"].split("/")[-1]
                    if total_str.isdigit():
                        total = int(total_str)

                if bar is None:
                    bar = tqdm(
                        total=total,
                        initial=initial,
                        unit="B",
                        unit_scale=True,
                        unit_divisor=1024,
                        desc=dest.name,
                    )
                else:
                    bar.total = total
                    bar.refresh()

                with open(temp, mode) as f:
                    for chunk in resp.iter_content(chunk_size=chunk_size):
                        if chunk:
                            f.write(chunk)
                            bar.update(len(chunk))

            temp.rename(dest)
            if bar:
                bar.close()
            return

        except (requests.RequestException, OSError) as exc:
            attempt += 1
            wait = min(60, 2 ** min(attempt, 6))
            print(f"\nConnection error (attempt {attempt}/{max_retries}): {exc}")
            print(f"Retrying in {wait}s...")
            time.sleep(wait)

    if bar:
        bar.close()
    raise RuntimeError(f"Failed to download {dest.name} after {max_retries} attempts")


def fetch_catalog(dest: Path) -> list[dict[str, str]]:
    """Download pg_catalog.csv and return Text ebook entries."""
    print(f"Fetching catalog: {GUTENBERG_CATALOG_URL}")
    resp = requests.get(GUTENBERG_CATALOG_URL, timeout=120)
    resp.raise_for_status()

    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(resp.content)

    reader = csv.DictReader(io.StringIO(resp.text))
    entries = []
    for row in reader:
        if row.get("Type") != "Text":
            continue
        book_id = row["Text#"].strip()
        if not book_id.isdigit():
            continue
        entries.append({
            "id": book_id,
            "title": row.get("Title", "").strip(),
            "language": row.get("Language", "").strip(),
            "authors": row.get("Authors", "").strip(),
        })

    print(f"Catalog: {len(entries):,} text ebooks")
    return entries


def text_url(book_id: str) -> str:
    return f"{GUTENBERG_FILES_BASE}/{book_id}/{book_id}-0.txt"


def download_one(book_id: str, output_dir: Path, session: requests.Session) -> tuple[str, str]:
    """Download a single ebook. Returns (book_id, status)."""
    dest = output_dir / "raw" / f"{book_id}.txt"
    if dest.exists() and dest.stat().st_size > 100:
        return book_id, "skipped"

    url = text_url(book_id)
    try:
        resp = session.get(url, timeout=30)
        if resp.status_code == 404:
            alt = f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt"
            resp = session.get(alt, timeout=30)
        if resp.status_code != 200:
            return book_id, f"http_{resp.status_code}"

        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(resp.content)
        return book_id, "ok"
    except Exception as exc:
        return book_id, f"error:{exc}"


def download_catalog_method(
    output_dir: Path,
    workers: int = 8,
    max_books: int | None = None,
) -> None:
    catalog_path = output_dir / "pg_catalog.csv"
    entries = fetch_catalog(catalog_path)
    if max_books:
        entries = entries[:max_books]

    raw_dir = output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {len(entries):,} ebooks with {workers} workers")
    print(f"Output: {raw_dir.resolve()}")

    session = requests.Session()
    session.headers["User-Agent"] = "transformer-corpus/1.0 (educational ML project)"

    stats = {"ok": 0, "skipped": 0, "failed": 0}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(download_one, e["id"], output_dir, session): e["id"]
            for e in entries
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="Ebooks"):
            _, status = future.result()
            if status == "ok":
                stats["ok"] += 1
            elif status == "skipped":
                stats["skipped"] += 1
            else:
                stats["failed"] += 1

    print(f"Done: {stats['ok']:,} downloaded, {stats['skipped']:,} skipped, {stats['failed']:,} failed")
    print("Next: python process_gutenberg.py")


def download_zenodo_method(
    output_dir: Path,
    subset: str,
    *,
    redownload: bool = False,
) -> None:
    info = GUTENBERG_ZENODO_SUBSETS[subset]
    dest = gutenberg_zenodo_path(output_dir, subset)
    partial = dest.with_suffix(dest.suffix + ".part")

    if redownload:
        for path in (dest, partial):
            if not path.exists():
                continue
            try:
                path.unlink()
                print(f"Removed {path}")
            except OSError as exc:
                print(f"Could not remove {path}: {exc}")
                sys.exit(1)

    if dest.exists():
        ok, msg = validate_gutenberg_zip(dest, subset, verify_crc=False)
        if ok:
            print(f"Zenodo subset already present: {dest}")
            write_subset_marker(output_dir, subset)
            return
        print(f"Existing archive invalid ({msg}); re-downloading...")

    url = gutenberg_zenodo_url(subset)
    print(f"Downloading Zenodo Gutenberg subset: {subset}")
    print(f"  {info['label']}")
    print(f"  URL: {url}")
    download_file(url, dest)

    ok, msg = validate_gutenberg_zip(dest, subset, verify_md5=True)
    if not ok:
        print(f"\nDownload finished but archive validation failed:\n  {msg}")
        sys.exit(1)

    write_subset_marker(output_dir, subset)
    print("Archive validated successfully.")
    print("Next: python extract_gutenberg.py")
    print("  then: python process_gutenberg.py")


def main() -> None:
    if "--method" in sys.argv and "bulk" in sys.argv:
        print(
            "The legacy bulk Gutenberg archive (~10 GB) is no longer supported.\n"
            "Use Zenodo subsets instead:\n"
            "  python download_gutenberg.py --method zenodo --subset 1.7gb"
        )
        sys.exit(1)

    parser = argparse.ArgumentParser(description="Download Project Gutenberg ebooks")
    parser.add_argument("--output-dir", type=Path, default=DataConfig().gutenberg_dir)
    parser.add_argument(
        "--method",
        choices=["zenodo", "catalog"],
        default="zenodo",
        help="zenodo = Zenodo subset zip (default); catalog = per-book download",
    )
    parser.add_argument(
        "--subset",
        choices=tuple(GUTENBERG_ZENODO_SUBSETS.keys()),
        default=GUTENBERG_DEFAULT_SUBSET,
        help=f"Zenodo subset size (default: {GUTENBERG_DEFAULT_SUBSET})",
    )
    parser.add_argument("--workers", type=int, default=8, help="Parallel downloads (catalog method)")
    parser.add_argument("--max-books", type=int, default=None, help="Limit downloads (catalog method)")
    parser.add_argument(
        "--redownload",
        action="store_true",
        help="Delete existing subset zip/partial file and download again",
    )
    args = parser.parse_args()

    removed = remove_legacy_bulk_artifacts(args.output_dir)
    if removed:
        print(f"Removed legacy bulk archive file(s): {', '.join(removed)}")

    print("Project Gutenberg download")
    print(f"  Method : {args.method}")
    if args.method == "zenodo":
        print(f"  Subset : {args.subset} — {GUTENBERG_ZENODO_SUBSETS[args.subset]['label']}")
    print(f"  Output : {args.output_dir.resolve()}")
    print()

    if args.method == "zenodo":
        download_zenodo_method(args.output_dir, args.subset, redownload=args.redownload)
    else:
        download_catalog_method(args.output_dir, args.workers, args.max_books)


if __name__ == "__main__":
    main()