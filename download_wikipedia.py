#!/usr/bin/env python3
"""
Download English Wikipedia text dumps from Wikimedia.

All profiles use pages-articles dumps (current revision only per page).
History dumps (pages-meta-history) are not supported.

Profiles:
  full   — pages-articles-multistream, all articles (~20 GB) + index
  small  — first pages-articles shard (~300 MB, ~41k pages)  [default]
  simple — Simple English pages-articles-multistream (~380 MB)

Dated dumps:
  --date YYYYMMDD downloads every pages-articles file listed in dumpstatus.json
  for that snapshot (recombined archives plus shard files).

Supports resume on interrupted downloads.

Usage:
    python download_wikipedia.py --small
    python download_wikipedia.py --profile full
    python download_wikipedia.py --date 20260601
    python download_wikipedia.py --date 20260601 --recombined-only
    python download_wikipedia.py --list-dates
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
from pathlib import Path

import requests
from tqdm import tqdm

from config import (
    WIKI_DUMP_PROFILES,
    fetch_wiki_dumpstatus,
    find_wikipedia_dump,
    list_pages_articles_files,
    list_wiki_dump_dates,
    resolve_wiki_dump_date,
    wiki_dump_base,
)


def download_file(
    url: str,
    dest: Path,
    chunk_size: int = 8 * 1024 * 1024,
    max_retries: int = 100,
) -> None:
    """Stream-download a file with resume support, retries, and a progress bar."""
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
                print(f"Resuming {dest.name} from {initial / 1e9:.2f} GB")

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
            print(f"Saved {dest} ({dest.stat().st_size / 1e9:.2f} GB)")
            return

        except (requests.RequestException, OSError) as exc:
            attempt += 1
            if bar:
                bar.close()
                bar = None
            if attempt >= max_retries:
                raise
            wait = min(60, 2 ** attempt)
            print(f"Download interrupted ({exc}); retry {attempt}/{max_retries} in {wait}s")
            time.sleep(wait)


def verify_md5(dest: Path, expected_md5: str | None) -> bool:
    if not expected_md5:
        return True
    digest = hashlib.md5()
    with open(dest, "rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    actual = digest.hexdigest()
    if actual != expected_md5:
        print(f"MD5 mismatch for {dest.name}: expected {expected_md5}, got {actual}")
        return False
    print(f"MD5 verified: {dest.name}")
    return True


def profile_files(profile: dict[str, str | None]) -> list[tuple[str, str]]:
    files: list[tuple[str, str]] = []
    articles = profile["articles"]
    if articles:
        files.append(("articles", articles))
    index = profile.get("index")
    if index:
        files.append(("index", index))
    return files


def download_dated_dump(
    wiki: str,
    date: str,
    output_dir: Path,
    *,
    include_shards: bool,
    verify_checksums: bool,
) -> None:
    dumpstatus = fetch_wiki_dumpstatus(wiki, date)
    files = list_pages_articles_files(dumpstatus, include_shards=include_shards)
    if not files:
        print(f"No completed pages-articles files found for {wiki} {date}")
        sys.exit(1)

    base = wiki_dump_base(wiki, date)
    total_gb = sum(size for _, size, _ in files) / 1e9
    print("Wikipedia dated dump download")
    print(f"  Wiki    : {wiki}")
    print(f"  Date    : {date}")
    print(f"  Type    : pages-articles (current revisions only)")
    print(f"  Source  : {base}")
    print(f"  Output  : {output_dir.resolve()}")
    print(f"  Files   : {len(files)} ({total_gb:.1f} GB total)")
    print()

    downloaded = 0
    skipped = 0
    for fname, _size, md5 in files:
        dest = output_dir / fname
        if dest.exists() and dest.stat().st_size > 0:
            print(f"Skipping {fname} (already exists)")
            skipped += 1
            continue
        download_file(f"{base}/{fname}", dest)
        if verify_checksums and not verify_md5(dest, md5):
            dest.unlink(missing_ok=True)
            sys.exit(1)
        downloaded += 1

    print()
    print(f"Finished: {downloaded} downloaded, {skipped} skipped, {len(files)} total")
    print("Next steps:")
    print("  python process_wikipedia.py")
    print("  python process_all.py")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Wikipedia dump")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/wikipedia"),
        help="Directory to store dump files",
    )
    parser.add_argument(
        "--wiki",
        default="enwiki",
        help="Wikimedia wiki name (default: enwiki)",
    )
    parser.add_argument(
        "--date",
        metavar="YYYYMMDD",
        help="Download all pages-articles files for this dump date",
    )
    parser.add_argument(
        "--list-dates",
        action="store_true",
        help="List available dump dates and exit",
    )
    parser.add_argument(
        "--recombined-only",
        action="store_true",
        help="With --date, download only the two recombined multistream files",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify MD5 checksums from dumpstatus.json after each download",
    )
    parser.add_argument(
        "--profile",
        choices=list(WIKI_DUMP_PROFILES.keys()),
        default="small",
        help="Dump size/profile (default: small)",
    )
    parser.add_argument(
        "--small",
        action="store_const",
        const="small",
        dest="profile",
        help="Shortcut for --profile small",
    )
    parser.add_argument(
        "--file",
        choices=["articles", "index", "all"],
        default="all",
        help="Which dump file to download (default: all for profile)",
    )
    args = parser.parse_args()

    if args.list_dates:
        dates = list_wiki_dump_dates(args.wiki)
        print(f"Available {args.wiki} dump dates:")
        for d in dates:
            print(f"  {d}")
        return

    if args.date is not None:
        date = resolve_wiki_dump_date(args.wiki, args.date)
        download_dated_dump(
            args.wiki,
            date,
            args.output_dir,
            include_shards=not args.recombined_only,
            verify_checksums=args.verify,
        )
        return

    profile = WIKI_DUMP_PROFILES[args.profile]
    base = profile["base"]
    manifest = {
        key: f"{base}/{fname}"
        for key, fname in profile_files(profile)
    }
    targets = list(manifest.keys()) if args.file == "all" else [args.file]

    existing = find_wikipedia_dump(args.output_dir)
    if existing and args.file == "all":
        print(f"Wikipedia dump already present: {existing.name}")
        print("Delete it first, use --date, or use reset_processing.py --include-downloads to re-download.")
        return

    print("Wikipedia dump download")
    print(f"  Profile : {args.profile} ({profile['size_hint']})")
    print(f"  Type    : {profile.get('dump_type', 'pages-articles')} — current revisions only")
    print(f"  Source  : {base}")
    print(f"  Output  : {args.output_dir.resolve()}")
    print(f"  Files   : {', '.join(targets)}")
    print()

    for key in targets:
        fname = profile["articles"] if key == "articles" else profile.get("index")
        if not fname:
            continue
        dest = args.output_dir / fname
        if dest.exists():
            print(f"Skipping {fname} (already exists)")
            continue
        download_file(manifest[key], dest)

    print("\nDownload complete. Next steps:")
    print("  python process_all.py")
    print("  python download_gutenberg.py")


if __name__ == "__main__":
    main()