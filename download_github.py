#!/usr/bin/env python3
"""
Download public GitHub repositories for Python, JavaScript, TypeScript, Kotlin, and Swift.

Uses the GitHub Search API to find popular repos, then downloads each as a zip
archive. Supports resume and GITHUB_TOKEN for higher rate limits.

Usage:
    python download_github.py
    python download_github.py --max-repos 50 --languages python javascript
    set GITHUB_TOKEN=ghp_...   # optional, recommended
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import requests
from tqdm import tqdm

from config import GITHUB_API_BASE, GITHUB_DEFAULT_LANGUAGES, GITHUB_DEFAULT_MAX_REPOS, DataConfig

PER_PAGE = 100
SEARCH_DELAY = 2.0


def load_env_file() -> None:
    env_path = Path(__file__).parent / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def github_session() -> requests.Session:
    session = requests.Session()
    session.headers.update({
        "Accept": "application/vnd.github+json",
        "User-Agent": "transformer-corpus/1.0",
        "X-GitHub-Api-Version": "2022-11-28",
    })
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        session.headers["Authorization"] = f"Bearer {token}"
    return session


def wait_for_rate_limit(resp: requests.Response) -> None:
    remaining = resp.headers.get("X-RateLimit-Remaining")
    if remaining == "0":
        reset = int(resp.headers.get("X-RateLimit-Reset", 0))
        sleep_for = max(reset - time.time(), 1)
        print(f"Rate limit hit, sleeping {sleep_for:.0f}s...")
        time.sleep(sleep_for)


def search_repos(
    session: requests.Session,
    language: str,
    max_repos: int,
    min_stars: int,
) -> list[dict]:
    repos: list[dict] = []
    page = 1

    while len(repos) < max_repos:
        q = f"language:{language} stars:>={min_stars}"
        url = f"{GITHUB_API_BASE}/search/repositories"
        params = {"q": q, "sort": "stars", "order": "desc", "per_page": PER_PAGE, "page": page}

        resp = session.get(url, params=params, timeout=30)
        if resp.status_code == 403:
            wait_for_rate_limit(resp)
            continue
        resp.raise_for_status()

        items = resp.json().get("items", [])
        if not items:
            break

        for item in items:
            repos.append({
                "full_name": item["full_name"],
                "language": item.get("language") or language,
                "stars": item.get("stargazers_count", 0),
                "default_branch": item.get("default_branch", "main"),
                "html_url": item.get("html_url", ""),
            })
            if len(repos) >= max_repos:
                break

        page += 1
        if page > 10:  # GitHub search cap: 1000 results
            break
        time.sleep(SEARCH_DELAY)

    return repos


def download_repo_zip(
    session: requests.Session,
    repo: dict,
    output_dir: Path,
) -> bool:
    owner, name = repo["full_name"].split("/", 1)
    dest = output_dir / f"{owner}_{name}.zip"
    if dest.exists() and dest.stat().st_size > 1000:
        return False

    branch = repo.get("default_branch", "main")
    url = f"https://api.github.com/repos/{owner}/{name}/zipball/{branch}"

    with session.get(url, stream=True, timeout=120, allow_redirects=True) as resp:
        if resp.status_code == 404:
            for fallback in ("main", "master"):
                if fallback == branch:
                    continue
                url = f"https://api.github.com/repos/{owner}/{name}/zipball/{fallback}"
                resp = session.get(url, stream=True, timeout=120, allow_redirects=True)
                if resp.status_code == 200:
                    break
        if resp.status_code != 200:
            return False
        resp.raise_for_status()

        output_dir.mkdir(parents=True, exist_ok=True)
        with open(dest, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)

    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Download GitHub repos (JS/TS/Python)")
    parser.add_argument("--output-dir", type=Path, default=DataConfig().github_dir / "raw")
    parser.add_argument("--manifest", type=Path, default=DataConfig().github_dir / "repos.jsonl")
    parser.add_argument(
        "--languages",
        nargs="+",
        default=GITHUB_DEFAULT_LANGUAGES,
        help="Languages to search (default: python javascript typescript kotlin swift)",
    )
    parser.add_argument(
        "--max-repos",
        type=int,
        default=GITHUB_DEFAULT_MAX_REPOS,
        help="Max repos per language",
    )
    parser.add_argument("--min-stars", type=int, default=100, help="Minimum star count")
    parser.add_argument("--workers", type=int, default=1, help="Sequential only (GitHub rate limits)")
    args = parser.parse_args()
    load_env_file()

    session = github_session()
    has_token = "Authorization" in session.headers
    print("GitHub repo download")
    print(f"  Languages : {', '.join(args.languages)}")
    print(f"  Max repos : {args.max_repos} per language")
    print(f"  Output    : {args.output_dir.resolve()}")
    print(f"  Token     : {'yes' if has_token else 'no (set GITHUB_TOKEN for faster downloads)'}")
    print()

    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    all_repos: list[dict] = []

    for lang in args.languages:
        print(f"Searching {lang} repositories...")
        repos = search_repos(session, lang, args.max_repos, args.min_stars)
        print(f"  Found {len(repos):,} repos")
        for repo in repos:
            if repo["full_name"] not in seen:
                seen.add(repo["full_name"])
                all_repos.append(repo)

    with open(args.manifest, "w", encoding="utf-8") as f:
        for repo in all_repos:
            f.write(json.dumps(repo) + "\n")

    downloaded = 0
    skipped = 0
    failed = 0

    for repo in tqdm(all_repos, desc="Downloading repos"):
        try:
            if download_repo_zip(session, repo, args.output_dir):
                downloaded += 1
            else:
                skipped += 1
        except requests.RequestException:
            failed += 1
        time.sleep(0.5 if has_token else 1.5)

    print(f"\nDone: {downloaded:,} downloaded, {skipped:,} skipped, {failed:,} failed")
    print(f"Manifest: {args.manifest}")
    print("Next: python process_github.py")


if __name__ == "__main__":
    main()