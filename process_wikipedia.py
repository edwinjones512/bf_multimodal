#!/usr/bin/env python3
"""
Parse the Wikipedia XML dump and extract plain-text articles for training.

Streams the bz2-compressed XML (no full decompression to disk required) and
writes one article per line as JSONL.

Usage:
    python process_wikipedia.py
    python process_wikipedia.py --max-articles 10000   # quick test run
"""

import argparse
import bz2
import json
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import mwparserfromhell
from tqdm import tqdm

from config import DataConfig, find_wikipedia_dump
from data.corpus import make_record, write_record


def detect_namespace(tag: str) -> str:
    if tag.startswith("{"):
        return tag.split("}")[0][1:]
    return ""


def wikitext_to_plain(wikitext: str) -> str:
    """Convert wikitext markup to plain text."""
    if not wikitext or wikitext.strip() == "":
        return ""
    try:
        parsed = mwparserfromhell.parse(wikitext)
        text = parsed.strip_code(normalize=True, collapse=True)
    except Exception:
        text = re.sub(r"\{\{[^}]*\}\}", "", wikitext)
        text = re.sub(r"\[\[(?:[^|\]]*\|)?([^\]]+)\]\]", r"\1", text)
        text = re.sub(r"<[^>]+>", "", text)
        text = re.sub(r"''+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def is_valid_article(title: str, ns: str) -> bool:
    if ns != "0":
        return False
    skip_prefixes = (
        "Wikipedia:", "Template:", "Category:", "File:", "Portal:",
        "Help:", "Draft:", "Module:", "MediaWiki:", "Talk:",
    )
    return not any(title.startswith(p) for p in skip_prefixes)


def _find_child(elem: ET.Element, local_name: str, ns_uri: str) -> ET.Element | None:
    return elem.find(f"{{{ns_uri}}}{local_name}")


def stream_articles(dump_path: Path, max_articles: int | None = None):
    """Yield (title, text) tuples from a Wikipedia XML bz2 dump."""
    count = 0
    ns_uri: str | None = None

    with bz2.open(dump_path, "rb") as f:
        context = ET.iterparse(f, events=("start", "end"))
        for event, elem in context:
            if event == "start" and ns_uri is None and elem.tag.endswith("}siteinfo"):
                ns_uri = detect_namespace(elem.tag)
                continue

            if event != "end" or not elem.tag.endswith("}page"):
                continue

            if ns_uri is None:
                ns_uri = detect_namespace(elem.tag)

            title_el = _find_child(elem, "title", ns_uri)
            ns_el = _find_child(elem, "ns", ns_uri)
            text_el = elem.find(f".//{{{ns_uri}}}text")

            title = title_el.text if title_el is not None else ""
            ns = ns_el.text if ns_el is not None else ""
            raw = text_el.text if text_el is not None and text_el.text else ""

            elem.clear()

            if not is_valid_article(title, ns):
                continue

            plain = wikitext_to_plain(raw)
            if len(plain) < 100:
                continue

            yield title, plain
            count += 1
            if max_articles and count >= max_articles:
                return


def main() -> None:
    parser = argparse.ArgumentParser(description="Process Wikipedia dump to training text")
    parser.add_argument("--dump-dir", type=Path, default=DataConfig().wiki_dump_dir)
    parser.add_argument("--output", type=Path, default=DataConfig().processed_dir / "wikipedia.jsonl")
    parser.add_argument("--max-articles", type=int, default=None, help="Limit articles (for testing)")
    args = parser.parse_args()

    dump_path = find_wikipedia_dump(args.dump_dir)
    if dump_path is None:
        print(f"No Wikipedia dump found in {args.dump_dir}")
        print("Run: python download_wikipedia.py --small")
        sys.exit(1)
    print(f"Using dump: {dump_path.name}")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    with open(args.output, "w", encoding="utf-8") as out:
        for title, text in tqdm(stream_articles(dump_path, args.max_articles), desc="Processing"):
            record = make_record(
                source="wikipedia",
                text=text,
                title=title,
                language="en",
            )
            write_record(out, record)
            written += 1

    print(f"Wrote {written:,} articles to {args.output}")
    print("Next: python merge_corpus.py")


if __name__ == "__main__":
    main()