#!/usr/bin/env python3
"""Build Wikipedia QA evaluation set from processed articles."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import DataConfig
from eval.wiki_qa import build_wiki_qa_set, save_qa_set


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Wikipedia QA eval set")
    parser.add_argument("--input", type=Path, default=DataConfig().processed_dir / "wikipedia.jsonl")
    parser.add_argument("--output", type=Path, default=DataConfig().processed_dir / "wiki_qa.jsonl")
    parser.add_argument("--max-questions", type=int, default=None)
    args = parser.parse_args()

    qa_pairs = build_wiki_qa_set(args.input, max_questions=args.max_questions)
    save_qa_set(qa_pairs, args.output)
    print(f"Wrote {len(qa_pairs):,} QA pairs to {args.output}")


if __name__ == "__main__":
    main()