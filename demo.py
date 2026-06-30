#!/usr/bin/env python3
"""
Quick demo pipeline — trains a small model on sample text without downloading Wikipedia.

Usage:
    python demo.py
"""

import subprocess
import sys
from pathlib import Path

from data.corpus import make_record, write_record

SAMPLE_WIKI = [
    ("Paris", "Paris is the capital and most populous city of France. Situated on the River Seine, it is a major European cultural and economic center known for art, fashion, and cuisine."),
    ("Python", "Python is a high-level programming language known for its readability and versatility. It supports multiple programming paradigms and has a large standard library."),
    ("Transformer", "The transformer is a neural network architecture that uses self-attention mechanisms. It was introduced in the paper Attention Is All You Need and forms the basis of modern language models."),
] * 100

SAMPLE_GUTENBERG = [
    ("1", "The Declaration of Independence", "en", "Jefferson, Thomas", "We hold these truths to be self-evident, that all men are created equal, that they are endowed by their Creator with certain unalienable Rights."),
    ("1342", "Pride and Prejudice", "en", "Austen, Jane", "It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want of a wife."),
    ("11", "Alice's Adventures in Wonderland", "en", "Carroll, Lewis", "Alice was beginning to get very tired of sitting by her sister on the bank, and of having nothing to do."),
] * 100


def main() -> None:
    data_dir = Path("data/processed")
    data_dir.mkdir(parents=True, exist_ok=True)

    wiki_path = data_dir / "wikipedia.jsonl"
    with open(wiki_path, "w", encoding="utf-8") as f:
        for title, text in SAMPLE_WIKI:
            write_record(f, make_record("wikipedia", text, title, language="en"))

    gutenberg_path = data_dir / "gutenberg.jsonl"
    with open(gutenberg_path, "w", encoding="utf-8") as f:
        for book_id, title, lang, authors, text in SAMPLE_GUTENBERG:
            write_record(
                f,
                make_record("gutenberg", text, title, record_id=book_id, language=lang, authors=authors),
            )

    print(f"Created sample Wikipedia: {wiki_path} ({len(SAMPLE_WIKI)} records)")
    print(f"Created sample Gutenberg: {gutenberg_path} ({len(SAMPLE_GUTENBERG)} records)")

    steps = [
        [sys.executable, "merge_corpus.py", "--shuffle"],
        [sys.executable, "train_tokenizer.py", "--vocab-size", "1000"],
        [sys.executable, "train.py", "--epochs", "1", "--batch-size", "4", "--max-steps", "5"],
    ]

    for cmd in steps:
        print(f"\n>>> {' '.join(cmd)}")
        subprocess.run(cmd, check=True)

    print("\nDemo complete. Try inference:")
    print('  python inference.py --prompt "Paris is the capital of"')


if __name__ == "__main__":
    main()