#!/usr/bin/env python3
"""
Train a BPE tokenizer on the merged training corpus (Wikipedia + Gutenberg).

Usage:
    python train_tokenizer.py
    python train_tokenizer.py --vocab-size 32000
"""

import argparse
import sys
from pathlib import Path

from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, trainers, processors

from config import DataConfig, SPECIAL_TOKENS
from data.corpus import iter_jsonl, normalize_legacy_record


def train_tokenizer(
    jsonl_path: Path,
    output_path: Path,
    vocab_size: int = 32000,
) -> Tokenizer:
    tokenizer = Tokenizer(models.BPE())
    tokenizer.normalizer = normalizers.Sequence([
        normalizers.NFKC(),
        normalizers.Lowercase(),
    ])
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)

    special_tokens = list(SPECIAL_TOKENS.keys())
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        show_progress=True,
    )

    def batch_iterator():
        batch = []
        for record in iter_jsonl(jsonl_path):
            record = normalize_legacy_record(record)
            batch.append(record["text"])
            if len(batch) >= 1000:
                yield batch
                batch = []
        if batch:
            yield batch

    tokenizer.train_from_iterator(batch_iterator(), trainer=trainer)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(output_path))
    print(f"Tokenizer saved to {output_path} ({vocab_size} vocab)")
    return tokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Train BPE tokenizer on training corpus")
    parser.add_argument("--input", type=Path, default=DataConfig().corpus_path)
    parser.add_argument("--output", type=Path, default=DataConfig().tokenizer_path)
    parser.add_argument("--vocab-size", type=int, default=32000)
    args = parser.parse_args()

    if not args.input.exists() or args.input.stat().st_size == 0:
        print(f"Corpus not found: {args.input}")
        print("Run: python process_all.py")
        sys.exit(1)

    train_tokenizer(args.input, args.output, args.vocab_size)


if __name__ == "__main__":
    main()