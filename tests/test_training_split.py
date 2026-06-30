import json
from pathlib import Path

import torch
from tokenizers import Tokenizer, models, pre_tokenizers, trainers

from data.corpus import make_record, write_record
from data.dataset import CorpusDataset, train_val_split


def _tiny_tokenizer() -> Tokenizer:
    tok = Tokenizer(models.WordLevel(unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    trainer = trainers.WordLevelTrainer(vocab_size=64, special_tokens=["<pad>", "<unk>"])
    tok.train_from_iterator(
        ["paris france city", "python code language", "transformer model network"],
        trainer=trainer,
    )
    return tok


def test_train_val_split_sizes(tmp_path: Path):
    corpus = tmp_path / "corpus.jsonl"
    with open(corpus, "w", encoding="utf-8") as f:
        for i in range(20):
            write_record(f, make_record("wikipedia", f"article text number {i}", f"Title{i}"))

    tok = _tiny_tokenizer()
    ds = CorpusDataset(corpus, tok, lazy=True)
    train_ds, val_ds = train_val_split(ds, val_fraction=0.1, seed=0)
    assert len(train_ds) == 18
    assert len(val_ds) == 2


def test_dataset_lazy_item_shapes(tmp_path: Path):
    corpus = tmp_path / "corpus.jsonl"
    with open(corpus, "w", encoding="utf-8") as f:
        write_record(f, make_record("wikipedia", "short article", "Paris"))

    tok = _tiny_tokenizer()
    ds = CorpusDataset(corpus, tok, max_seq_len=32, lazy=True)
    item = ds[0]
    assert item["text_ids"].shape == (32,)
    assert item["labels"].shape == (32,)