from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset, Subset
from tokenizers import Tokenizer

from data.corpus import iter_jsonl, normalize_legacy_record


class CorpusDataset(Dataset):
    """Tokenized Wikipedia + Gutenberg records for language-model training."""

    def __init__(
        self,
        jsonl_path: Path,
        tokenizer: Tokenizer,
        max_seq_len: int = 1024,
        bos_id: int = 2,
        eos_id: int = 3,
        pad_id: int = 0,
        sources: set[str] | None = None,
        indices: list[int] | None = None,
        lazy: bool = True,
    ):
        self.jsonl_path = jsonl_path
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.pad_id = pad_id
        self.sources = sources
        self.lazy = lazy

        self.records: list[dict] = []
        self.samples: list[list[int]] = []

        for record in iter_jsonl(jsonl_path):
            record = normalize_legacy_record(record)
            if sources and record.get("source") not in sources:
                continue
            self.records.append(record)

        self.active_indices = indices if indices is not None else list(range(len(self.records)))

        if not lazy:
            for idx in self.active_indices:
                self.samples.append(self._encode(self.records[idx]["text"]))

    def _encode(self, text: str) -> list[int]:
        ids = self.tokenizer.encode(text).ids
        return [self.bos_id] + ids[: self.max_seq_len - 2] + [self.eos_id]

    def __len__(self) -> int:
        return len(self.active_indices)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        record_idx = self.active_indices[idx]
        if self.lazy:
            ids = self._encode(self.records[record_idx]["text"])
        else:
            ids = self.samples[idx]

        length = len(ids)
        padded = ids + [self.pad_id] * (self.max_seq_len - length)
        input_ids = torch.tensor(padded[: self.max_seq_len], dtype=torch.long)
        labels = input_ids.clone()
        labels[length:] = self.pad_id

        return {
            "text_ids": input_ids,
            "labels": labels,
            "attention_mask": torch.tensor(
                [1] * min(length, self.max_seq_len) + [0] * max(0, self.max_seq_len - length),
                dtype=torch.long,
            ),
        }


def train_val_split(
    dataset: CorpusDataset,
    val_fraction: float = 0.05,
    seed: int = 42,
) -> tuple[Subset, Subset]:
    n = len(dataset)
    val_size = max(1, int(n * val_fraction))
    train_size = n - val_size

    generator = torch.Generator().manual_seed(seed)
    train_subset, val_subset = torch.utils.data.random_split(
        dataset,
        [train_size, val_size],
        generator=generator,
    )
    return train_subset, val_subset


# Backward-compatible alias
WikipediaDataset = CorpusDataset