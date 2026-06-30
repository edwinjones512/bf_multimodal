"""Batch collation for multimodal training."""

from __future__ import annotations

import torch


def _stack_optional(tensors: list[torch.Tensor | None]) -> torch.Tensor | None:
    present = [t for t in tensors if t is not None]
    if not present:
        return None
    if len(present) != len(tensors):
        # Drive I/O errors can drop media for some samples; train as text-only.
        return None
    return torch.stack(present)


def collate_multimodal(batch: list[dict], pad_id: int = 0) -> dict:
    """Collate a homogeneous-modality batch from MultimodalDataset items."""
    text_ids = torch.stack([item["text_ids"] for item in batch])
    labels = torch.stack([item["labels"] for item in batch])
    attention_mask = torch.stack([item["attention_mask"] for item in batch])

    result: dict = {
        "text_ids": text_ids,
        "labels": labels,
        "attention_mask": attention_mask,
        "bucket": batch[0].get("bucket", "text"),
    }

    for key in ("audio", "images", "video", "speech_target"):
        if key not in batch[0]:
            continue
        stacked = _stack_optional([item.get(key) for item in batch])
        if stacked is not None:
            result[key] = stacked

    return result