"""Evaluation metrics for language-model training and Wikipedia QA."""

from __future__ import annotations

import re

import torch
import torch.nn.functional as F

from config import SPECIAL_TOKENS


def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def answer_f1(expected: str, generated: str) -> float:
    exp_tokens = normalize_text(expected).split()
    gen_tokens = normalize_text(generated).split()
    if not exp_tokens:
        return 0.0
    if not gen_tokens:
        return 0.0

    exp_set = set(exp_tokens)
    gen_set = set(gen_tokens)
    overlap = len(exp_set & gen_set)
    if overlap == 0:
        return 0.0

    precision = overlap / len(gen_set)
    recall = overlap / len(exp_set)
    return 2 * precision * recall / (precision + recall)


def answer_contains(expected: str, generated: str, min_recall: float = 0.5) -> bool:
    exp_tokens = normalize_text(expected).split()
    gen_norm = normalize_text(generated)
    if not exp_tokens:
        return False
    hits = sum(1 for t in exp_tokens if t in gen_norm)
    return hits / len(exp_tokens) >= min_recall


def token_accuracy(logits: torch.Tensor, labels: torch.Tensor, pad_id: int = SPECIAL_TOKENS["<pad>"]) -> float:
    """Fraction of non-padding next-token predictions that are correct."""
    seq_len = logits.shape[1]
    text_len = min(labels.shape[1], seq_len)
    aligned_labels = torch.full(
        (labels.shape[0], seq_len),
        pad_id,
        dtype=labels.dtype,
        device=labels.device,
    )
    text_start = seq_len - text_len
    aligned_labels[:, text_start:] = labels[:, :text_len]

    preds = logits[:, :-1, :].argmax(dim=-1)
    targets = aligned_labels[:, 1:]
    mask = targets != pad_id
    if mask.sum() == 0:
        return 0.0
    correct = (preds == targets) & mask
    return correct.sum().item() / mask.sum().item()


@torch.no_grad()
def validation_loss(
    model,
    loader,
    device: torch.device,
    amp: bool = False,
    amp_dtype: torch.dtype = torch.float16,
) -> tuple[float, float]:
    """Return (average loss, token accuracy) over a validation loader."""
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    batches = 0

    autocast = torch.autocast(
        device_type=device.type,
        enabled=amp and device.type == "cuda",
        dtype=amp_dtype,
    )

    for batch in loader:
        text_ids = batch["text_ids"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        kwargs: dict = {}
        if batch.get("audio") is not None:
            kwargs["audio"] = batch["audio"].to(device, non_blocking=True)
        if batch.get("images") is not None:
            kwargs["images"] = batch["images"].to(device, non_blocking=True)
        if batch.get("video") is not None:
            kwargs["video_frames"] = batch["video"].to(device, non_blocking=True)

        with autocast:
            out = model(text_ids=text_ids, labels=labels, **kwargs)

        total_loss += out["loss"].item()
        total_acc += token_accuracy(out["logits"], labels)
        batches += 1

    model.train()
    if batches == 0:
        return 0.0, 0.0
    return total_loss / batches, total_acc / batches