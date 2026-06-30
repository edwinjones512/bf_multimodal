"""Evaluate audio-conditioned transcription quality."""

from __future__ import annotations

from pathlib import Path

import torch
from tokenizers import Tokenizer

from config import SPECIAL_TOKENS
from data.corpus import iter_jsonl
from data.multimodal_dataset import resolve_media_path
from eval.metrics import answer_f1, normalize_text
from inference import decode
from utils.media import load_audio


AUDIO_PROMPT = "Transcribe the following audio:"


def load_audio_eval_set(
    jsonl_path: Path,
    data_root: Path | None = None,
    holdout_book_ids: set[str] | None = None,
) -> list[dict]:
    """Load LibriVox records with resolvable audio paths."""
    data_root = data_root or Path.cwd()
    holdout_book_ids = holdout_book_ids or set()
    items: list[dict] = []

    for record in iter_jsonl(jsonl_path):
        if record.get("source") != "librivox":
            continue
        audio_path = record.get("audio_path")
        if not audio_path:
            continue
        resolved = resolve_media_path(audio_path, data_root)
        if resolved is None:
            continue
        book_id = str(record.get("id", "")).split(":")[0]
        if book_id in holdout_book_ids:
            continue
        items.append({
            "audio_path": resolved,
            "reference": record.get("text", ""),
            "book_id": book_id,
        })
    return items


@torch.no_grad()
def generate_transcription(
    model,
    tokenizer: Tokenizer,
    audio_path: Path,
    device: torch.device,
    sample_rate: int,
    max_duration_sec: float,
    max_tokens: int = 128,
    temperature: float = 0.1,
) -> str:
    ids = tokenizer.encode(AUDIO_PROMPT).ids
    text_ids = torch.tensor([ids], dtype=torch.long, device=device)
    audio = load_audio(audio_path, sample_rate=sample_rate, max_duration_sec=max_duration_sec)
    audio_batch = audio.squeeze(0).unsqueeze(0).to(device)

    output_ids = model.generate(
        text_ids=text_ids,
        audio=audio_batch,
        max_new_tokens=max_tokens,
        temperature=temperature,
        top_k=10,
        eos_token_id=SPECIAL_TOKENS["<eos>"],
    )
    new_ids = output_ids[0].tolist()[len(ids):]
    return decode(tokenizer, new_ids)


def evaluate_audio_caption(
    model,
    tokenizer: Tokenizer,
    eval_items: list[dict],
    device: torch.device,
    sample_rate: int = 16000,
    max_duration_sec: float = 5.0,
    max_samples: int | None = None,
    max_tokens: int = 128,
) -> dict[str, float]:
    model.eval()
    subset = eval_items[:max_samples] if max_samples else eval_items
    if not subset:
        return {"audio_caption_f1": 0.0, "audio_caption_n": 0.0}

    scores: list[float] = []
    for item in subset:
        generated = generate_transcription(
            model,
            tokenizer,
            item["audio_path"],
            device,
            sample_rate,
            max_duration_sec,
            max_tokens=max_tokens,
        )
        scores.append(answer_f1(item["reference"], generated))

    model.train()
    return {
        "audio_caption_f1": sum(scores) / len(scores),
        "audio_caption_n": float(len(scores)),
    }


def holdout_book_ids_from_manifest(manifest_path: Path, fraction: float = 0.1) -> set[str]:
    """Reserve the last fraction of books for audio eval."""
    if not manifest_path.exists():
        return set()
    import json

    books: list[str] = []
    with open(manifest_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                books.append(json.loads(line)["id"])
    n_holdout = max(1, int(len(books) * fraction))
    return set(books[-n_holdout:])