"""Run Wikipedia QA evaluation against a trained model."""

from __future__ import annotations

from pathlib import Path

import torch
from tokenizers import Tokenizer

from config import SPECIAL_TOKENS
from eval.metrics import answer_contains, answer_f1, normalize_text
from eval.wiki_qa import load_qa_set
from inference import decode, encode_text_prompt, load_model


@torch.no_grad()
def generate_answer(
    model,
    tokenizer: Tokenizer,
    question: str,
    device: torch.device,
    max_tokens: int = 64,
    temperature: float = 0.1,
) -> str:
    prompt_ids = encode_text_prompt(tokenizer, question)
    text_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    output_ids = model.generate(
        text_ids=text_ids,
        max_new_tokens=max_tokens,
        temperature=temperature,
        top_k=10,
        eos_token_id=SPECIAL_TOKENS["<eos>"],
    )
    new_ids = output_ids[0].tolist()[len(prompt_ids):]
    return decode(tokenizer, new_ids)


def evaluate_qa(
    model,
    tokenizer: Tokenizer,
    qa_pairs: list[dict],
    device: torch.device,
    max_samples: int | None = None,
    max_tokens: int = 64,
) -> dict[str, float]:
    """Score model on Wikipedia-grounded QA pairs."""
    model.eval()
    subset = qa_pairs[:max_samples] if max_samples else qa_pairs

    f1_scores: list[float] = []
    contains_hits = 0

    for item in subset:
        generated = generate_answer(
            model, tokenizer, item["question"], device, max_tokens=max_tokens
        )
        expected = item["answer"]
        f1_scores.append(answer_f1(expected, generated))
        if answer_contains(expected, generated):
            contains_hits += 1
        if device.type == "cuda":
            torch.cuda.empty_cache()

    n = max(len(subset), 1)
    return {
        "qa_count": len(subset),
        "qa_f1": sum(f1_scores) / n,
        "qa_contains_acc": contains_hits / n,
    }


def evaluate_checkpoint(
    checkpoint: Path,
    qa_path: Path,
    tokenizer_path: Path,
    device: torch.device | None = None,
    max_samples: int | None = None,
) -> dict[str, float]:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    model = load_model(checkpoint, device)
    qa_pairs = load_qa_set(qa_path)
    return evaluate_qa(model, tokenizer, qa_pairs, device, max_samples=max_samples)