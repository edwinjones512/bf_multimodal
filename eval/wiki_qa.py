"""Build Wikipedia-answerable question/answer pairs from article text."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from data.corpus import iter_jsonl, normalize_legacy_record

SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
IS_PATTERN = re.compile(
    r"\b(the\s+)?([A-Z][A-Za-z0-9'’\- ]{1,60}?)\s+is\s+([^.!?]{8,120}[.!?])",
)


def first_sentence(text: str) -> str:
    parts = SENTENCE_SPLIT.split(text.strip(), maxsplit=1)
    return parts[0].strip() if parts else text.strip()


def answer_after_is(title: str, text: str) -> str | None:
    sentences = SENTENCE_SPLIT.split(text.strip())
    for sentence in sentences:
        if title.lower() in sentence.lower() and " is " in sentence.lower():
            _, _, tail = sentence.lower().partition(" is ")
            idx = sentence.lower().find(" is ")
            return sentence[idx + 4 :].strip()
    return first_sentence(text)


def build_wiki_qa_set(
    wikipedia_path: Path,
    max_questions: int | None = None,
    min_answer_len: int = 10,
) -> list[dict[str, Any]]:
    """Generate factual QA pairs grounded in Wikipedia article text."""
    qa_pairs: list[dict[str, Any]] = []
    seen: set[str] = set()

    for record in iter_jsonl(wikipedia_path):
        record = normalize_legacy_record(record)
        if record.get("source") != "wikipedia":
            continue

        title = record.get("title", "").strip()
        text = record.get("text", "").strip()
        if not title or len(text) < min_answer_len:
            continue

        candidates = [
            {
                "question": f"What is {title}?",
                "answer": first_sentence(text),
                "type": "title_definition",
                "article": title,
            },
            {
                "question": f"Tell me about {title}.",
                "answer": first_sentence(text),
                "type": "title_summary",
                "article": title,
            },
        ]

        after_is = answer_after_is(title, text)
        if after_is and len(after_is) >= min_answer_len:
            candidates.append({
                "question": f"What is {title}?",
                "answer": after_is,
                "type": "title_is",
                "article": title,
            })

        for match in IS_PATTERN.finditer(text):
            subject = match.group(2).strip()
            answer = match.group(3).strip()
            if len(subject) < 3 or len(answer) < min_answer_len:
                continue
            candidates.append({
                "question": f"What is {subject}?",
                "answer": answer,
                "type": "extracted_is",
                "article": title,
            })

        for item in candidates:
            key = (item["question"], normalize_answer_key(item["answer"]))
            if key in seen:
                continue
            seen.add(key)
            qa_pairs.append(item)
            if max_questions and len(qa_pairs) >= max_questions:
                return qa_pairs

    return qa_pairs


def normalize_answer_key(answer: str) -> str:
    return re.sub(r"\s+", " ", answer.lower().strip())


def save_qa_set(qa_pairs: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in qa_pairs:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def load_qa_set(path: Path) -> list[dict[str, Any]]:
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items