import json
from pathlib import Path

from eval.wiki_qa import build_wiki_qa_set, save_qa_set, load_qa_set
from data.corpus import make_record, write_record


def test_build_wiki_qa_from_articles(tmp_path: Path):
    wiki_path = tmp_path / "wikipedia.jsonl"
    with open(wiki_path, "w", encoding="utf-8") as f:
        write_record(
            f,
            make_record(
                "wikipedia",
                "Paris is the capital and largest city of France. It is known for the Eiffel Tower.",
                "Paris",
                language="en",
            ),
        )

    qa = build_wiki_qa_set(wiki_path)
    assert len(qa) >= 2
    assert any(item["question"] == "What is Paris?" for item in qa)
    assert any("capital" in item["answer"].lower() for item in qa)


def test_save_and_load_qa(tmp_path: Path):
    qa = [{"question": "What is Python?", "answer": "A programming language.", "type": "test"}]
    path = tmp_path / "qa.jsonl"
    save_qa_set(qa, path)
    loaded = load_qa_set(path)
    assert loaded == qa