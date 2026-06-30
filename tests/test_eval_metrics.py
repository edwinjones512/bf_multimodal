from eval.metrics import answer_contains, answer_f1, normalize_text, token_accuracy
import torch

from model.transformer import _sample_next_token


def test_normalize_text():
    assert normalize_text("Hello, World!") == "hello world"


def test_answer_f1_exact_overlap():
    assert answer_f1("Paris is the capital", "paris is the capital of france") > 0.5


def test_answer_contains():
    assert answer_contains("Paris France", "the city of paris in france is lovely")
    assert not answer_contains("Berlin Germany", "paris france")


def test_sample_next_token_handles_nan_logits():
    logits = torch.tensor([[float("nan"), 1.0, 2.0]])
    token = _sample_next_token(logits, temperature=0.1, top_k=10)
    assert token.shape == (1, 1)
    assert token.item() == 2


def test_token_accuracy_ignores_padding():
    logits = torch.zeros(1, 4, 5)
    logits[0, 0, 1] = 10
    logits[0, 1, 2] = 10
    labels = torch.tensor([[0, 1, 2, 0]])
    acc = token_accuracy(logits, labels, pad_id=0)
    assert 0.0 <= acc <= 1.0