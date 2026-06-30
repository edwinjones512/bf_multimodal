import struct
import wave
from pathlib import Path

import torch
from tokenizers import Tokenizer, models, pre_tokenizers, trainers

from config import ModelConfig
from data.collate import collate_multimodal
from data.corpus import make_record, write_record
from data.modality import corpus_modality_counts, modality_bucket
from data.multimodal_dataset import MultimodalDataset, resolve_media_path, safe_path_exists, subset_dataset
from data.sampler import MixedBatchSampler
from model import MultimodalModel


def _tiny_tokenizer() -> Tokenizer:
    tok = Tokenizer(models.WordLevel(unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    trainer = trainers.WordLevelTrainer(
        vocab_size=128,
        special_tokens=["<pad>", "<unk>", "<bos>", "<eos>"],
    )
    tok.train_from_iterator(["describe image cat", "transcribe audio hello"], trainer=trainer)
    return tok


def _write_wav(path: Path, duration_sec: float = 0.25, sample_rate: int = 16000) -> None:
    n = int(sample_rate * duration_sec)
    samples = [0] * n
    with wave.open(str(path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(struct.pack(f"<{n}h", *samples))


def _write_corpus(tmp_path: Path, audio_path: str | None = None, image_path: str | None = None) -> Path:
    corpus = tmp_path / "corpus.jsonl"
    with open(corpus, "w", encoding="utf-8") as f:
        write_record(f, make_record("wikipedia", "paris is the capital", "Paris"))
        if audio_path:
            write_record(
                f,
                make_record(
                    "librivox",
                    "hello world transcript",
                    "Book",
                    audio_path=audio_path,
                    modalities=["audio", "text"],
                ),
            )
        if image_path:
            write_record(
                f,
                make_record(
                    "coco",
                    "Describe this image. a red box",
                    "img.jpg",
                    image_path=image_path,
                    modalities=["image", "text"],
                ),
            )
    return corpus


def test_safe_path_exists_handles_oserror(monkeypatch, tmp_path: Path):
    path = tmp_path / "broken.jpg"

    def boom(_self, follow_symlinks=True):
        raise OSError(5, "Input/output error")

    monkeypatch.setattr(type(path), "exists", boom)
    assert safe_path_exists(path) is False


def test_resolve_media_path_returns_none_on_oserror(monkeypatch, tmp_path: Path):
    path = tmp_path / "data" / "images" / "coco.jpg"
    path.parent.mkdir(parents=True)

    def boom(_self, follow_symlinks=True):
        raise OSError(5, "Input/output error")

    monkeypatch.setattr(type(path), "exists", boom)
    assert resolve_media_path("data/images/coco.jpg", data_root=tmp_path) is None


def test_multimodal_dataset_skips_unreadable_image(tmp_path: Path, monkeypatch):
    corpus = _write_corpus(tmp_path, image_path="data/images/coco.jpg")
    tok = _tiny_tokenizer()
    ds = MultimodalDataset(corpus, tok, max_seq_len=32, data_root=tmp_path)
    record = ds.records[ds.active_indices[1]]

    def broken_image(self, rec):
        if rec is record:
            return None
        return MultimodalDataset._load_image(self, rec)

    monkeypatch.setattr(MultimodalDataset, "_load_image", broken_image)
    item = ds[1]
    assert item["bucket"] == "text"
    assert "images" not in item


def test_modality_bucket():
    assert modality_bucket({"text": "x"}) == "text"
    assert modality_bucket({"text": "x", "audio_path": "a.wav"}) == "audio"
    assert modality_bucket({"text": "x", "image_path": "a.jpg"}) == "image"
    assert modality_bucket({"text": "x", "video_path": "a.mp4"}) == "video"


def test_multimodal_dataset_audio(tmp_path: Path):
    wav = tmp_path / "clip.wav"
    _write_wav(wav)
    corpus = _write_corpus(tmp_path, audio_path=wav.as_posix())
    tok = _tiny_tokenizer()
    cfg = ModelConfig(
        d_model=64,
        n_heads=4,
        n_layers=2,
        d_ff=128,
        max_seq_len=64,
        vocab_size=128,
        audio_duration_sec=0.25,
        n_audio_tokens=4,
    )
    ds = MultimodalDataset(corpus, tok, model_cfg=cfg, data_root=tmp_path, max_seq_len=64)
    item = ds[1]
    assert "audio" in item
    assert item["audio"].shape[0] == cfg.max_audio_samples


def test_collate_drops_partial_media_on_drive_errors():
    batch = [
        {"text_ids": torch.zeros(1, 4, dtype=torch.long), "labels": torch.zeros(1, 4, dtype=torch.long),
         "attention_mask": torch.ones(1, 4, dtype=torch.long), "images": torch.zeros(3, 8, 8)},
        {"text_ids": torch.zeros(1, 4, dtype=torch.long), "labels": torch.zeros(1, 4, dtype=torch.long),
         "attention_mask": torch.ones(1, 4, dtype=torch.long)},
    ]
    batch[0]["text_ids"] = batch[1]["text_ids"] = torch.zeros(4, dtype=torch.long).unsqueeze(0)
    batch[0]["labels"] = batch[1]["labels"] = torch.zeros(4, dtype=torch.long).unsqueeze(0)
    batch[0]["attention_mask"] = batch[1]["attention_mask"] = torch.ones(4, dtype=torch.long).unsqueeze(0)
    out = collate_multimodal(batch)
    assert "images" not in out


def test_collate_homogeneous_audio_batch(tmp_path: Path):
    wav = tmp_path / "clip.wav"
    _write_wav(wav)
    corpus = _write_corpus(tmp_path, audio_path=wav.as_posix())
    tok = _tiny_tokenizer()
    cfg = ModelConfig(d_model=64, n_heads=4, n_layers=2, d_ff=128, max_seq_len=64, vocab_size=128)
    ds = MultimodalDataset(corpus, tok, model_cfg=cfg, data_root=tmp_path, max_seq_len=64)
    audio_ds = subset_dataset(ds, [1])
    batch = collate_multimodal([audio_ds[0]])
    assert batch["audio"].shape[0] == 1


def test_mixed_batch_sampler_yields_homogeneous_buckets(tmp_path: Path):
    wav = tmp_path / "clip.wav"
    _write_wav(wav)
    corpus = _write_corpus(tmp_path, audio_path=wav.as_posix())
    tok = _tiny_tokenizer()
    ds = MultimodalDataset(corpus, tok, max_seq_len=32, data_root=tmp_path)
    sampler = MixedBatchSampler(ds, batch_size=2, multimodal_fraction=1.0, seed=0)
    batch = next(iter(sampler))
    buckets = {ds[i]["bucket"] for i in batch}
    assert len(buckets) == 1


def test_mixed_batch_sampler_uses_dataset_positions(tmp_path: Path):
    wav = tmp_path / "clip.wav"
    _write_wav(wav)
    corpus = _write_corpus(tmp_path, audio_path=wav.as_posix())
    tok = _tiny_tokenizer()
    ds = MultimodalDataset(corpus, tok, max_seq_len=32, data_root=tmp_path)
    train_ds = subset_dataset(ds, [0])
    sampler = MixedBatchSampler(train_ds, batch_size=1, multimodal_fraction=1.0, seed=0)
    batch = next(iter(sampler))
    assert batch == [0]
    assert train_ds[batch[0]]["bucket"] == "text"


def test_multimodal_backward_finite_grads(tmp_path: Path):
    wav = tmp_path / "clip.wav"
    _write_wav(wav)
    corpus = _write_corpus(tmp_path, audio_path=wav.as_posix())
    tok = _tiny_tokenizer()
    cfg = ModelConfig(
        d_model=64,
        n_heads=4,
        n_layers=2,
        d_ff=128,
        max_seq_len=64,
        vocab_size=128,
        audio_duration_sec=0.25,
        n_audio_tokens=4,
    )
    ds = MultimodalDataset(corpus, tok, model_cfg=cfg, data_root=tmp_path, max_seq_len=64)
    batch = collate_multimodal([ds[1]])
    model = MultimodalModel(cfg)
    model.train()

    with torch.autocast(device_type="cpu", enabled=False):
        out = model(
            text_ids=batch["text_ids"],
            labels=batch["labels"],
            audio=batch["audio"],
        )

    out["loss"].backward()
    for name, param in model.named_parameters():
        if param.grad is not None:
            assert torch.isfinite(param.grad).all(), name


def test_training_step_with_audio(tmp_path: Path):
    wav = tmp_path / "clip.wav"
    _write_wav(wav)
    corpus = _write_corpus(tmp_path, audio_path=wav.as_posix())
    tok = _tiny_tokenizer()
    cfg = ModelConfig(
        d_model=64,
        n_heads=4,
        n_layers=2,
        d_ff=128,
        max_seq_len=64,
        vocab_size=128,
        audio_duration_sec=0.25,
        n_audio_tokens=4,
    )
    ds = MultimodalDataset(corpus, tok, model_cfg=cfg, data_root=tmp_path, max_seq_len=64)
    item = ds[1]
    batch = collate_multimodal([item])

    model = MultimodalModel(cfg)
    model.train()
    out = model(
        text_ids=batch["text_ids"],
        labels=batch["labels"],
        audio=batch["audio"],
    )
    assert out["loss"].item() > 0
    out["loss"].backward()
    assert any(p.grad is not None for p in model.audio_encoder.parameters())


def test_speech_loss_forward(tmp_path: Path):
    wav = tmp_path / "clip.wav"
    _write_wav(wav, duration_sec=0.5)
    corpus = _write_corpus(tmp_path, audio_path=wav.as_posix())
    tok = _tiny_tokenizer()
    cfg = ModelConfig(
        d_model=64,
        n_heads=4,
        n_layers=2,
        d_ff=128,
        max_seq_len=128,
        vocab_size=128,
        audio_duration_sec=0.5,
        n_audio_tokens=4,
    )
    ds = MultimodalDataset(
        corpus, tok, model_cfg=cfg, data_root=tmp_path, max_seq_len=128, include_speech_target=True
    )
    item = ds[1]
    batch = collate_multimodal([item])
    model = MultimodalModel(cfg)
    model.train()
    out = model(
        text_ids=batch["text_ids"],
        labels=batch["labels"],
        audio=batch["audio"],
        speech_target=batch["speech_target"],
        speech_loss_weight=0.1,
    )
    assert "speech_loss" in out
    assert out["speech_loss"].item() >= 0


def test_corpus_modality_counts(tmp_path: Path):
    corpus = _write_corpus(tmp_path, audio_path="a.wav", image_path="b.jpg")
    from data.corpus import iter_jsonl

    records = list(iter_jsonl(corpus))
    counts = corpus_modality_counts(records)
    assert counts["audio"] == 1
    assert counts["image"] == 1
    assert counts["text"] == 1