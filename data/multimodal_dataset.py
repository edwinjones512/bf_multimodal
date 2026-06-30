"""Dataset that loads paired media + text from the unified corpus."""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import Dataset
from tokenizers import Tokenizer

from config import ModelConfig, SPECIAL_TOKENS
from data.corpus import iter_jsonl, normalize_legacy_record
from data.modality import has_media, modality_bucket
from utils.media import load_audio, load_image, load_video_frames


def safe_path_exists(path: Path) -> bool:
    """Return False on missing paths or Google Drive I/O errors instead of raising."""
    try:
        return path.exists()
    except OSError:
        return False


def resolve_media_path(path: str | Path, data_root: Path | None = None) -> Path | None:
    """Resolve a media path, returning None when the file is missing or unreadable."""
    p = Path(path)
    candidates = [p]
    if data_root:
        candidates.append(data_root / p)
    candidates.append(Path.cwd() / p)

    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        if safe_path_exists(candidate):
            return candidate
    return None


def _effective_bucket(record: dict, loaded: dict[str, torch.Tensor | None]) -> str:
    bucket = modality_bucket(record)
    if bucket == "audio" and loaded.get("audio") is None:
        return "text"
    if bucket == "image" and loaded.get("images") is None:
        return "text"
    if bucket == "video" and loaded.get("video") is None:
        return "text"
    return bucket


class MultimodalDataset(Dataset):
    """Tokenized corpus records with optional audio, image, and video tensors."""

    def __init__(
        self,
        jsonl_path: Path,
        tokenizer: Tokenizer,
        model_cfg: ModelConfig | None = None,
        max_seq_len: int = 1024,
        bos_id: int = 2,
        eos_id: int = 3,
        pad_id: int = 0,
        sources: set[str] | None = None,
        indices: list[int] | None = None,
        data_root: Path | None = None,
        include_speech_target: bool = False,
    ):
        self.jsonl_path = jsonl_path
        self.tokenizer = tokenizer
        self.model_cfg = model_cfg or ModelConfig()
        self.max_seq_len = max_seq_len
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.pad_id = pad_id
        self.data_root = data_root or Path.cwd()
        self.include_speech_target = include_speech_target

        self.records: list[dict] = []
        for record in iter_jsonl(jsonl_path):
            record = normalize_legacy_record(record)
            if sources and record.get("source") not in sources:
                continue
            self.records.append(record)

        self.active_indices = indices if indices is not None else list(range(len(self.records)))

        self.bucket_indices: dict[str, list[int]] = {
            "text": [],
            "audio": [],
            "image": [],
            "video": [],
        }
        for pos, record_idx in enumerate(self.active_indices):
            bucket = modality_bucket(self.records[record_idx])
            self.bucket_indices[bucket].append(pos)

    def _encode(self, text: str) -> list[int]:
        ids = self.tokenizer.encode(text).ids
        return [self.bos_id] + ids[: self.max_seq_len - 2] + [self.eos_id]

    def _load_audio(self, record: dict) -> torch.Tensor | None:
        path = record.get("audio_path")
        if not path:
            return None
        resolved = resolve_media_path(path, self.data_root)
        if resolved is None:
            return None
        try:
            return load_audio(
                resolved,
                sample_rate=self.model_cfg.audio_sample_rate,
                max_duration_sec=self.model_cfg.audio_duration_sec,
            ).squeeze(0)
        except OSError:
            return None

    def _load_image(self, record: dict) -> torch.Tensor | None:
        path = record.get("image_path")
        if not path:
            return None
        resolved = resolve_media_path(path, self.data_root)
        if resolved is None:
            return None
        try:
            return load_image(resolved, self.model_cfg.image_size).squeeze(0)
        except OSError:
            return None

    def _load_video(self, record: dict) -> torch.Tensor | None:
        path = record.get("video_path")
        if not path:
            return None
        resolved = resolve_media_path(path, self.data_root)
        if resolved is None:
            return None
        n_frames = self.model_cfg.video_train_frames
        try:
            return load_video_frames(
                resolved,
                n_frames=n_frames,
                image_size=self.model_cfg.image_size,
            ).squeeze(0)
        except OSError:
            return None

    def __len__(self) -> int:
        return len(self.active_indices)

    def __getitem__(self, idx: int) -> dict:
        record_idx = self.active_indices[idx]
        record = self.records[record_idx]
        ids = self._encode(record["text"])
        length = len(ids)
        padded = ids + [self.pad_id] * (self.max_seq_len - length)
        input_ids = torch.tensor(padded[: self.max_seq_len], dtype=torch.long)
        labels = input_ids.clone()
        labels[length:] = self.pad_id

        item: dict = {
            "text_ids": input_ids,
            "labels": labels,
            "attention_mask": torch.tensor(
                [1] * min(length, self.max_seq_len) + [0] * max(0, self.max_seq_len - length),
                dtype=torch.long,
            ),
            "bucket": modality_bucket(record),
            "source": record.get("source", ""),
            "record_id": record.get("id"),
        }

        audio = self._load_audio(record)
        if audio is not None:
            item["audio"] = audio

        image = self._load_image(record)
        if image is not None:
            item["images"] = image

        video = self._load_video(record)
        if video is not None:
            item["video"] = video

        if self.include_speech_target and audio is not None:
            item["speech_target"] = audio

        item["bucket"] = _effective_bucket(
            record,
            {"audio": audio, "images": image, "video": video},
        )

        return item

    @property
    def multimodal_indices(self) -> list[int]:
        return [
            pos
            for pos, record_idx in enumerate(self.active_indices)
            if has_media(self.records[record_idx])
        ]


def train_val_split_positions(
    dataset: MultimodalDataset,
    val_fraction: float = 0.05,
    seed: int = 42,
    holdout_sources: set[str] | None = None,
) -> tuple[list[int], list[int]]:
    """Return (train_positions, val_positions) as indices into ``dataset``."""
    holdout_sources = holdout_sources or {"librivox"}
    generator = torch.Generator().manual_seed(seed)

    book_ids: dict[str, list[int]] = {}
    plain_indices: list[int] = []

    for i in range(len(dataset)):
        record = dataset.records[dataset.active_indices[i]]
        source = record.get("source", "")
        if source in holdout_sources and record.get("id"):
            book_key = str(record["id"]).split(":")[0]
            book_ids.setdefault(book_key, []).append(i)
        else:
            plain_indices.append(i)

    val_positions: list[int] = []
    train_positions: list[int] = []

    for _book_key, idxs in book_ids.items():
        n_val = max(1, int(len(idxs) * val_fraction)) if len(idxs) > 1 else 0
        perm = torch.randperm(len(idxs), generator=generator).tolist()
        val_pick = {idxs[j] for j in perm[:n_val]}
        for idx in idxs:
            (val_positions if idx in val_pick else train_positions).append(idx)

    n_plain_val = max(1, int(len(plain_indices) * val_fraction)) if plain_indices else 0
    perm = torch.randperm(len(plain_indices), generator=generator).tolist()
    for j, pos in enumerate(perm):
        idx = plain_indices[pos]
        (val_positions if j < n_plain_val else train_positions).append(idx)

    return train_positions, val_positions


def subset_dataset(dataset: MultimodalDataset, positions: list[int]) -> MultimodalDataset:
    """Create a dataset view containing only the selected positions."""
    record_indices = [dataset.active_indices[p] for p in positions]
    return MultimodalDataset(
        dataset.jsonl_path,
        dataset.tokenizer,
        model_cfg=dataset.model_cfg,
        max_seq_len=dataset.max_seq_len,
        bos_id=dataset.bos_id,
        eos_id=dataset.eos_id,
        pad_id=dataset.pad_id,
        data_root=dataset.data_root,
        include_speech_target=dataset.include_speech_target,
        indices=record_indices,
    )