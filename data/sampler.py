"""Samplers that guarantee multimodal examples appear in training."""

from __future__ import annotations

import random
from collections.abc import Iterator

from torch.utils.data import Sampler

from data.multimodal_dataset import MultimodalDataset


class MixedBatchSampler(Sampler[list[int]]):
    """
    Yield homogeneous batches: all text, all audio, all image, or all video.

    ``multimodal_fraction`` is the probability each batch is non-text (split evenly
    across available media buckets). Homogeneous batches keep collation simple and
    avoid mixing tensor shapes within a step.
    """

    BUCKET_CAPS = {
        "text": None,
        "audio": "max_audio_per_batch",
        "image": "max_images_per_batch",
        "video": "max_video_per_batch",
    }

    def __init__(
        self,
        dataset: MultimodalDataset,
        batch_size: int,
        multimodal_fraction: float = 0.25,
        max_audio_per_batch: int = 4,
        max_images_per_batch: int = 4,
        max_video_per_batch: int = 2,
        seed: int = 42,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.multimodal_fraction = multimodal_fraction
        self.max_audio_per_batch = max_audio_per_batch
        self.max_images_per_batch = max_images_per_batch
        self.max_video_per_batch = max_video_per_batch
        self.seed = seed

        self._buckets = {
            "text": list(dataset.bucket_indices.get("text", [])),
            "audio": list(dataset.bucket_indices.get("audio", [])),
            "image": list(dataset.bucket_indices.get("image", [])),
            "video": list(dataset.bucket_indices.get("video", [])),
        }
        self._mm_buckets = [b for b in ("video", "image", "audio") if self._buckets[b]]
        self._epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch

    def _cap(self, bucket: str) -> int:
        return {
            "text": self.batch_size,
            "audio": min(self.batch_size, self.max_audio_per_batch),
            "image": min(self.batch_size, self.max_images_per_batch),
            "video": min(self.batch_size, self.max_video_per_batch),
        }[bucket]

    def __len__(self) -> int:
        n = len(self.dataset)
        return max(1, (n + self.batch_size - 1) // self.batch_size)

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(self.seed + self._epoch)
        pools = {k: v[:] for k, v in self._buckets.items()}
        for pool in pools.values():
            rng.shuffle(pool)
        cursors = {k: 0 for k in pools}

        for _ in range(len(self)):
            if self._mm_buckets and rng.random() < self.multimodal_fraction:
                bucket = rng.choice(self._mm_buckets)
            else:
                bucket = "text"

            cap = self._cap(bucket)
            pool = pools[bucket]
            if not pool:
                bucket = "text"
                cap = self.batch_size
                pool = pools["text"]

            batch: list[int] = []
            while len(batch) < cap and pool:
                batch.append(pool[cursors[bucket] % len(pool)])
                cursors[bucket] += 1

            rng.shuffle(batch)
            yield batch