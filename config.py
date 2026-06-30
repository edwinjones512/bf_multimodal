"""Configuration for the multimodal transformer."""

import os
from dataclasses import dataclass, field
from pathlib import Path

# Repo root (directory containing this file). Resolves default data paths regardless of cwd.
PROJECT_ROOT = Path(__file__).resolve().parent


def data_path_candidates(path: Path) -> list[Path]:
    """Return candidate locations for a data file (cwd, project root, optional env override)."""
    candidates: list[Path] = []
    seen: set[str] = set()

    def add(candidate: Path) -> None:
        key = str(candidate)
        if key not in seen:
            seen.add(key)
            candidates.append(candidate)

    env_root = os.environ.get("TRANSFORMER_DATA_ROOT")
    if env_root:
        env = Path(env_root)
        if path.is_absolute():
            try:
                add(env / path.relative_to(PROJECT_ROOT))
            except ValueError:
                add(path)
        else:
            add(env / path)

    if path.is_absolute():
        add(path)
        try:
            add(Path.cwd() / path.relative_to(PROJECT_ROOT))
        except ValueError:
            pass
    else:
        add(Path.cwd() / path)
        add(PROJECT_ROOT / path)

    return candidates


def resolve_data_path(path: Path) -> Path | None:
    """Return the first existing path among data_path_candidates, or None."""
    for candidate in data_path_candidates(path):
        if candidate.exists():
            return candidate.resolve()
    return None


@dataclass
class ModelConfig:
    vocab_size: int = 32000
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 6
    d_ff: int = 2048
    max_seq_len: int = 1024
    inference_max_seq_len: int = 8192
    rope_theta: float = 10000.0
    rope_scale: float = 1.0
    sliding_window: int = 512  # local attention window (0 = full causal)
    dropout: float = 0.1
    image_size: int = 224
    patch_size: int = 16
    n_image_tokens: int = 196  # (224/16)^2
    n_video_frames: int = 8
    n_video_tokens: int = 8  # one token per frame after pooling
    audio_sample_rate: int = 16000
    audio_duration_sec: float = 5.0
    n_mel_bins: int = 80
    audio_n_fft: int = 400
    audio_hop_length: int = 160
    n_audio_tokens: int = 32
    # Fewer frames during training saves VRAM; inference still uses n_video_frames.
    video_train_frames: int = 4
    video_encode_micro_batch: int = 1  # encode one video at a time to avoid OOM

    @property
    def max_audio_samples(self) -> int:
        return int(self.audio_sample_rate * self.audio_duration_sec)

    @classmethod
    def colab(cls) -> "ModelConfig":
        """Smaller config that fits a Colab T4 GPU (~15 GB)."""
        return cls(
            d_model=256,
            n_heads=4,
            n_layers=4,
            d_ff=1024,
            max_seq_len=512,
            inference_max_seq_len=8192,
            sliding_window=256,
            image_size=112,
            patch_size=16,
            n_image_tokens=49,
            n_video_frames=4,
            audio_duration_sec=3.0,
            n_audio_tokens=16,
        )


@dataclass
class TrainConfig:
    batch_size: int = 8
    multimodal_fraction: float = 0.25
    max_audio_per_batch: int = 4
    max_video_per_batch: int = 2
    max_images_per_batch: int = 4
    freeze_encoders_steps: int = 0
    train_speech: bool = False
    speech_loss_weight: float = 0.1
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    max_epochs: int = 3
    warmup_steps: int = 1000
    grad_clip: float = 1.0
    save_every: int = 100_000
    latest_every: int = 1000
    eval_every: int = 500
    num_workers: int = 0
    val_fraction: float = 0.05
    grad_accum_steps: int = 1
    amp: bool = True
    label_smoothing: float = 0.05
    early_stop_patience: int = 5


@dataclass
class EvalConfig:
    wiki_qa_path: Path = field(default_factory=lambda: PROJECT_ROOT / "data/processed/wiki_qa.jsonl")
    audio_eval_path: Path = field(default_factory=lambda: PROJECT_ROOT / "data/processed/librivox.jsonl")
    qa_eval_samples: int = 50
    qa_max_tokens: int = 64
    audio_eval_samples: int = 50
    audio_max_tokens: int = 128


@dataclass
class DataConfig:
    data_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data")
    wiki_dump_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data/wikipedia")
    gutenberg_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data/gutenberg")
    github_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data/github")
    librivox_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data/librivox")
    image_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data/images")
    video_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data/videos")
    coco_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data/images/coco")
    msrvtt_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data/videos/msrvtt")
    processed_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data/processed")
    corpus_path: Path = field(default_factory=lambda: PROJECT_ROOT / "data/processed/corpus.jsonl")
    tokenizer_path: Path = field(default_factory=lambda: PROJECT_ROOT / "data/tokenizer.json")
    checkpoint_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "checkpoints")
    metrics_path: Path = field(default_factory=lambda: PROJECT_ROOT / "checkpoints/training_metrics.jsonl")


WIKI_DUMP_BASE = "https://dumps.wikimedia.org/enwiki/latest"
# All profiles use pages-articles dumps = current revision only per page.
# Do not use pages-meta-history (full edit history, 100+ GB).
WIKI_DUMP_FILES = {
    "articles": "enwiki-latest-pages-articles-multistream.xml.bz2",
    "index": "enwiki-latest-pages-articles-multistream-index.txt.bz2",
}

WIKI_DUMP_PROFILES: dict[str, dict[str, str | None]] = {
    "full": {
        "base": WIKI_DUMP_BASE,
        "articles": WIKI_DUMP_FILES["articles"],
        "index": WIKI_DUMP_FILES["index"],
        "dump_type": "pages-articles-multistream",
        "revisions": "current",
        "size_hint": "~20 GB, all articles (current revisions only)",
    },
    "small": {
        "base": WIKI_DUMP_BASE,
        "articles": "enwiki-latest-pages-articles1.xml-p1p41242.bz2",
        "index": None,
        "dump_type": "pages-articles",
        "revisions": "current",
        "size_hint": "~300 MB, first ~41k articles (current revisions only)",
    },
    "simple": {
        "base": "https://dumps.wikimedia.org/simplewiki/latest",
        "articles": "simplewiki-latest-pages-articles-multistream.xml.bz2",
        "index": "simplewiki-latest-pages-articles-multistream-index.txt.bz2",
        "dump_type": "pages-articles-multistream",
        "revisions": "current",
        "size_hint": "~380 MB, Simple English (current revisions only)",
    },
}

WIKI_ARTICLE_DUMP_NAMES = tuple(
    profile["articles"]
    for profile in WIKI_DUMP_PROFILES.values()
    if profile["articles"]
)


WIKI_DUMPS_HOST = "https://dumps.wikimedia.org"


def wiki_dump_base(wiki: str, date: str) -> str:
    return f"{WIKI_DUMPS_HOST}/{wiki}/{date}"


def fetch_wiki_dumpstatus(wiki: str, date: str) -> dict:
    """Return dumpstatus.json for a dated Wikimedia dump."""
    import requests

    url = f"{wiki_dump_base(wiki, date)}/dumpstatus.json"
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    return resp.json()


def list_wiki_dump_dates(wiki: str) -> list[str]:
    """Return available dump dates (YYYYMMDD), newest first."""
    import re

    import requests

    resp = requests.get(f"{WIKI_DUMPS_HOST}/{wiki}/", timeout=60)
    resp.raise_for_status()
    return sorted(
        re.findall(r'href="(\d{8})/"', resp.text),
        reverse=True,
    )


def resolve_wiki_dump_date(wiki: str, date: str | None = None) -> str:
    """Return an explicit dump date, defaulting to the newest available."""
    dates = list_wiki_dump_dates(wiki)
    if not dates:
        raise RuntimeError(f"No dated dumps found for {wiki}")
    if date:
        if date not in dates:
            raise ValueError(
                f"Unknown dump date {date} for {wiki}. "
                f"Recent dates: {', '.join(dates[:5])}"
            )
        return date
    return dates[0]


def list_pages_articles_files(
    dumpstatus: dict,
    *,
    include_shards: bool = True,
) -> list[tuple[str, int, str | None]]:
    """
    Return (filename, size_bytes, md5) for all done pages-articles dump files.

    By default includes both recombined archives and per-shard files.
    """
    files: dict[str, tuple[int, str | None]] = {}
    for job in dumpstatus.get("jobs", {}).values():
        if job.get("status") != "done":
            continue
        for fname, meta in job.get("files", {}).items():
            if "pages-articles" not in fname:
                continue
            if not include_shards and not (
                fname.endswith("pages-articles-multistream.xml.bz2")
                or fname.endswith("pages-articles-multistream-index.txt.bz2")
            ):
                continue
            size = int(meta.get("size") or 0)
            md5 = meta.get("md5")
            prev = files.get(fname)
            if prev is None or size > prev[0]:
                files[fname] = (size, md5)
    return sorted(
        ((fname, size, md5) for fname, (size, md5) in files.items()),
        key=lambda item: item[0],
    )


def find_wikipedia_dump(dump_dir: Path) -> Path | None:
    """Return the best Wikipedia articles dump in dump_dir."""
    for name in WIKI_ARTICLE_DUMP_NAMES:
        path = dump_dir / name
        if path.exists() and path.stat().st_size > 0:
            return path

    multistream = [
        p
        for p in dump_dir.glob("*pages-articles-multistream.xml.bz2")
        if p.is_file() and p.stat().st_size > 0
    ]
    if multistream:
        return max(multistream, key=lambda p: p.stat().st_size)

    shards = [
        p
        for p in dump_dir.glob("*pages-articles*.xml*.bz2")
        if p.is_file() and p.stat().st_size > 0
    ]
    if shards:
        return max(shards, key=lambda p: p.stat().st_size)
    return None

GUTENBERG_FEEDS_BASE = "https://www.gutenberg.org/cache/epub/feeds"
GUTENBERG_CATALOG_URL = f"{GUTENBERG_FEEDS_BASE}/pg_catalog.csv"
GUTENBERG_FILES_BASE = "https://www.gutenberg.org/files"

# Legacy 10 GB bulk archive (deprecated — use Zenodo subsets below)
GUTENBERG_LEGACY_BULK_FILES = (
    "txt-files.tar.zip",
    "txt-files.tar.zip.part",
)

# Zenodo Gutenberg text subsets: https://zenodo.org/records/3360392
GUTENBERG_ZENODO_RECORD = "3360392"
GUTENBERG_ZENODO_BASE = f"https://zenodo.org/records/{GUTENBERG_ZENODO_RECORD}/files"
GUTENBERG_DEFAULT_SUBSET = "357mb"
GUTENBERG_SUBSET_MARKER = "zenodo_subset.txt"

GUTENBERG_ZENODO_SUBSETS: dict[str, dict[str, str | int]] = {
    "184mb": {
        "filename": "D184MB.zip",
        "label": "~184 MB uncompressed, 68 MB download",
        "bytes": 68_129_136,
        "md5": "b400e353aff56022035cbebc328878cb",
    },
    "357mb": {
        "filename": "D357MB.zip",
        "label": "~357 MB uncompressed, 134 MB download",
        "bytes": 134_219_820,
        "md5": "3ce8872ec280123aea826766a11d89f2",
    },
    "670mb": {
        "filename": "D670MB.zip",
        "label": "~670 MB uncompressed, 251 MB download",
        "bytes": 251_200_879,
        "md5": "6bb3d099581bebc1f13b8dca33592a80",
    },
    "1gb": {
        "filename": "D1GB.zip",
        "label": "~1 GB uncompressed, 356 MB download",
        "bytes": 356_114_814,
        "md5": "58493488f94d43ae114399014edc1ae4",
    },
    "1.7gb": {
        "filename": "D1.7GB.zip",
        "label": "~1.7 GB uncompressed, 563 MB download",
        "bytes": 563_367_210,
        "md5": "c4fb5421d358ab25d3a17f8c9fa48ff1",
    },
}


def gutenberg_zenodo_url(subset: str) -> str:
    filename = GUTENBERG_ZENODO_SUBSETS[subset]["filename"]
    return f"{GUTENBERG_ZENODO_BASE}/{filename}?download=1"


def gutenberg_zenodo_path(gutenberg_dir: Path, subset: str | None = None) -> Path:
    key = subset or GUTENBERG_DEFAULT_SUBSET
    return gutenberg_dir / str(GUTENBERG_ZENODO_SUBSETS[key]["filename"])


def read_gutenberg_subset(gutenberg_dir: Path) -> str | None:
    marker = gutenberg_dir / GUTENBERG_SUBSET_MARKER
    if marker.exists():
        key = marker.read_text(encoding="utf-8").strip()
        if key in GUTENBERG_ZENODO_SUBSETS:
            return key
    for key, info in GUTENBERG_ZENODO_SUBSETS.items():
        path = gutenberg_dir / str(info["filename"])
        if path.exists() and path.stat().st_size >= int(info["bytes"]) * 0.99:
            return key
    return None


def has_gutenberg_archive(gutenberg_dir: Path) -> bool:
    raw = gutenberg_dir / "raw"
    if raw.is_dir() and any(raw.glob("*.txt")):
        return True
    return read_gutenberg_subset(gutenberg_dir) is not None

GITHUB_API_BASE = "https://api.github.com"
GITHUB_DEFAULT_MAX_REPOS = 50
GITHUB_DEFAULT_LANGUAGES = (
    "python", "javascript", "typescript", "kotlin", "swift",
)
GITHUB_CODE_EXTENSIONS = {
    ".py", ".js", ".jsx", ".mjs", ".cjs", ".ts", ".tsx", ".kt", ".kts", ".swift",
}
LIBRIVOX_API_BASE = "https://librivox.org/api/feed/audiobooks"
LIBRIVOX_DEFAULT_LANGUAGE = "English"
LIBRIVOX_WHISPER_MODEL = "base"

# MSR-VTT mirrors (Oxford VGG msrvtt.zip is offline; use HuggingFace)
MSRVTT_HF_BASE = "https://huggingface.co/datasets/friedrichor/MSR-VTT/resolve/main"
MSRVTT_ANNOTATION_FILES = {
    "train_7k": "msrvtt_train_7k.json",
    "train_9k": "msrvtt_train_9k.json",
    "test_1k": "msrvtt_test_1k.json",
}
MSRVTT_VIDEOS_ZIP = "MSRVTT_Videos.zip"
MSRVTT_DEFAULT_SPLIT = "train_7k"

GITHUB_SKIP_DIRS = {
    "node_modules", "dist", "build", "coverage", "vendor", "venv", ".venv",
    "__pycache__", ".git", ".github", ".next", "target", "out", "bin", "obj",
    "site-packages", "eggs", ".tox", ".mypy_cache", ".pytest_cache",
}

SPECIAL_TOKENS = {
    "<pad>": 0,
    "<unk>": 1,
    "<bos>": 2,
    "<eos>": 3,
    "<image>": 4,
    "<video>": 5,
    "<text>": 6,
    "<audio>": 7,
    "<speech>": 8,
}