"""Checkpoint sync between Colab local storage and mounted Google Drive."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from config import PROJECT_ROOT, resolve_data_path


def is_colab() -> bool:
    try:
        import google.colab  # noqa: F401

        return True
    except ImportError:
        return False


def colab_local_root() -> Path:
    return Path(os.environ.get("TRANSFORMER_LOCAL_ROOT", "/content/transformer_local"))


def colab_local_checkpoint_dir() -> Path:
    return colab_local_root() / "checkpoints"


def drive_is_writable(path: Path) -> bool:
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".drive_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        return True
    except OSError:
        return False


def _should_copy(src: Path, dst: Path) -> bool:
    if not dst.exists():
        return True
    src_stat = src.stat()
    dst_stat = dst.stat()
    if src_stat.st_mtime_ns > dst_stat.st_mtime_ns:
        return True
    if src_stat.st_mtime_ns == dst_stat.st_mtime_ns:
        return src_stat.st_size != dst_stat.st_size
    return False


def _atomic_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    shutil.copy2(src, tmp)
    tmp.replace(dst)


def sync_directory(
    local_dir: Path,
    drive_dir: Path,
    *,
    patterns: tuple[str, ...] = ("*.pt", "*.jsonl"),
    verbose: bool = True,
) -> list[str]:
    """Copy newer or missing files from local_dir to drive_dir."""
    if not local_dir.exists():
        return []

    if not drive_is_writable(drive_dir):
        if verbose:
            print(f"[drive-sync] Drive unavailable, skipping sync to {drive_dir}")
        return []

    synced: list[str] = []
    for pattern in patterns:
        for src in sorted(local_dir.glob(pattern)):
            if not src.is_file():
                continue
            dst = drive_dir / src.name
            if not _should_copy(src, dst):
                continue
            try:
                _atomic_copy(src, dst)
                synced.append(src.name)
            except OSError as exc:
                if verbose:
                    print(f"[drive-sync] Failed to copy {src.name}: {exc}")
    if verbose and synced:
        print(f"[drive-sync] Synced {len(synced)} file(s) to {drive_dir}: {', '.join(synced)}")
    return synced


def seed_local_from_drive(
    local_dir: Path,
    drive_dir: Path,
    *,
    patterns: tuple[str, ...] = ("*.pt", "*.jsonl"),
    verbose: bool = True,
) -> list[str]:
    """Restore local checkpoints from Drive after a Colab VM restart."""
    if not drive_dir.exists():
        return []

    local_dir.mkdir(parents=True, exist_ok=True)
    restored: list[str] = []
    for pattern in patterns:
        for src in sorted(drive_dir.glob(pattern)):
            if not src.is_file():
                continue
            dst = local_dir / src.name
            if not _should_copy(src, dst):
                continue
            try:
                _atomic_copy(src, dst)
                restored.append(src.name)
            except OSError as exc:
                if verbose:
                    print(f"[drive-sync] Failed to restore {src.name}: {exc}")
    if verbose and restored:
        print(
            f"[drive-sync] Restored {len(restored)} file(s) from {drive_dir} "
            f"to {local_dir}: {', '.join(restored)}"
        )
    return restored


def resolve_checkpoint_dirs(
    checkpoint_dir: Path | None = None,
    drive_checkpoint_dir: Path | None = None,
    *,
    project_root: Path = PROJECT_ROOT,
) -> tuple[Path, Path | None]:
    """
    Return (write_dir, sync_dir).

    On Colab, checkpoints are written to ephemeral local storage and optionally
    synced to the mounted Drive copy under project_root/checkpoints.
    """
    env_drive = os.environ.get("TRANSFORMER_DRIVE_CHECKPOINT_DIR")
    drive_default = drive_checkpoint_dir or (
        Path(env_drive) if env_drive else project_root / "checkpoints"
    )

    if checkpoint_dir is not None:
        write_dir = checkpoint_dir
        if drive_checkpoint_dir is not None:
            return write_dir, drive_checkpoint_dir
        if is_colab() and not _is_local_ephemeral(write_dir):
            sync_dir = drive_default if _drive_sync_available(drive_default) else None
            return write_dir, sync_dir
        return write_dir, None

    if is_colab():
        write_dir = colab_local_checkpoint_dir()
        sync_dir = drive_checkpoint_dir
        if sync_dir is None and _drive_sync_available(drive_default):
            sync_dir = drive_default
        return write_dir, sync_dir

    return drive_default, None


def resolve_metrics_path(write_dir: Path, metrics_path: Path | None = None) -> Path:
    if metrics_path is not None:
        return metrics_path
    return write_dir / "training_metrics.jsonl"


def resolve_checkpoint_file(
    path: Path,
    *,
    write_dir: Path,
    sync_dir: Path | None = None,
) -> Path | None:
    """Find a checkpoint on local storage, cwd/project paths, or Drive."""
    priority = [write_dir / path.name, path]
    if sync_dir is not None:
        priority.append(sync_dir / path.name)

    for candidate in priority:
        if candidate.exists():
            return candidate.resolve()

    resolved = resolve_data_path(path)
    if resolved is not None:
        return resolved
    return None


def _is_under_drive(path: Path) -> bool:
    parts = path.resolve().parts
    if "drive" not in parts:
        return False
    return "MyDrive" in parts or "My Drive" in parts


def _is_local_ephemeral(path: Path) -> bool:
    resolved = path.resolve()
    parts = resolved.parts
    if not parts or parts[0] != "/":
        return False
    if len(parts) < 2 or parts[1] != "content":
        return False
    return len(parts) < 3 or parts[2] != "drive"


def _drive_sync_available(drive_dir: Path) -> bool:
    if _is_under_drive(drive_dir):
        return True
    return drive_is_writable(drive_dir)