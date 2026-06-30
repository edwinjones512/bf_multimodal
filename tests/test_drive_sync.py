from pathlib import Path
from unittest.mock import patch

from utils.drive_sync import (
    resolve_checkpoint_dirs,
    resolve_checkpoint_file,
    resolve_metrics_path,
    seed_local_from_drive,
    sync_directory,
)


def test_sync_directory_copies_newer_local_files(tmp_path: Path):
    local_dir = tmp_path / "local"
    drive_dir = tmp_path / "drive"
    local_dir.mkdir()
    drive_dir.mkdir()

    drive_ckpt = drive_dir / "latest.pt"
    drive_ckpt.write_bytes(b"drive-v1")
    local_ckpt = local_dir / "latest.pt"
    local_ckpt.write_bytes(b"local-v2-longer")

    synced = sync_directory(local_dir, drive_dir, verbose=False)
    assert synced == ["latest.pt"]
    assert drive_ckpt.read_bytes() == b"local-v2-longer"


def test_seed_local_from_drive_restores_missing_files(tmp_path: Path):
    local_dir = tmp_path / "local"
    drive_dir = tmp_path / "drive"
    local_dir.mkdir()
    drive_dir.mkdir()

    drive_ckpt = drive_dir / "best.pt"
    drive_ckpt.write_bytes(b"drive-best")

    restored = seed_local_from_drive(local_dir, drive_dir, verbose=False)
    assert restored == ["best.pt"]
    assert (local_dir / "best.pt").read_bytes() == b"drive-best"


def test_resolve_checkpoint_file_prefers_local_write_dir(tmp_path: Path):
    write_dir = tmp_path / "local"
    drive_dir = tmp_path / "drive"
    write_dir.mkdir()
    drive_dir.mkdir()
    (write_dir / "best.pt").write_bytes(b"local")

    resolved = resolve_checkpoint_file(
        Path("checkpoints/best.pt"),
        write_dir=write_dir,
        sync_dir=drive_dir,
    )
    assert resolved == (write_dir / "best.pt").resolve()


@patch("utils.drive_sync.is_colab", return_value=True)
@patch("utils.drive_sync._drive_sync_available", return_value=True)
def test_resolve_checkpoint_dirs_uses_local_storage_on_colab(
    _drive_available,
    _colab,
    tmp_path: Path,
):
    project_root = tmp_path / "content" / "drive" / "MyDrive" / "transformer"
    project_root.mkdir(parents=True)

    write_dir, sync_dir = resolve_checkpoint_dirs(project_root=project_root)
    assert write_dir == Path("/content/transformer_local/checkpoints")
    assert sync_dir == project_root / "checkpoints"


def test_resolve_metrics_path_defaults_to_write_dir(tmp_path: Path):
    write_dir = tmp_path / "local_checkpoints"
    assert resolve_metrics_path(write_dir) == write_dir / "training_metrics.jsonl"