import json
import zipfile
from pathlib import Path

from process_github import language_from_extension, parse_repo_name_from_zip, process_zip, should_skip_path


def test_language_from_extension():
    assert language_from_extension(".py") == "python"
    assert language_from_extension(".tsx") == "typescript"
    assert language_from_extension(".jsx") == "javascript"
    assert language_from_extension(".kt") == "kotlin"
    assert language_from_extension(".kts") == "kotlin"
    assert language_from_extension(".swift") == "swift"


def test_should_skip_node_modules():
    assert should_skip_path("repo-main/node_modules/pkg/index.js")
    assert not should_skip_path("repo-main/src/main.py")


def test_parse_repo_name():
    assert parse_repo_name_from_zip("torvalds_linux.zip") == "torvalds/linux"


def test_process_zip_extracts_python(tmp_path: Path):
    zip_path = tmp_path / "acme_demo.zip"
    code = "def hello():\n    return 'world'\n" * 20

    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("acme-demo-abc123/src/app.py", code)

    records = process_zip(zip_path, min_chars=50, max_file_bytes=100_000, chunk_size=500)
    assert len(records) >= 1
    assert records[0]["source"] == "github"
    assert records[0]["language"] == "python"
    assert "def hello" in records[0]["text"]