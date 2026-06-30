import json
from pathlib import Path

import requests

from config import MSRVTT_ANNOTATION_FILES, MSRVTT_HF_BASE
from download_msrvtt import hf_url
from process_msrvtt import caption_from_entry, load_msrvtt_entries, resolve_video_path, video_filename_from_entry


def test_hf_annotation_urls_respond():
    for filename in MSRVTT_ANNOTATION_FILES.values():
        url = hf_url(filename)
        assert url.startswith(MSRVTT_HF_BASE)
        resp = requests.head(url, timeout=30, allow_redirects=True)
        assert resp.status_code == 200, f"{url} returned {resp.status_code}"


def test_msrvtt_json_structure():
    url = hf_url(MSRVTT_ANNOTATION_FILES["test_1k"])
    data = json.loads(requests.get(url, timeout=60).content)
    entries = data
    assert len(entries) == 1000
    entry = entries[0]
    assert video_filename_from_entry(entry).endswith(".mp4")
    assert caption_from_entry(entry)


def test_resolve_video_path(tmp_path: Path):
    entry = {"video_id": "video7020", "video": "video7020.mp4", "caption": "test"}
    (tmp_path / "video7020.mp4").write_bytes(b"fake")
    assert resolve_video_path(tmp_path, entry) is not None
    assert resolve_video_path(tmp_path, {"video": "missing.mp4"}) is None