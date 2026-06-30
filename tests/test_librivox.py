import json
import struct
import wave
from pathlib import Path

import torch

from data.corpus import make_record
from download_librivox import format_authors, is_english, section_filename, strip_html
from process_librivox import build_chunk_text, load_manifest
from utils.audio_convert import chunk_waveform, chunk_waveform_with_offsets
from utils.transcribe import align_words_to_chunk, words_to_text


def test_is_english():
    assert is_english("English")
    assert is_english("english")
    assert not is_english("French")
    assert not is_english(None)


def test_format_authors():
    authors = format_authors([
        {"first_name": "Jane", "last_name": "Austen"},
        {"first_name": "Charles", "last_name": "Dickens"},
    ])
    assert authors == "Jane Austen; Charles Dickens"


def test_strip_html():
    text = strip_html("<b>Hello</b> <i>world</i>")
    assert text == "Hello world"


def test_section_filename():
    assert section_filename("52", "3") == "52_0003.mp3"


def test_build_chunk_text():
    book = {"title": "Pride and Prejudice", "authors": "Jane Austen"}
    section = {"title": "Chapter 1"}
    assert "Pride and Prejudice" in build_chunk_text(book, section, 0)
    assert "Chapter 1" in build_chunk_text(book, section, 0)
    assert "part 2" in build_chunk_text(book, section, 1)


def test_chunk_waveform():
    waveform = torch.ones(32000)
    chunks = chunk_waveform(waveform, sample_rate=16000, chunk_sec=1.0, overlap_sec=0.25)
    assert len(chunks) >= 2
    assert all(chunk.shape[0] == 16000 for chunk in chunks)


def test_chunk_waveform_with_offsets():
    waveform = torch.ones(32000)
    windows = chunk_waveform_with_offsets(waveform, sample_rate=16000, chunk_sec=1.0, overlap_sec=0.25)
    assert windows[0]["start_sec"] == 0.0
    assert windows[1]["start_sec"] > 0.0
    assert windows[0]["end_sec"] == 1.0


def test_align_words_to_chunk():
    words = [
        {"word": "Hello", "start": 0.2, "end": 0.6},
        {"word": "world", "start": 0.7, "end": 1.1},
        {"word": "again", "start": 4.8, "end": 5.2},
    ]
    aligned = align_words_to_chunk(words, chunk_start_sec=0.0, chunk_end_sec=1.5)
    assert len(aligned) == 2
    assert aligned[0]["word"] == "Hello"
    assert aligned[0]["start"] == 0.2
    assert aligned[1]["word"] == "world"


def test_words_to_text():
    assert words_to_text([{"word": "Hello"}, {"word": "world"}]) == "Hello world"


def test_make_record_with_audio_path():
    record = make_record(
        "librivox",
        "Chapter one.",
        "Test Book",
        record_id="1:2:0",
        language="en",
        audio_path="data/librivox/chunks/1/1_0001_0000.wav",
    )
    assert record["source"] == "librivox"
    assert record["audio_path"].endswith(".wav")


def test_load_manifest(tmp_path):
    manifest = tmp_path / "books.jsonl"
    book = {
        "id": "1",
        "title": "Demo",
        "sections": [{"section_number": "1", "title": "Intro", "listen_url": "http://example.com/a.mp3"}],
    }
    manifest.write_text(json.dumps(book) + "\n", encoding="utf-8")
    loaded = load_manifest(manifest)
    assert len(loaded) == 1
    assert loaded[0]["title"] == "Demo"


def _write_wav(path, waveform: torch.Tensor, sample_rate: int = 16000) -> None:
    samples = (waveform.clamp(-1, 1) * 32767).short().tolist()
    with wave.open(str(path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(struct.pack(f"<{len(samples)}h", *samples))


def test_process_section_creates_records(tmp_path):
    from process_librivox import process_section

    book = {
        "id": "99",
        "title": "Sample Book",
        "authors": "Test Author",
        "sections": [],
    }
    section = {"id": "7", "section_number": "1", "title": "Opening"}
    raw_dir = tmp_path / "raw"
    wav_dir = tmp_path / "wav"
    chunk_dir = tmp_path / "chunks"
    mp3_dir = raw_dir / "99"
    mp3_dir.mkdir(parents=True)
    mp3_path = mp3_dir / "99_0001.mp3"

    tone = 0.4 * torch.sin(torch.linspace(0, 80 * 3.14159, 8000))
    wav_path = mp3_dir / "temp.wav"
    _write_wav(wav_path, tone)
    wav_path.rename(mp3_dir / "99_0001.wav")
    mp3_path.write_bytes(b"not-a-real-mp3")

    # Bypass MP3 conversion by pre-creating the section WAV.
    prebuilt_wav = wav_dir / "99" / "99_0001.wav"
    prebuilt_wav.parent.mkdir(parents=True)
    _write_wav(prebuilt_wav, tone)

    transcript_dir = tmp_path / "transcripts"
    section_words = [
        {"word": "Hello", "start": 0.05, "end": 0.12},
        {"word": "world", "start": 0.13, "end": 0.22},
    ]
    cache = transcript_dir / "99" / "99_0001.json"
    cache.parent.mkdir(parents=True)
    cache.write_text(
        json.dumps({"text": "Hello world", "words": section_words}),
        encoding="utf-8",
    )

    records = process_section(
        book,
        section,
        raw_dir=raw_dir,
        wav_dir=wav_dir,
        chunk_dir=chunk_dir,
        transcript_dir=transcript_dir,
        sample_rate=16000,
        chunk_sec=0.25,
        overlap_sec=0.05,
        transcribe=True,
    )
    assert len(records) >= 1
    assert records[0]["source"] == "librivox"
    assert "audio_path" in records[0]
    assert Path(records[0]["audio_path"]).exists()
    assert records[0]["text"] == "Hello world"
    assert records[0]["words"][0]["word"] == "Hello"