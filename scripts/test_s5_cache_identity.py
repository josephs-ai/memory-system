from __future__ import annotations

import json
import importlib.util
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
MODULE_PATH = SCRIPT_DIR / "extract_chunk_updates.py"

spec = importlib.util.spec_from_file_location("extract_chunk_updates", MODULE_PATH)
extract_chunk_updates = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(extract_chunk_updates)


def test_cursor_present_key_stability_ignores_volatile_session_name(tmp_path: Path):
    chunk_file = tmp_path / "001.txt"
    chunk_file.write_text("hello\n", encoding="utf-8")
    cursor_data = {
        "head_hash": "abcdef1234567890fedcba",
        "inode": 4242,
    }

    key_a = extract_chunk_updates.cache_key(
        chunk_file,
        source_agent="main",
        source_session="123e4567-e89b-12d3-a456-426614174000.topic-a.jsonl",
        legacy_pipeline=False,
        extractor_fingerprint=extract_chunk_updates.EXTRACTOR_FINGERPRINT,
        cursor_data=cursor_data,
    )
    key_b = extract_chunk_updates.cache_key(
        chunk_file,
        source_agent="main",
        source_session="987e6543-e21b-12d3-a456-426614174999.topic-b.jsonl",
        legacy_pipeline=False,
        extractor_fingerprint=extract_chunk_updates.EXTRACTOR_FINGERPRINT,
        cursor_data=cursor_data,
    )

    assert key_a == key_b
    payload = json.loads(key_a)
    assert payload["session_family"] == "abcdef123456-4242"


def test_cursor_absent_fallback_uses_filename_stem_split(tmp_path: Path):
    chunk_file = tmp_path / "001.txt"
    chunk_file.write_text("hello\n", encoding="utf-8")

    key = extract_chunk_updates.cache_key(
        chunk_file,
        source_agent="main",
        source_session="session-123.topic.jsonl",
        legacy_pipeline=False,
        extractor_fingerprint=extract_chunk_updates.EXTRACTOR_FINGERPRINT,
        cursor_data=None,
    )

    payload = json.loads(key)
    assert payload["session_family"] == "session-123"


def test_inode_rotation_invalidates_key(tmp_path: Path):
    chunk_file = tmp_path / "001.txt"
    chunk_file.write_text("hello\n", encoding="utf-8")

    key_a = extract_chunk_updates.cache_key(
        chunk_file,
        source_agent="main",
        source_session="same-name.jsonl",
        legacy_pipeline=False,
        extractor_fingerprint=extract_chunk_updates.EXTRACTOR_FINGERPRINT,
        cursor_data={"head_hash": "abcdef1234567890", "inode": 100},
    )
    key_b = extract_chunk_updates.cache_key(
        chunk_file,
        source_agent="main",
        source_session="same-name.jsonl",
        legacy_pipeline=False,
        extractor_fingerprint=extract_chunk_updates.EXTRACTOR_FINGERPRINT,
        cursor_data={"head_hash": "abcdef1234567890", "inode": 101},
    )

    assert key_a != key_b


def test_stale_cursor_graceful_fallback(tmp_path: Path):
    chunk_file = tmp_path / "001.txt"
    chunk_file.write_text("hello\n", encoding="utf-8")
    bad_cursor = tmp_path / "broken.cursor.json"
    bad_cursor.write_text("{not-json", encoding="utf-8")

    cursor_data = extract_chunk_updates.load_cursor_file(str(bad_cursor))
    assert cursor_data is None

    key = extract_chunk_updates.cache_key(
        chunk_file,
        source_agent="main",
        source_session="rotating-session.topic.jsonl",
        legacy_pipeline=False,
        extractor_fingerprint=extract_chunk_updates.EXTRACTOR_FINGERPRINT,
        cursor_data=cursor_data,
    )

    payload = json.loads(key)
    assert payload["session_family"] == "rotating-session"
