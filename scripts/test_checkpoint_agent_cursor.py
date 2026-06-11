"""
Tests for S1 (cursor) + S2 (classification + replay delta) + S3 (delta gating).
"""
import json
import pytest
from pathlib import Path
import checkpoint_agent as ca


def write_lines(path, lines):
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_fresh_start(tmp_path):
    """No prior cursor → fresh_start."""
    transcript = tmp_path / "session.jsonl"
    write_lines(transcript, ['{"type":"message","role":"user","content":"hello"}'])
    result = ca.classify_transcript(transcript, None)
    assert result["classification"] == "fresh_start"
    assert result["committed_offset"] == 0


def test_append_happy_path(tmp_path):
    """Same inode, head hash stable, size grew → same_file_append."""
    transcript = tmp_path / "session.jsonl"
    write_lines(transcript, ['{"type":"message","role":"user","content":"hello"}'])
    cursor = ca.build_cursor(transcript)
    # Append more content
    with open(transcript, "a") as f:
        f.write('{"type":"message","role":"assistant","content":"hi"}\n')
    result = ca.classify_transcript(transcript, cursor)
    assert result["classification"] == "same_file_append"
    assert result["committed_offset"] == cursor["committed_offset"]


def test_truncation(tmp_path):
    """File shrank → same_file_truncated → committed_offset reset to 0."""
    transcript = tmp_path / "session.jsonl"
    write_lines(transcript, ['{"type":"message","role":"user","content":"hello"}'] * 10)
    cursor = ca.build_cursor(transcript)
    # Truncate
    write_lines(transcript, ['{"type":"message","role":"user","content":"hi"}'])
    result = ca.classify_transcript(transcript, cursor)
    assert result["classification"] == "same_file_truncated"
    assert result["committed_offset"] == 0


def test_rotate_or_replace(tmp_path):
    """Different inode → rotated_or_replaced."""
    transcript = tmp_path / "session.jsonl"
    write_lines(transcript, ['{"type":"message","role":"user","content":"hello"}'])
    cursor = ca.build_cursor(transcript)
    # Delete and recreate (new inode)
    transcript.unlink()
    write_lines(transcript, ['{"type":"message","role":"user","content":"new session"}'])
    result = ca.classify_transcript(transcript, cursor)
    assert result["classification"] in ("rotated_or_replaced", "unknown_identity_change")
    assert result["committed_offset"] == 0


def test_cursor_version_mismatch(tmp_path):
    """Wrong cursor version → unknown_identity_change."""
    transcript = tmp_path / "session.jsonl"
    write_lines(transcript, ['{"type":"message","role":"user","content":"hello"}'])
    cursor = ca.build_cursor(transcript)
    cursor["cursor_version"] = "v1-old"
    result = ca.classify_transcript(transcript, cursor)
    assert result["classification"] == "unknown_identity_change"


def test_read_safe_replay_delta(tmp_path):
    """Delta read returns only new content past committed_offset."""
    transcript = tmp_path / "session.jsonl"
    initial = '{"type":"message","role":"user","content":"hello"}\n'
    transcript.write_text(initial, encoding="utf-8")
    cursor = ca.build_cursor(transcript, processed_offset=len(initial))
    committed = cursor["committed_offset"]

    new_line = '{"type":"message","role":"assistant","content":"world"}\n'
    with open(transcript, "a") as f:
        f.write(new_line)

    delta_text, replay_start, safe_end = ca.read_safe_replay_delta(transcript, committed)
    assert new_line in delta_text
    assert safe_end > committed


def test_can_use_delta_true():
    assert ca.can_use_delta("same_file_append", 100, {"size": 200}) is True


def test_can_use_delta_false_fresh():
    assert ca.can_use_delta("fresh_start", 0, {"size": 200}) is False


def test_can_use_delta_false_no_new_bytes():
    assert ca.can_use_delta("same_file_append", 200, {"size": 200}) is False


def test_save_load_cursor(tmp_path, monkeypatch):
    """Round-trip cursor through save/load."""
    monkeypatch.setattr(ca, "STATE_DIR", tmp_path)
    transcript = tmp_path / "session.jsonl"
    write_lines(transcript, ['{"type":"message"}'])
    cursor = ca.build_cursor(transcript)
    ca.save_cursor("test-agent", cursor)
    loaded = ca.load_cursor("test-agent")
    assert loaded["cursor_version"] == ca.CURSOR_VERSION
    assert loaded["committed_offset"] == cursor["committed_offset"]
