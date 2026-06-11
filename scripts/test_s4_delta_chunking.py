"""
S4 delta-aware chunking tests.
Verifies that chunk_by_topic._load_text respects byte offsets.
"""
import pytest
from pathlib import Path
import chunk_by_topic as cbt


def test_load_text_full(tmp_path):
    f = tmp_path / "t.txt"
    f.write_text("line1\nline2\nline3\n", encoding="utf-8")
    text = cbt._load_text(f, offset=0)
    assert "line1" in text
    assert "line3" in text


def test_load_text_offset(tmp_path):
    f = tmp_path / "t.txt"
    content = "line1\nline2\nline3\n"
    f.write_text(content, encoding="utf-8")
    offset = len("line1\n")
    text = cbt._load_text(f, offset=offset)
    assert "line1" not in text
    assert "line2" in text
    assert "line3" in text


def test_load_text_offset_zero_boundary(tmp_path):
    """offset=0 should return full content."""
    f = tmp_path / "t.txt"
    f.write_text("hello world\n", encoding="utf-8")
    assert cbt._load_text(f, offset=0) == "hello world\n"


def test_load_text_malformed_tail(tmp_path):
    """Partial UTF-8 / malformed tail should not crash."""
    f = tmp_path / "t.bin"
    f.write_bytes(b"good line\n\xff\xfe bad bytes")
    text = cbt._load_text(f, offset=0)
    assert "good line" in text
