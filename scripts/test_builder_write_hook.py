"""
Tests for builder_write_hook.py — L9-S2 of Level 9.

Covers:
    - before_write snapshots content
    - after_write parses + embeds + updates ledger
    - handle_builder_write full lifecycle
    - Incremental hash-skip (unchanged file not re-parsed)
    - Python file indexing
    - C++ file indexing
    - Non-parseable file gracefully skipped
    - Patch ledger records correct before/after state
    - Rollback after handle_builder_write restores original
    - Facade hooks route correctly
"""

from __future__ import annotations

import os
import sys

import psycopg
import pytest

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from builder_write_hook import before_write, after_write, handle_builder_write
from memory_integration_facade import (
    before_builder_write,
    after_builder_write,
    handle_builder_write as facade_handle,
)

DB_DSN = os.environ.get("OPENCLAW_MEMORY_DSN", "dbname=openclaw_memory")


@pytest.fixture
def db_conn():
    conn = psycopg.connect(DB_DSN)
    yield conn
    with conn.cursor() as cur:
        cur.execute("DELETE FROM code_chunks WHERE file_path LIKE '/tmp/%'")
        cur.execute("DELETE FROM patch_ledger WHERE step_id LIKE 'bwh-test-%'")
    conn.commit()
    conn.close()


@pytest.fixture
def py_file(tmp_path):
    f = tmp_path / "module.py"
    f.write_text("def original(): pass\n")
    return f


@pytest.fixture
def cpp_file(tmp_path):
    f = tmp_path / "module.cpp"
    f.write_text("void original() { return; }\n")
    return f


@pytest.fixture
def txt_file(tmp_path):
    f = tmp_path / "readme.txt"
    f.write_text("hello world")
    return f


# ---------------------------------------------------------------------------
# Tests: before_write
# ---------------------------------------------------------------------------

class TestBeforeWrite:
    def test_snapshots_existing_file(self, py_file, db_conn):
        result = before_write("bwh-test-001", str(py_file))
        assert result["success"]
        assert result["file_existed"]
        assert result["file_hash_before"] is not None
        assert result["patch_id"] is not None

    def test_handles_new_file(self, tmp_path, db_conn):
        new_file = tmp_path / "new.py"
        result = before_write("bwh-test-002", str(new_file))
        assert result["success"]
        assert not result["file_existed"]
        assert result["file_hash_before"] is None


# ---------------------------------------------------------------------------
# Tests: after_write
# ---------------------------------------------------------------------------

class TestAfterWrite:
    def test_indexes_python_file(self, py_file, db_conn):
        before_write("bwh-test-010", str(py_file))
        py_file.write_text("def modified(): return 42\ndef new_func(): pass\n")

        result = after_write("bwh-test-010", str(py_file))
        assert result["success"]
        assert result["chunks_parsed"] > 0
        assert result["file_hash_after"] is not None

    def test_indexes_cpp_file(self, cpp_file, db_conn):
        before_write("bwh-test-011", str(cpp_file))
        cpp_file.write_text("void modified() { return; }\nvoid added() {}\n")

        result = after_write("bwh-test-011", str(cpp_file))
        assert result["success"]
        assert result["chunks_parsed"] > 0

    def test_skips_non_parseable(self, txt_file, db_conn):
        before_write("bwh-test-012", str(txt_file))
        result = after_write("bwh-test-012", str(txt_file))
        assert result["success"]
        assert result["skipped"]

    def test_hash_skip_unchanged(self, py_file, db_conn):
        # First write — will parse
        before_write("bwh-test-013", str(py_file))
        after_write("bwh-test-013", str(py_file), force_reparse=True)

        # Second write with same content — should skip
        before_write("bwh-test-014", str(py_file))
        r2 = after_write("bwh-test-014", str(py_file))
        assert r2["skipped"]

    def test_updates_ledger_chunk_ids(self, py_file, db_conn):
        before_write("bwh-test-015", str(py_file))
        py_file.write_text("def a(): pass\ndef b(): pass\n")
        after_write("bwh-test-015", str(py_file), force_reparse=True)

        from patch_ledger import PatchLedger
        conn = psycopg.connect(DB_DSN)
        try:
            ledger = PatchLedger(conn)
            patches = ledger.get_step_patches("bwh-test-015")
            assert len(patches) == 1
            assert len(patches[0].chunk_ids_after) > 0
        finally:
            conn.close()


# ---------------------------------------------------------------------------
# Tests: handle_builder_write (full lifecycle)
# ---------------------------------------------------------------------------

class TestHandleBuilderWrite:
    def test_full_lifecycle_python(self, py_file, db_conn):
        result = handle_builder_write(
            "bwh-test-020",
            str(py_file),
            "def rewritten(): return 99\n",
            campaign_id="test-campaign",
        )
        assert result["success"]
        assert result["before"]["file_existed"]
        assert result["after"]["chunks_parsed"] > 0
        assert py_file.read_text() == "def rewritten(): return 99\n"

    def test_full_lifecycle_new_file(self, tmp_path, db_conn):
        new_file = tmp_path / "brand_new.py"
        result = handle_builder_write(
            "bwh-test-021",
            str(new_file),
            "def brand_new(): pass\n",
        )
        assert result["success"]
        assert new_file.exists()
        assert not result["before"]["file_existed"]

    def test_rollback_after_write(self, py_file, db_conn):
        original = py_file.read_text()

        handle_builder_write(
            "bwh-test-022",
            str(py_file),
            "def changed(): pass\n",
        )
        assert py_file.read_text() == "def changed(): pass\n"

        # Rollback via facade
        from memory_integration_facade import rollback_step
        rb = rollback_step("bwh-test-022", reparse=False, reembed=False, regraph=False)
        assert rb["patches_rolled_back"] == 1
        assert py_file.read_text() == original


# ---------------------------------------------------------------------------
# Tests: Facade routing
# ---------------------------------------------------------------------------

class TestFacadeRouting:
    def test_before_builder_write(self, py_file, db_conn):
        result = before_builder_write("bwh-test-030", str(py_file))
        assert result["success"]

    def test_after_builder_write(self, py_file, db_conn):
        before_builder_write("bwh-test-031", str(py_file))
        py_file.write_text("def x(): pass\n")
        result = after_builder_write("bwh-test-031", str(py_file))
        assert result["success"]

    def test_facade_handle(self, py_file, db_conn):
        result = facade_handle(
            "bwh-test-032",
            str(py_file),
            "def via_facade(): pass\n",
        )
        assert result["success"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
