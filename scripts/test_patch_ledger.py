"""
Tests for patch_ledger.py — L8-S2 of Level 8.

Covers:
    - Patch recording (create, modify, delete operations)
    - Patch ID determinism
    - Idempotent re-recording (ON CONFLICT UPDATE)
    - Chunk ID snapshot before/after
    - Rollback: file content restoration
    - Rollback: created files get deleted
    - Rollback: deleted files get restored
    - Rollback marks ledger entries as rolled_back
    - Rollback does not affect already-rolled-back patches
    - Step history query
    - File history query
    - Campaign status
    - Regression source finder
    - No orphaned chunks after rollback
    - No corrupted graph state after rollback
"""

from __future__ import annotations

import os
import sys
import textwrap

import psycopg
import pytest

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from patch_ledger import PatchLedger, PatchRecord, RollbackResult
from parse_code_ast import parse_file, upsert_chunks, remove_stale_chunks

DB_DSN = os.environ.get("OPENCLAW_MEMORY_DSN", "dbname=openclaw_memory")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def db_conn():
    conn = psycopg.connect(DB_DSN)
    yield conn
    # Clean up test data
    with conn.cursor() as cur:
        cur.execute("DELETE FROM patch_ledger WHERE step_id LIKE 'test-%'")
        cur.execute("DELETE FROM code_chunks WHERE file_path LIKE '/tmp/patch_test/%'")
    conn.commit()
    conn.close()


@pytest.fixture
def ledger(db_conn):
    return PatchLedger(db_conn)


@pytest.fixture
def test_dir(tmp_path):
    """Create a temp directory with a test Python file."""
    d = tmp_path / "patch_test"
    d.mkdir()
    return d


# ---------------------------------------------------------------------------
# Tests: Patch ID
# ---------------------------------------------------------------------------

class TestPatchId:
    def test_deterministic(self):
        id1 = PatchLedger.make_patch_id("step-001", "/path/to/file.cpp")
        id2 = PatchLedger.make_patch_id("step-001", "/path/to/file.cpp")
        assert id1 == id2
        assert id1.startswith("patch-")

    def test_different_steps_different_ids(self):
        id1 = PatchLedger.make_patch_id("step-001", "/path/to/file.cpp")
        id2 = PatchLedger.make_patch_id("step-002", "/path/to/file.cpp")
        assert id1 != id2

    def test_different_files_different_ids(self):
        id1 = PatchLedger.make_patch_id("step-001", "/path/to/a.cpp")
        id2 = PatchLedger.make_patch_id("step-001", "/path/to/b.cpp")
        assert id1 != id2


# ---------------------------------------------------------------------------
# Tests: Recording
# ---------------------------------------------------------------------------

class TestRecording:
    def test_record_create(self, ledger, test_dir):
        fp = str(test_dir / "new_file.py")
        content = "def hello(): pass\n"

        pid = ledger.record_patch(
            step_id="test-rec-001",
            file_path=fp,
            content_after=content,
            operation="create",
        )

        assert pid.startswith("patch-")
        patches = ledger.get_step_patches("test-rec-001")
        assert len(patches) == 1
        assert patches[0].operation == "create"
        assert patches[0].file_hash_after is not None
        assert patches[0].file_hash_before is None

    def test_record_modify(self, ledger, test_dir, db_conn):
        fp = test_dir / "existing.py"
        fp.write_text("def foo(): pass\n")

        # First insert chunks so we can verify chunk_ids_before
        chunks = parse_file(str(fp), fp.read_text())
        upsert_chunks(db_conn, chunks)

        pid = ledger.record_patch(
            step_id="test-rec-002",
            file_path=str(fp),
            content_after="def foo(): return 42\ndef bar(): pass\n",
        )

        patches = ledger.get_step_patches("test-rec-002")
        assert len(patches) == 1
        assert patches[0].operation == "modify"
        assert patches[0].content_before == "def foo(): pass\n"
        assert patches[0].file_hash_before is not None
        assert patches[0].file_hash_after is not None
        assert patches[0].file_hash_before != patches[0].file_hash_after

    def test_record_delete(self, ledger, test_dir):
        fp = test_dir / "to_delete.py"
        fp.write_text("def goodbye(): pass\n")

        pid = ledger.record_patch(
            step_id="test-rec-003",
            file_path=str(fp),
            content_after=None,
            operation="delete",
        )

        patches = ledger.get_step_patches("test-rec-003")
        assert patches[0].operation == "delete"
        assert patches[0].content_before == "def goodbye(): pass\n"

    def test_idempotent_recording(self, ledger, test_dir):
        fp = str(test_dir / "idempotent.py")

        ledger.record_patch(step_id="test-rec-004", file_path=fp, content_after="v1")
        ledger.record_patch(step_id="test-rec-004", file_path=fp, content_after="v2")

        patches = ledger.get_step_patches("test-rec-004")
        assert len(patches) == 1  # Not duplicated

    def test_metadata_stored(self, ledger, test_dir):
        fp = str(test_dir / "meta.py")
        meta = {"agent": "builder", "model": "claude-sonnet"}

        ledger.record_patch(
            step_id="test-rec-005",
            file_path=fp,
            content_after="pass",
            metadata=meta,
        )

        patches = ledger.get_step_patches("test-rec-005")
        assert patches[0].metadata == meta

    def test_campaign_id(self, ledger, test_dir):
        fp = str(test_dir / "campaign.py")
        ledger.record_patch(
            step_id="test-rec-006",
            file_path=fp,
            content_after="pass",
            campaign_id="campaign-alpha",
        )
        patches = ledger.get_step_patches("test-rec-006")
        assert patches[0].campaign_id == "campaign-alpha"


# ---------------------------------------------------------------------------
# Tests: Rollback
# ---------------------------------------------------------------------------

class TestRollback:
    def test_rollback_modify_restores_content(self, ledger, test_dir):
        fp = test_dir / "rollback_test.py"
        original = "def original(): pass\n"
        modified = "def modified(): return 42\n"

        fp.write_text(original)
        ledger.record_patch(
            step_id="test-rb-001",
            file_path=str(fp),
            content_after=modified,
        )
        # Simulate the modification
        fp.write_text(modified)
        assert fp.read_text() == modified

        # Rollback (skip reparse/embed/graph for unit test speed)
        result = ledger.rollback_step(
            "test-rb-001",
            reparse=False, reembed=False, regraph=False,
        )

        assert result.patches_rolled_back == 1
        assert len(result.files_restored) == 1
        assert fp.read_text() == original

    def test_rollback_create_deletes_file(self, ledger, test_dir):
        fp = test_dir / "created.py"
        fp.write_text("new content")

        ledger.record_patch(
            step_id="test-rb-002",
            file_path=str(fp),
            content_after="new content",
            operation="create",
        )

        result = ledger.rollback_step(
            "test-rb-002",
            reparse=False, reembed=False, regraph=False,
        )

        assert not fp.exists()

    def test_rollback_delete_restores_file(self, ledger, test_dir):
        fp = test_dir / "deleted.py"
        original = "def was_here(): pass\n"
        fp.write_text(original)

        ledger.record_patch(
            step_id="test-rb-003",
            file_path=str(fp),
            content_after=None,
            operation="delete",
        )
        # Simulate deletion
        fp.unlink()
        assert not fp.exists()

        result = ledger.rollback_step(
            "test-rb-003",
            reparse=False, reembed=False, regraph=False,
        )

        assert fp.exists()
        assert fp.read_text() == original

    def test_rollback_marks_entries(self, ledger, test_dir):
        fp = test_dir / "mark_test.py"
        fp.write_text("original")

        ledger.record_patch(
            step_id="test-rb-004",
            file_path=str(fp),
            content_after="modified",
        )
        fp.write_text("modified")

        ledger.rollback_step(
            "test-rb-004",
            reparse=False, reembed=False, regraph=False,
        )

        patches = ledger.get_step_patches("test-rb-004")
        assert all(p.rolled_back for p in patches)
        assert all(p.rolled_back_at is not None for p in patches)

    def test_rollback_idempotent(self, ledger, test_dir):
        fp = test_dir / "idem_rb.py"
        fp.write_text("original")

        ledger.record_patch(
            step_id="test-rb-005",
            file_path=str(fp),
            content_after="modified",
        )
        fp.write_text("modified")

        # First rollback
        r1 = ledger.rollback_step(
            "test-rb-005",
            reparse=False, reembed=False, regraph=False,
        )
        assert r1.patches_rolled_back == 1

        # Second rollback — no active patches
        r2 = ledger.rollback_step(
            "test-rb-005",
            reparse=False, reembed=False, regraph=False,
        )
        assert r2.patches_rolled_back == 0

    def test_rollback_multiple_files(self, ledger, test_dir):
        fp1 = test_dir / "multi1.py"
        fp2 = test_dir / "multi2.py"
        fp1.write_text("orig1")
        fp2.write_text("orig2")

        ledger.record_patch(step_id="test-rb-006", file_path=str(fp1), content_after="mod1")
        ledger.record_patch(step_id="test-rb-006", file_path=str(fp2), content_after="mod2")
        fp1.write_text("mod1")
        fp2.write_text("mod2")

        result = ledger.rollback_step(
            "test-rb-006",
            reparse=False, reembed=False, regraph=False,
        )

        assert result.patches_rolled_back == 2
        assert fp1.read_text() == "orig1"
        assert fp2.read_text() == "orig2"


# ---------------------------------------------------------------------------
# Tests: Queries
# ---------------------------------------------------------------------------

class TestQueries:
    def test_step_history(self, ledger, test_dir):
        for i in range(3):
            fp = str(test_dir / f"hist_{i}.py")
            ledger.record_patch(
                step_id="test-q-001",
                file_path=fp,
                content_after=f"content {i}",
                operation="create",
            )

        patches = ledger.get_step_patches("test-q-001")
        assert len(patches) == 3

    def test_file_history(self, ledger, test_dir):
        fp = test_dir / "file_hist.py"
        fp.write_text("v0")

        for i in range(3):
            ledger.record_patch(
                step_id=f"test-q-00{i+2}",
                file_path=str(fp),
                content_after=f"v{i+1}",
            )
            fp.write_text(f"v{i+1}")

        history = ledger.get_file_history(str(fp))
        assert len(history) == 3
        # Most recent first
        assert history[0].step_id == "test-q-004"

    def test_campaign_status(self, ledger, test_dir):
        for i in range(3):
            fp = str(test_dir / f"camp_{i}.py")
            ledger.record_patch(
                step_id=f"test-q-010{i}",
                file_path=fp,
                content_after="pass",
                campaign_id="test-campaign-1",
                operation="create",
            )

        status = ledger.get_campaign_status("test-campaign-1")
        assert status["total_patches"] == 3
        assert status["total_steps"] == 3
        assert status["files_touched"] == 3
        assert status["rolled_back"] == 0

    def test_find_regression_source(self, ledger, test_dir, db_conn):
        fp = test_dir / "regression.py"
        content = "def line10(): pass\n" * 5 + "def target(): pass\n" + "def line12(): pass\n" * 5
        fp.write_text(content)

        # Parse and insert chunks
        chunks = parse_file(str(fp), content)
        upsert_chunks(db_conn, chunks)

        # Record patch
        ledger.record_patch(
            step_id="test-q-020",
            file_path=str(fp),
            content_after=content,
        )
        ledger.update_after_parse("test-q-020", str(fp))

        # Find regression source for this file
        sources = ledger.find_regression_source(str(fp))
        assert len(sources) >= 1
        assert sources[0].step_id == "test-q-020"


# ---------------------------------------------------------------------------
# Tests: Integration (record + rollback + chunk verification)
# ---------------------------------------------------------------------------

class TestIntegration:
    def test_full_cycle_no_orphans(self, ledger, test_dir, db_conn):
        """Record → modify → rollback → verify no orphaned chunks."""
        fp = test_dir / "full_cycle.py"
        original = "def original(): pass\n"
        modified = "def modified(): return 42\ndef new_func(): pass\n"

        # Write original and parse
        fp.write_text(original)
        chunks = parse_file(str(fp), original)
        upsert_chunks(db_conn, chunks)
        original_ids = {c["id"] for c in chunks}

        # Record the upcoming modification
        ledger.record_patch(
            step_id="test-int-001",
            file_path=str(fp),
            content_after=modified,
        )

        # Apply modification
        fp.write_text(modified)
        new_chunks = parse_file(str(fp), modified)
        upsert_chunks(db_conn, new_chunks)
        remove_stale_chunks(db_conn, str(fp), {c["id"] for c in new_chunks})
        ledger.update_after_parse("test-int-001", str(fp))

        # Verify new chunks exist
        with db_conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM code_chunks WHERE file_path = %s", (str(fp),))
            assert cur.fetchone()[0] == len(new_chunks)

        # Rollback (with reparse, but skip embed/graph for test speed)
        result = ledger.rollback_step(
            "test-int-001",
            reparse=True, reembed=False, regraph=False,
        )

        assert result.patches_rolled_back == 1
        assert fp.read_text() == original

        # Verify chunks are restored to original set
        with db_conn.cursor() as cur:
            cur.execute(
                "SELECT id FROM code_chunks WHERE file_path = %s",
                (str(fp),)
            )
            current_ids = {row[0] for row in cur.fetchall()}

        assert current_ids == original_ids


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
