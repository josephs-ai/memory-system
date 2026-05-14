"""
Tests for memory_integration_facade.py — L9-S1 of Level 9.

Covers:
    - context_for_builder() returns prompt_block
    - classify_build_output() returns structured errors
    - record_file_change() + rollback_step() roundtrip
    - parse_test_results() returns structured results
    - index_files() for Python files
    - find_regression() after recording a patch
    - Error resilience (all hooks return safe defaults on failure)
    - Builder packet integration (simulated packet_builder injection)
"""

from __future__ import annotations

import os
import sys

import psycopg
import pytest

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from memory_integration_facade import (
    classify_build_output,
    context_for_builder,
    find_regression,
    index_files,
    parse_test_results,
    record_file_change,
    rollback_step,
    update_after_parse,
)

DB_DSN = os.environ.get("OPENCLAW_MEMORY_DSN", "dbname=openclaw_memory")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def db_conn():
    conn = psycopg.connect(DB_DSN)
    yield conn
    with conn.cursor() as cur:
        cur.execute("DELETE FROM code_chunks WHERE file_path LIKE '/tmp/facade_test/%'")
        cur.execute("DELETE FROM patch_ledger WHERE step_id LIKE 'facade-test-%'")
    conn.commit()
    conn.close()


@pytest.fixture
def test_file(tmp_path):
    f = tmp_path / "facade_test" / "test_module.py"
    f.parent.mkdir(parents=True)
    f.write_text("def hello(): pass\ndef world(): return 42\n")
    return f


# ---------------------------------------------------------------------------
# Tests: context_for_builder
# ---------------------------------------------------------------------------

class TestContextForBuilder:
    def test_returns_prompt_block(self, db_conn):
        # Use a known existing chunk from the real DB
        result = context_for_builder(
            "fix checkpoint failure handling",
            target_qualified_name="checkpoint_db.fail_trigger",
        )
        assert "prompt_block" in result
        assert result["total_chars"] > 0
        assert "fail_trigger" in result["prompt_block"]

    def test_nonexistent_target_returns_no_sections(self):
        result = context_for_builder(
            "fix something",
            target_qualified_name="nonexistent.function.xyz",
        )
        assert result["total_chars"] == 0
        assert len(result["sections"]) == 0

    def test_with_error_groups(self, db_conn):
        errors = [{
            "category": "undeclared_identifier",
            "is_cascade": False,
            "fix_strategy": "Add missing include",
            "root_error": {
                "file": "test.cpp", "line": 10,
                "message": "undeclared 'foo'",
                "chunk_name": None, "notes": [],
            },
        }]
        result = context_for_builder(
            "fix compile error",
            target_qualified_name="checkpoint_db.fail_trigger",
            error_groups=errors,
        )
        assert "undeclared" in result["prompt_block"]

    def test_budget_respected(self):
        result = context_for_builder(
            "fix something",
            target_qualified_name="checkpoint_db.fail_trigger",
            total_budget_chars=500,
        )
        assert result["total_chars"] <= 600


# ---------------------------------------------------------------------------
# Tests: classify_build_output
# ---------------------------------------------------------------------------

class TestClassifyBuildOutput:
    def test_basic_error(self):
        log = "test.cpp:10:5: error: use of undeclared identifier 'foo'"
        result = classify_build_output(log, resolve_chunks=False)
        assert result["summary"]["errors"] == 1
        assert len(result["groups"]) == 1
        assert result["groups"][0]["category"] == "undeclared_identifier"

    def test_empty_log(self):
        result = classify_build_output("", resolve_chunks=False)
        assert result["summary"]["errors"] == 0

    def test_fix_strategy_present(self):
        log = "test.cpp:10:5: error: use of undeclared identifier 'foo'"
        result = classify_build_output(log, resolve_chunks=False)
        assert result["groups"][0]["fix_strategy"] != ""


# ---------------------------------------------------------------------------
# Tests: record + rollback roundtrip
# ---------------------------------------------------------------------------

class TestPatchLedgerFacade:
    def test_record_and_find(self, test_file):
        rec = record_file_change(
            "facade-test-001",
            str(test_file),
            content_after="modified content",
            campaign_id="facade-campaign",
        )
        assert rec["success"]
        assert rec["patch_id"] is not None

        sources = find_regression(str(test_file))
        assert len(sources) >= 1
        assert sources[0]["step_id"] == "facade-test-001"

    def test_rollback(self, test_file):
        original = test_file.read_text()

        record_file_change(
            "facade-test-002",
            str(test_file),
            content_after="modified",
        )
        test_file.write_text("modified")

        result = rollback_step(
            "facade-test-002",
            reparse=False, reembed=False, regraph=False,
        )
        assert result["patches_rolled_back"] == 1
        assert test_file.read_text() == original


# ---------------------------------------------------------------------------
# Tests: parse_test_results
# ---------------------------------------------------------------------------

class TestParseTestResults:
    def test_passing_log(self):
        log = """LogAutomationTest: Test Started: MyGame.Test
LogAutomationTest: Info: PASS: All good
LogAutomationTest: Test Completed: MyGame.Test (Success) in 5.0 ms"""

        result = parse_test_results(log, resolve_chunks=False, link_to_ledger=False)
        assert result["total"] == 1
        assert result["passed"] == 1
        assert result["success"]
        assert "All 1 runtime tests passed" in result["context_block"]

    def test_failing_log(self):
        log = """LogAutomationTest: Test Started: MyGame.Fail
LogAutomationTest: Error: FAIL: expected 0, got 1
LogAutomationTest: Test Completed: MyGame.Fail (Fail)"""

        result = parse_test_results(log, resolve_chunks=False, link_to_ledger=False)
        assert result["failed"] == 1
        assert not result["success"]
        assert "FAIL" in result["context_block"]


# ---------------------------------------------------------------------------
# Tests: index_files
# ---------------------------------------------------------------------------

class TestIndexFiles:
    def test_index_python_file(self, test_file, db_conn):
        stats = index_files([str(test_file)])
        assert stats["files"] == 1
        assert stats["parsed"] > 0
        assert stats["errors"] == 0

    def test_nonexistent_file(self):
        stats = index_files(["/nonexistent/file.py"])
        assert stats["errors"] == 1


# ---------------------------------------------------------------------------
# Tests: Error resilience
# ---------------------------------------------------------------------------

class TestErrorResilience:
    def test_context_bad_dsn(self, monkeypatch):
        """Facade returns safe default on DB connection failure."""
        monkeypatch.setenv("OPENCLAW_MEMORY_DSN", "dbname=nonexistent_db_xyz")
        # Re-import to pick up new DSN
        import importlib
        import memory_integration_facade as mif
        importlib.reload(mif)
        result = mif.context_for_builder("test", target_qualified_name="x.y")
        assert result["prompt_block"] == ""
        assert "error" in result
        # Restore
        monkeypatch.setenv("OPENCLAW_MEMORY_DSN", DB_DSN)
        importlib.reload(mif)


# ---------------------------------------------------------------------------
# Tests: Builder Packet Integration (simulated)
# ---------------------------------------------------------------------------

class TestBuilderPacketIntegration:
    def test_inject_into_packet(self, db_conn):
        """Simulate what packet_builder.py would do."""
        # 1. Get target info from work item metadata
        target_qname = "checkpoint_db.fail_trigger"
        task = "fix checkpoint failure handling"

        # 2. Call the facade
        builder_context = context_for_builder(
            task,
            target_qualified_name=target_qname,
        )

        # 3. Inject into a simulated packet (as packet_builder would)
        packet = {
            "work_id": "work-123",
            "role": "builder",
            "title": task,
            "target_files": [],
            "context_files": [],
        }

        # This is the integration point:
        packet["memory_context_block"] = builder_context["prompt_block"]
        packet["memory_context_meta"] = {
            "total_chars": builder_context["total_chars"],
            "total_tokens_approx": builder_context["total_tokens_approx"],
            "sections": builder_context["sections"],
        }

        # Verify the packet is enriched
        assert packet["memory_context_block"] != ""
        assert "fail_trigger" in packet["memory_context_block"]
        assert packet["memory_context_meta"]["total_tokens_approx"] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
