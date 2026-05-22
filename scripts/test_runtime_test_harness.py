"""
Tests for runtime_test_harness.py — L8-S4 of Level 8.

Covers:
    - Test spec creation and validation
    - C++ test code generation (deterministic, template-based)
    - UE automation log parsing (pass, fail, error, timeout, crash)
    - Assertion extraction (pass/fail with expected/actual)
    - Suite summary statistics
    - Multi-test log parsing
    - Crash detection
    - Timeout detection
    - Chunk resolution for test failures
    - Patch ledger regression linking
    - Context formatting for Builder agents
    - Edge cases (empty log, no tests, partial log)
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

from runtime_test_harness import (
    TestAssertion,
    TestResult,
    TestSpec,
    format_for_context,
    generate_test_cpp,
    parse_test_log,
    resolve_test_failures_to_chunks,
    spec_from_dict,
)

DB_DSN = os.environ.get("OPENCLAW_MEMORY_DSN", "dbname=openclaw_memory")


# ---------------------------------------------------------------------------
# Sample UE automation logs
# ---------------------------------------------------------------------------

PASSING_LOG = textwrap.dedent("""\
    LogAutomationTest: Test Started: MyGame.Combat.DamageTest
    LogAutomationTest: Info: Setting up combat test...
    LogAutomationTest: Info: PASS: Damage reduces health (got 50.0)
    LogAutomationTest: Info: PASS: Health never goes negative (got 0.0)
    LogAutomationTest: Test Completed: MyGame.Combat.DamageTest (Success) in 12.5 ms
""")

FAILING_LOG = textwrap.dedent("""\
    LogAutomationTest: Test Started: MyGame.Combat.HealTest
    LogAutomationTest: Info: Setting up heal test...
    LogAutomationTest: Error: FAIL: Heal restores health — expected 100.0, got 50.0
    LogAutomationTest: Test Completed: MyGame.Combat.HealTest (Fail) in 8.3 ms
""")

MULTI_TEST_LOG = textwrap.dedent("""\
    LogAutomationTest: Test Started: MyGame.Combat.DamageTest
    LogAutomationTest: Info: PASS: Damage applied correctly
    LogAutomationTest: Test Completed: MyGame.Combat.DamageTest (Success) in 10.0 ms
    LogAutomationTest: Test Started: MyGame.Combat.HealTest
    LogAutomationTest: Error: FAIL: Heal amount wrong — expected 100.0, got 75.0
    LogAutomationTest: Test Completed: MyGame.Combat.HealTest (Fail) in 5.0 ms
    LogAutomationTest: Test Started: MyGame.Movement.DashTest
    LogAutomationTest: Info: PASS: Dash distance correct
    LogAutomationTest: Info: PASS: Dash cooldown respected
    LogAutomationTest: Test Completed: MyGame.Movement.DashTest (Success) in 15.0 ms
    LogAutomationController: 2 tests passed, 1 tests failed
""")

CRASH_LOG = textwrap.dedent("""\
    LogAutomationTest: Test Started: MyGame.AI.SpawnTest
    LogAutomationTest: Info: Spawning enemy...
    Signal 11 (SIGSEGV) caught by crash handler
    Fatal error: [File:Runtime/Core/Private/HAL/PlatformProcess.cpp] [Line: 42]
""")

TIMEOUT_LOG = textwrap.dedent("""\
    LogAutomationTest: Test Started: MyGame.World.LoadTest
    LogAutomationTest: Info: Loading room...
    LogAutomationTest: Warning: Test timed out after 30 seconds
""")

EMPTY_LOG = ""

PROCESS_CRASH_LOG = textwrap.dedent("""\
    Launching UnrealEditor...
    Signal 6 (SIGABRT) caught
    Assertion failed: IsValid()
""")


# ---------------------------------------------------------------------------
# Tests: Test Spec
# ---------------------------------------------------------------------------

class TestSpecCreation:
    def test_from_dict(self):
        d = {
            "test_name": "MyGame.Combat.Test",
            "category": "Combat",
            "description": "Test combat",
            "setup_code": "auto* Char = SpawnCharacter();",
            "assertions": [
                {"description": "Health is 100", "expression": "Char->GetHealth()", "expected": "100.0f"},
            ],
        }
        spec = spec_from_dict(d)
        assert spec.test_name == "MyGame.Combat.Test"
        assert len(spec.assertions) == 1
        assert spec.assertions[0].expected == "100.0f"

    def test_defaults(self):
        d = {"test_name": "MyGame.Test"}
        spec = spec_from_dict(d)
        assert spec.category == "General"
        assert spec.timeout_seconds == 30.0


# ---------------------------------------------------------------------------
# Tests: C++ Code Generation
# ---------------------------------------------------------------------------

class TestCodeGeneration:
    def test_generates_valid_cpp(self):
        spec = TestSpec(
            test_name="MyGame.Combat.DamageTest",
            category="Combat",
            description="Test damage",
            setup_code="auto* Char = SpawnCharacter();",
            assertions=[
                TestAssertion(
                    description="Health reduced",
                    expression="Char->GetHealth()",
                    expected="50.0f",
                ),
            ],
        )
        cpp = generate_test_cpp(spec)
        assert "IMPLEMENT_SIMPLE_AUTOMATION_TEST" in cpp
        assert "MyGame.Combat.DamageTest" in cpp
        assert "RunTest" in cpp
        assert "Health reduced" in cpp

    def test_boolean_assertion(self):
        spec = TestSpec(
            test_name="MyGame.Test",
            category="General",
            description="Bool test",
            setup_code="",
            assertions=[
                TestAssertion(
                    description="Is alive",
                    expression="Char->IsAlive()",
                    expected="true",
                ),
            ],
        )
        cpp = generate_test_cpp(spec)
        # Should use the boolean template
        assert "if (Char->IsAlive())" in cpp

    def test_extra_includes(self):
        spec = TestSpec(test_name="MyGame.Test", category="G", description="",
                        setup_code="", assertions=[])
        cpp = generate_test_cpp(spec, extra_includes=["MyCharacter.h", "CombatSystem.h"])
        assert '#include "MyCharacter.h"' in cpp
        assert '#include "CombatSystem.h"' in cpp

    def test_deterministic(self):
        spec = TestSpec(test_name="MyGame.Test", category="G", description="",
                        setup_code="int x = 1;", assertions=[])
        cpp1 = generate_test_cpp(spec)
        cpp2 = generate_test_cpp(spec)
        assert cpp1 == cpp2


# ---------------------------------------------------------------------------
# Tests: Log Parsing
# ---------------------------------------------------------------------------

class TestLogParsing:
    def test_passing_test(self):
        suite = parse_test_log(PASSING_LOG)
        assert suite.total == 1
        assert suite.passed == 1
        assert suite.failed == 0
        assert suite.success
        assert suite.tests[0].result == TestResult.PASS
        assert suite.tests[0].duration_ms == 12.5

    def test_failing_test(self):
        suite = parse_test_log(FAILING_LOG)
        assert suite.total == 1
        assert suite.failed == 1
        assert not suite.success
        assert suite.tests[0].result == TestResult.FAIL
        assert len(suite.tests[0].errors) >= 1
        assert "expected 100.0, got 50.0" in suite.tests[0].errors[0]

    def test_multi_test_log(self):
        suite = parse_test_log(MULTI_TEST_LOG)
        assert suite.total == 3
        assert suite.passed == 2
        assert suite.failed == 1

        results = {t.test_name: t.result for t in suite.tests}
        assert results["MyGame.Combat.DamageTest"] == TestResult.PASS
        assert results["MyGame.Combat.HealTest"] == TestResult.FAIL
        assert results["MyGame.Movement.DashTest"] == TestResult.PASS

    def test_crash_detection(self):
        suite = parse_test_log(CRASH_LOG)
        assert suite.total == 1
        assert suite.errors == 1
        assert suite.tests[0].result == TestResult.ERROR

    def test_timeout_detection(self):
        suite = parse_test_log(TIMEOUT_LOG)
        assert suite.total == 1
        assert suite.tests[0].result == TestResult.TIMEOUT

    def test_empty_log(self):
        suite = parse_test_log(EMPTY_LOG)
        assert suite.total == 0
        assert suite.success

    def test_process_crash_no_tests(self):
        suite = parse_test_log(PROCESS_CRASH_LOG)
        assert suite.total == 1
        assert suite.errors == 1
        assert suite.tests[0].test_name == "<process_crash>"

    def test_assertion_counting(self):
        suite = parse_test_log(PASSING_LOG)
        test = suite.tests[0]
        assert test.assertions_passed == 2
        assert test.assertions_total == 2
        assert test.assertions_failed == 0

    def test_failed_assertion_details(self):
        suite = parse_test_log(FAILING_LOG)
        test = suite.tests[0]
        assert test.assertions_failed >= 1

    def test_step_id_propagated(self):
        suite = parse_test_log(PASSING_LOG, step_id="step-42")
        assert suite.step_id == "step-42"

    def test_duration_summed(self):
        suite = parse_test_log(MULTI_TEST_LOG)
        assert suite.duration_ms == 30.0  # 10 + 5 + 15


# ---------------------------------------------------------------------------
# Tests: Output Formats
# ---------------------------------------------------------------------------

class TestOutputFormats:
    def test_to_dict(self):
        suite = parse_test_log(MULTI_TEST_LOG)
        d = suite.to_dict()
        assert d["total"] == 3
        assert d["passed"] == 2
        assert len(d["tests"]) == 3

    def test_summary(self):
        suite = parse_test_log(MULTI_TEST_LOG)
        summary = suite.summary()
        assert "2/3 passed" in summary
        assert "1 failed" in summary
        assert "✓" in summary
        assert "✗" in summary

    def test_format_for_context_pass(self):
        suite = parse_test_log(PASSING_LOG)
        ctx = format_for_context(suite)
        assert "All 1 runtime tests passed" in ctx

    def test_format_for_context_fail(self):
        suite = parse_test_log(MULTI_TEST_LOG)
        ctx = format_for_context(suite)
        assert "FAIL:" in ctx
        assert "HealTest" in ctx


# ---------------------------------------------------------------------------
# Tests: Database Integration
# ---------------------------------------------------------------------------

@pytest.fixture
def db_conn():
    conn = psycopg.connect(DB_DSN)
    yield conn
    with conn.cursor() as cur:
        cur.execute("DELETE FROM code_chunks WHERE file_path LIKE '/test/harness/%'")
    conn.commit()
    conn.close()


class TestChunkResolution:
    def test_resolve_failures_to_chunks(self, db_conn):
        """Test failures with matching keywords resolve to chunks."""
        from parse_code_ast import upsert_chunks

        chunks = [{
            "id": "cc-test-combat-resolve",
            "file_path": "/test/harness/Combat.py",
            "file_hash": "h", "chunk_type": "class",
            "qualified_name": "Combat.CombatSystem",
            "module_name": "Combat",
            "signature": "class CombatSystem:",
            "docstring": None, "body": "class CombatSystem: pass",
            "decorators": None, "imports": [],
            "line_start": 1, "line_end": 10,
            "parent_chunk_id": None, "class_name": None,
        }]
        upsert_chunks(db_conn, chunks)

        suite = parse_test_log(FAILING_LOG)
        resolved = resolve_test_failures_to_chunks(db_conn, suite)
        assert resolved >= 1
        assert len(suite.tests[0].chunk_ids) >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
