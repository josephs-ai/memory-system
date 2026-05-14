"""
Tests for automation_bridge.py — L9-S4 of Level 9.

Covers:
    - All tests pass → clean success summary
    - Mixed pass/fail → failed assertion details in tester block
    - Crash mid-test → truncated crash context
    - Timeout → clean timeout summary
    - Process crash (no tests) → fallback with raw snippet
    - Empty log → safe response
    - Binary log → safe fallback
    - Parser failure → fallback with raw snippet
    - Token budget enforcement (many failures stay bounded)
    - Passing tests compressed (count + names only)
    - Failed tests show exact assertion lines (expected vs actual)
    - Facade routing (process_automation_for_tester)
"""

from __future__ import annotations

import os
import sys
import textwrap

import pytest

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from automation_bridge import process_automation_log
from memory_integration_facade import process_automation_for_tester


# ---------------------------------------------------------------------------
# Sample UAT logs
# ---------------------------------------------------------------------------

ALL_PASS_LOG = textwrap.dedent("""\
    LogAutomationTest: Test Started: MyGame.Combat.DamageTest
    LogAutomationTest: Info: PASS: Damage applied correctly (got 70.0)
    LogAutomationTest: Info: PASS: Health never negative (got 0.0)
    LogAutomationTest: Test Completed: MyGame.Combat.DamageTest (Success) in 12.0 ms
    LogAutomationTest: Test Started: MyGame.Movement.DashTest
    LogAutomationTest: Info: PASS: Dash distance correct (got 500.0)
    LogAutomationTest: Test Completed: MyGame.Movement.DashTest (Success) in 8.0 ms
""")

MIXED_LOG = textwrap.dedent("""\
    LogAutomationTest: Test Started: MyGame.Combat.DamageTest
    LogAutomationTest: Info: PASS: Damage applied correctly
    LogAutomationTest: Test Completed: MyGame.Combat.DamageTest (Success) in 10.0 ms
    LogAutomationTest: Test Started: MyGame.Combat.HealTest
    LogAutomationTest: Error: FAIL: Heal restores health — expected 100.0, got 50.0
    LogAutomationTest: Error: FAIL: Max health cap — expected 100.0, got 150.0
    LogAutomationTest: Test Completed: MyGame.Combat.HealTest (Fail) in 5.0 ms
    LogAutomationTest: Test Started: MyGame.AI.PatrolTest
    LogAutomationTest: Info: PASS: Enemy spawns correctly
    LogAutomationTest: Test Completed: MyGame.AI.PatrolTest (Success) in 20.0 ms
""")

CRASH_LOG = textwrap.dedent("""\
    LogAutomationTest: Test Started: MyGame.AI.SpawnTest
    LogAutomationTest: Info: Spawning enemy at location (100, 200, 0)
    LogAutomationTest: Info: Enemy spawned successfully
    Signal 11 (SIGSEGV) caught by crash handler
    Fatal error: [File:Runtime/Core/Private/HAL/PlatformProcess.cpp] [Line: 42]
    Stack trace:
      0x00007fff AEnemyBase::BeginPlay()
      0x00007ffe AGameMode::StartMatch()
      0x00007ffd main()
""")

TIMEOUT_LOG = textwrap.dedent("""\
    LogAutomationTest: Test Started: MyGame.World.LoadTest
    LogAutomationTest: Info: Loading level...
    LogAutomationTest: Info: Waiting for level load...
    LogAutomationTest: Warning: Test timed out after 30 seconds
""")

PROCESS_CRASH_LOG = textwrap.dedent("""\
    Launching UnrealEditor-Cmd...
    LogInit: Build: ++UE5+Release-5.7
    Signal 6 (SIGABRT) caught
    Assertion failed: GEngine != nullptr
""")

MANY_FAILURES_LOG = "\n".join(
    f"LogAutomationTest: Test Started: MyGame.Test{i}\n"
    f"LogAutomationTest: Error: FAIL: assertion {i} — expected 1, got 0\n"
    f"LogAutomationTest: Test Completed: MyGame.Test{i} (Fail) in 1.0 ms"
    for i in range(30)
)

EMPTY_LOG = ""
BINARY_LOG = "\x00\x01\x02binary\xffdata"


# ---------------------------------------------------------------------------
# Tests: Success
# ---------------------------------------------------------------------------

class TestAllPass:
    def test_success_flag(self):
        result = process_automation_log(ALL_PASS_LOG, resolve_chunks=False, link_to_ledger=False)
        assert result["success"]
        assert result["passed"] == 2
        assert result["failed"] == 0

    def test_tester_block_shows_pass(self):
        result = process_automation_log(ALL_PASS_LOG, resolve_chunks=False, link_to_ledger=False)
        assert "ALL TESTS PASSED" in result["tester_block"]
        assert "2/2" in result["tester_block"]

    def test_pass_names_listed(self):
        result = process_automation_log(ALL_PASS_LOG, resolve_chunks=False, link_to_ledger=False)
        assert "DamageTest" in result["tester_block"]
        assert "DashTest" in result["tester_block"]


# ---------------------------------------------------------------------------
# Tests: Failures
# ---------------------------------------------------------------------------

class TestFailures:
    def test_failure_detected(self):
        result = process_automation_log(MIXED_LOG, resolve_chunks=False, link_to_ledger=False)
        assert not result["success"]
        assert result["failed"] == 1
        assert result["passed"] == 2

    def test_assertion_details_in_block(self):
        result = process_automation_log(MIXED_LOG, resolve_chunks=False, link_to_ledger=False)
        block = result["tester_block"]
        assert "HealTest" in block
        assert "expected 100.0, got 50.0" in block
        assert "FAIL:" in block

    def test_assertion_count(self):
        result = process_automation_log(MIXED_LOG, resolve_chunks=False, link_to_ledger=False)
        heal_test = next(t for t in result["structured"]["tests"] if "Heal" in t["test_name"])
        assert heal_test["assertions_failed"] >= 1


# ---------------------------------------------------------------------------
# Tests: Crash / Timeout
# ---------------------------------------------------------------------------

class TestCrashTimeout:
    def test_crash_detected(self):
        result = process_automation_log(CRASH_LOG, resolve_chunks=False, link_to_ledger=False)
        assert not result["success"]
        assert result["errors"] >= 1

    def test_crash_context_in_block(self):
        result = process_automation_log(CRASH_LOG, resolve_chunks=False, link_to_ledger=False)
        block = result["tester_block"]
        assert "CRASH" in block or "ERROR" in block
        # Should include last info messages as context
        assert "Spawning" in block or "spawned" in block.lower()

    def test_timeout_detected(self):
        result = process_automation_log(TIMEOUT_LOG, resolve_chunks=False, link_to_ledger=False)
        assert not result["success"]
        # Timeout counts as an error in the structured output
        assert result["total"] >= 1

    def test_timeout_in_block(self):
        result = process_automation_log(TIMEOUT_LOG, resolve_chunks=False, link_to_ledger=False)
        assert "TIMEOUT" in result["tester_block"]

    def test_process_crash_no_tests(self):
        result = process_automation_log(PROCESS_CRASH_LOG, resolve_chunks=False, link_to_ledger=False)
        assert not result["success"]
        assert "crashed" in result["tester_block"].lower()


# ---------------------------------------------------------------------------
# Tests: Edge Cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_empty_log(self):
        result = process_automation_log(EMPTY_LOG, resolve_chunks=False, link_to_ledger=False)
        assert not result["success"]
        assert "no output" in result["tester_block"].lower()

    def test_binary_log(self):
        result = process_automation_log(BINARY_LOG, resolve_chunks=False, link_to_ledger=False)
        assert not result["success"]
        assert "corrupted" in result["tester_block"].lower() or "binary" in result["tester_block"].lower()


# ---------------------------------------------------------------------------
# Tests: Token Budget
# ---------------------------------------------------------------------------

class TestTokenBudget:
    def test_many_failures_bounded(self):
        result = process_automation_log(MANY_FAILURES_LOG, resolve_chunks=False, link_to_ledger=False)
        assert result["failed"] == 30
        # Block should stay bounded
        assert len(result["tester_block"]) < 8000

    def test_passing_tests_compressed(self):
        result = process_automation_log(ALL_PASS_LOG, resolve_chunks=False, link_to_ledger=False)
        block = result["tester_block"]
        # Passing tests should be a single compact line
        pass_lines = [l for l in block.split("\n") if "Passed" in l]
        assert len(pass_lines) <= 1


# ---------------------------------------------------------------------------
# Tests: Facade
# ---------------------------------------------------------------------------

class TestFacade:
    def test_process_automation_for_tester(self):
        result = process_automation_for_tester(MIXED_LOG, resolve_chunks=False, link_to_ledger=False)
        assert not result["success"]
        assert "tester_block" in result
        assert "HealTest" in result["tester_block"]

    def test_facade_handles_empty(self):
        result = process_automation_for_tester("", resolve_chunks=False, link_to_ledger=False)
        assert "tester_block" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
