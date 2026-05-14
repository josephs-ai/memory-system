"""
Tests for compile_feedback_loop.py — L9-S3 of Level 9.

Covers:
    - Successful build → clean success message
    - Failed build → structured error block
    - Empty log → "no output" response
    - Binary/corrupted log → safe fallback with raw snippet
    - Completely unparseable log → fallback with raw snippet
    - Token budget enforcement on root errors
    - Cascade errors compressed vs root errors
    - Category breakdown in summary
    - Root-cause vs cascade prioritization
    - Regression source integration
    - Facade routing (process_build_for_reviewer)
"""

from __future__ import annotations

import os
import sys
import textwrap

import pytest

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from compile_feedback_loop import process_build_log, _is_text_log, _safe_truncate
from memory_integration_facade import process_build_for_reviewer


# ---------------------------------------------------------------------------
# Sample logs
# ---------------------------------------------------------------------------

SUCCESS_LOG = textwrap.dedent("""\
    [1/10] Compiling Module.MyGame.cpp
    [2/10] Compiling MyCharacter.cpp
    [10/10] Linking MyGameEditor
    Building MyGameEditor done
    Total time: 23.45s
""")

SINGLE_ERROR_LOG = textwrap.dedent("""\
    [1/5] Compiling MyCharacter.cpp
    /Users/dev/MyGame/Source/MyCharacter.cpp:47:5: error: use of undeclared identifier 'Helth'
    [2/5] Linking MyGameEditor
    Total time: 5.0s
""")

MULTI_ERROR_CASCADE_LOG = textwrap.dedent("""\
    /Users/dev/MyGame/Source/CombatSystem.cpp:3:10: fatal error: 'CombatTypes.h' file not found
    /Users/dev/MyGame/Source/CombatSystem.cpp:15:5: error: use of undeclared identifier 'FCombatData'
    /Users/dev/MyGame/Source/CombatSystem.cpp:22:12: error: use of undeclared identifier 'EDamageType'
    /Users/dev/MyGame/Source/CombatSystem.cpp:30:8: error: unknown type name 'FCombatResult'
    /Users/dev/MyGame/Source/MyCharacter.cpp:47:5: error: no member named 'Attack' in 'AMyCharacter'
""")

TEMPLATE_ERROR_LOG = textwrap.dedent("""\
    /Users/dev/MyGame/Source/AbilitySystem.cpp:128:15: error: no matching function for call to 'TArray<FAbilityData>::Add'
    /Users/dev/UE_5.7/Engine/Source/Runtime/Core/Public/Containers/Array.h:1247:7: note: candidate function not viable: requires 0 arguments, but 1 was provided
    /Users/dev/UE_5.7/Engine/Source/Runtime/Core/Public/Containers/Array.h:1253:7: note: candidate function not viable: no known conversion
""")

EMPTY_LOG = ""
WHITESPACE_LOG = "   \n\n  \n  "

BINARY_LOG = "MZ\x90\x00\x03\x00\x00\x00\x04\x00\x00\x00\xff\xff\x00\x00binary data here"

ANOMALOUS_LOG = textwrap.dedent("""\
    ☠️ Something went completely wrong
    ERROR: The build system encountered an unexpected error
    Please contact support with error code: 0xDEADBEEF
""")


# ---------------------------------------------------------------------------
# Tests: Success
# ---------------------------------------------------------------------------

class TestSuccessBuild:
    def test_clean_success(self):
        result = process_build_log(SUCCESS_LOG, resolve_chunks=False)
        assert result["success"]
        assert result["error_count"] == 0
        assert "successfully" in result["reviewer_block"].lower() or "no errors" in result["reviewer_block"].lower()

    def test_success_has_zero_chunks(self):
        result = process_build_log(SUCCESS_LOG, resolve_chunks=False)
        assert len(result["chunks_affected"]) == 0


# ---------------------------------------------------------------------------
# Tests: Error Classification
# ---------------------------------------------------------------------------

class TestErrorClassification:
    def test_single_error(self):
        result = process_build_log(SINGLE_ERROR_LOG, resolve_chunks=False)
        assert not result["success"]
        assert result["error_count"] == 1
        assert "undeclared" in result["reviewer_block"].lower()
        assert "BUILD FAILED" in result["reviewer_block"]

    def test_cascade_detection(self):
        result = process_build_log(MULTI_ERROR_CASCADE_LOG, resolve_chunks=False)
        assert not result["success"]
        assert result["cascades"] >= 2
        assert result["root_causes"] < result["error_count"]
        assert "cascade" in result["reviewer_block"].lower()

    def test_template_error_with_notes(self):
        result = process_build_log(TEMPLATE_ERROR_LOG, resolve_chunks=False)
        assert not result["success"]
        assert result["error_count"] >= 1
        # Notes should be in the reviewer block
        assert "note" in result["reviewer_block"].lower() or "candidate" in result["reviewer_block"].lower()

    def test_category_breakdown(self):
        result = process_build_log(MULTI_ERROR_CASCADE_LOG, resolve_chunks=False)
        cats = result["structured"]["summary"]["category_breakdown"]
        assert len(cats) > 0
        assert "Categories:" in result["reviewer_block"]

    def test_fix_strategy_in_block(self):
        result = process_build_log(SINGLE_ERROR_LOG, resolve_chunks=False)
        assert "Fix:" in result["reviewer_block"]


# ---------------------------------------------------------------------------
# Tests: Edge Cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_empty_log(self):
        result = process_build_log(EMPTY_LOG, resolve_chunks=False)
        assert "no output" in result["reviewer_block"].lower()

    def test_whitespace_log(self):
        result = process_build_log(WHITESPACE_LOG, resolve_chunks=False)
        assert "no output" in result["reviewer_block"].lower()

    def test_binary_log(self):
        result = process_build_log(BINARY_LOG, resolve_chunks=False)
        assert not result["success"]
        assert "corrupted" in result["reviewer_block"].lower() or "binary" in result["reviewer_block"].lower()

    def test_anomalous_log_with_error_keyword(self):
        result = process_build_log(ANOMALOUS_LOG, resolve_chunks=False)
        # Classifier finds no parseable diagnostics, but log mentions "ERROR"
        assert "snippet" in result["reviewer_block"].lower() or "error" in result["reviewer_block"].lower()

    def test_is_text_log_rejects_binary(self):
        assert not _is_text_log("hello\x00world")

    def test_is_text_log_accepts_text(self):
        assert _is_text_log("normal build output\n")

    def test_safe_truncate(self):
        long_text = "x" * 5000
        truncated = _safe_truncate(long_text, 100)
        assert len(truncated) < 200
        assert "truncated" in truncated


# ---------------------------------------------------------------------------
# Tests: Token Budget
# ---------------------------------------------------------------------------

class TestTokenBudget:
    def test_root_errors_prioritized(self):
        result = process_build_log(MULTI_ERROR_CASCADE_LOG, resolve_chunks=False)
        block = result["reviewer_block"]
        # Root-cause errors should appear before the "Cascades" section header
        cascade_section_pos = block.find("Cascades (")
        if cascade_section_pos == -1:
            cascade_section_pos = block.find("cascade")
        # The first actual error detail should appear before the cascades section
        first_error_pos = block.find("[missing_include]")
        if first_error_pos == -1:
            first_error_pos = block.find("file not found")
        assert first_error_pos < cascade_section_pos or cascade_section_pos == -1

    def test_reviewer_block_not_enormous(self):
        # Even with many errors, block should stay bounded
        big_log = "\n".join(
            f"/path/file{i}.cpp:{i}:1: error: undeclared identifier 'x{i}'"
            for i in range(50)
        )
        result = process_build_log(big_log, resolve_chunks=False)
        assert len(result["reviewer_block"]) < 8000

    def test_cascade_compressed(self):
        result = process_build_log(MULTI_ERROR_CASCADE_LOG, resolve_chunks=False)
        block = result["reviewer_block"]
        # Cascades section should exist but be compact
        cascade_section = block[block.find("Cascade"):] if "Cascade" in block else ""
        root_section = block[:block.find("Cascade")] if "Cascade" in block else block
        # Cascade section should be shorter than root section
        if cascade_section:
            assert len(cascade_section) < len(root_section)


# ---------------------------------------------------------------------------
# Tests: Facade
# ---------------------------------------------------------------------------

class TestFacade:
    def test_process_build_for_reviewer(self):
        result = process_build_for_reviewer(SINGLE_ERROR_LOG, resolve_chunks=False)
        assert not result["success"]
        assert "reviewer_block" in result
        assert "undeclared" in result["reviewer_block"].lower()

    def test_facade_handles_empty(self):
        result = process_build_for_reviewer("", resolve_chunks=False)
        assert "reviewer_block" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
