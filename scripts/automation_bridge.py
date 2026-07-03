"""
automation_bridge.py — UE Automation Tool integration for the Tester agent.

L9-S4 of the Level 9 "Orchestrator Fusion" implementation.

Accepts raw UE Automation Tool (UAT) logs from the execution bridge,
parses them via runtime_test_harness, and produces a token-budgeted
evaluation packet for the Tester agent.

Handles:
    - Normal pass/fail logs → structured results with assertion details
    - Crash logs (SIGSEGV, SIGABRT) → truncated summary with crash context
    - Timeout logs → clean timeout summary
    - Empty/unparseable logs → safe fallback
    - Massive stack traces → aggressively truncated

Token budgeting:
    - Passing tests: count only, no individual detail
    - Failed tests: full assertion details (expected vs actual)
    - Error/crash tests: crash context + truncated stack trace
    - Total tester block capped at configurable limit

Usage:
    from automation_bridge import process_automation_log

    # In the Tester adapter, after receiving UAT output:
    feedback = process_automation_log(
        uat_log=automation_stdout,
        step_id="work-123",
    )
    tester_packet["runtime_feedback"] = feedback["tester_block"]
    tester_packet["runtime_success"] = feedback["success"]
"""

from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger("openclaw.automation_bridge")
logging.basicConfig(
    level=os.environ.get("OPENCLAW_LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

# Token budget constants
TESTER_BLOCK_MAX_CHARS = 6000
PASS_SUMMARY_MAX_CHARS = 200
FAIL_DETAIL_MAX_CHARS = 500       # Per failed test
ERROR_DETAIL_MAX_CHARS = 400      # Per error/crash test
STACK_TRACE_MAX_CHARS = 300       # Per crash stack trace
RAW_FALLBACK_MAX_CHARS = 2000


def process_automation_log(
    uat_log: str,
    *,
    step_id: str | None = None,
    resolve_chunks: bool = True,
    link_to_ledger: bool = True,
    max_failed_shown: int = 10,
    max_errors_shown: int = 5,
) -> dict[str, Any]:
    """
    Process a UE Automation Tool log into a Tester-ready feedback packet.

    Returns:
        {
            "success": bool,
            "tester_block": str,         — Token-budgeted text for Tester prompt
            "structured": dict,          — Full parsed output
            "total": int,
            "passed": int,
            "failed": int,
            "errors": int,
            "chunks_affected": list,
            "regression_sources": dict,
            "processing_ms": float,
        }
    """
    t0 = time.monotonic()

    result: dict[str, Any] = {
        "success": True,
        "tester_block": "",
        "structured": {},
        "total": 0,
        "passed": 0,
        "failed": 0,
        "errors": 0,
        "chunks_affected": [],
        "regression_sources": {},
        "processing_ms": 0.0,
    }

    # --- Edge case: empty log ---
    if not uat_log or not uat_log.strip():
        result["tester_block"] = "Automation test produced no output. The test runner may have failed to start."
        result["success"] = False
        result["processing_ms"] = round((time.monotonic() - t0) * 1000, 1)
        return result

    # --- Edge case: binary/corrupted ---
    if _has_binary_content(uat_log):
        snippet = _safe_truncate(uat_log, RAW_FALLBACK_MAX_CHARS)
        result["success"] = False
        result["tester_block"] = (
            "Automation log appears corrupted or binary. Raw snippet:\n"
            f"```\n{snippet}\n```"
        )
        result["processing_ms"] = round((time.monotonic() - t0) * 1000, 1)
        return result

    # --- Parse ---
    try:
        from runtime_test_harness import (
            parse_test_log,
            resolve_test_failures_to_chunks,
            link_results_to_ledger,
        )

        suite = parse_test_log(uat_log, step_id=step_id)
    except Exception as e:
        LOGGER.error("Automation log parser failed: %s", e)
        snippet = _safe_truncate(uat_log, RAW_FALLBACK_MAX_CHARS)
        result["success"] = False
        result["tester_block"] = (
            f"Automation log parser failed ({e}). Raw snippet:\n"
            f"```\n{snippet}\n```"
        )
        result["processing_ms"] = round((time.monotonic() - t0) * 1000, 1)
        return result

    result["structured"] = suite.to_dict()
    result["total"] = suite.total
    result["passed"] = suite.passed
    result["failed"] = suite.failed
    result["errors"] = suite.errors
    # Treat timeouts and skips-with-issues as failures too
    result["success"] = suite.passed == suite.total and suite.total > 0

    # --- Chunk resolution + ledger linking ---
    if (resolve_chunks or link_to_ledger) and not suite.success:
        try:
            import psycopg
            DB_DSN = (os.environ.get("OPENCLAW_MEMORY_DSN") or os.environ.get("OPENCLAW_MEMORY_DB_DSN") or "dbname=openclaw_memory")  # honor both canonical DSN env names
            conn = psycopg.connect(DB_DSN)
            try:
                if resolve_chunks:
                    resolve_test_failures_to_chunks(conn, suite)
                if link_to_ledger and step_id:
                    result["regression_sources"] = link_results_to_ledger(conn, suite)
            finally:
                conn.close()
        except Exception as e:
            LOGGER.error("Chunk resolution / ledger link failed: %s", e)

    # Collect affected chunks
    for test in suite.tests:
        for cid in test.chunk_ids:
            if cid not in result["chunks_affected"]:
                result["chunks_affected"].append(cid)

    # --- Edge case: no tests found ---
    if suite.total == 0:
        log_lower = uat_log.lower()
        if "crash" in log_lower or "signal" in log_lower or "fatal" in log_lower:
            snippet = _safe_truncate(uat_log, RAW_FALLBACK_MAX_CHARS)
            result["success"] = False
            result["tester_block"] = (
                "Process crashed before any automation tests ran.\n"
                f"```\n{snippet}\n```"
            )
        else:
            result["tester_block"] = "No automation tests were found or executed."
        result["processing_ms"] = round((time.monotonic() - t0) * 1000, 1)
        return result

    # --- Build token-budgeted tester block ---
    result["tester_block"] = _build_tester_block(
        suite,
        regression_sources=result["regression_sources"],
        max_failed=max_failed_shown,
        max_errors=max_errors_shown,
    )

    result["processing_ms"] = round((time.monotonic() - t0) * 1000, 1)
    return result


# ---------------------------------------------------------------------------
# Token-budgeted tester block
# ---------------------------------------------------------------------------

def _build_tester_block(
    suite,
    *,
    regression_sources: dict | None = None,
    max_failed: int = 10,
    max_errors: int = 5,
) -> str:
    """Build the text block injected into the Tester agent's prompt."""
    from runtime_test_harness import TestResult

    parts: list[str] = []
    chars_used = 0

    # --- Summary ---
    if suite.success:
        status = "ALL TESTS PASSED"
    else:
        status = "TESTS FAILED"

    summary = (
        f"{status}: {suite.passed}/{suite.total} passed, "
        f"{suite.failed} failed, {suite.errors} errors"
    )
    if suite.duration_ms:
        summary += f" ({suite.duration_ms:.0f}ms)"
    parts.append(summary)
    parts.append("")
    chars_used += len(summary) + 1

    # --- Passing tests (count only) ---
    passing = [t for t in suite.tests if t.result == TestResult.PASS]
    if passing:
        pass_line = f"Passed ({len(passing)}): " + ", ".join(t.test_name for t in passing[:5])
        if len(passing) > 5:
            pass_line += f" ... (+{len(passing) - 5} more)"
        if len(pass_line) > PASS_SUMMARY_MAX_CHARS:
            pass_line = pass_line[:PASS_SUMMARY_MAX_CHARS] + "..."
        parts.append(pass_line)
        chars_used += len(pass_line)

    # --- Failed tests (full assertion details) ---
    failed = [t for t in suite.tests if t.result == TestResult.FAIL]
    if failed:
        parts.append("")
        parts.append(f"Failed ({len(failed)}):")
        chars_used += 20

        for test in failed[:max_failed]:
            if chars_used >= TESTER_BLOCK_MAX_CHARS:
                parts.append(f"  ... ({len(failed) - failed.index(test)} more failures)")
                break

            block = _format_failed_test(test)
            if chars_used + len(block) > TESTER_BLOCK_MAX_CHARS:
                block = _format_failed_test(test, compact=True)
            parts.append(block)
            chars_used += len(block)

    # --- Error/crash/timeout tests ---
    errored = [t for t in suite.tests if t.result in (TestResult.ERROR, TestResult.TIMEOUT)]
    if errored:
        parts.append("")
        parts.append(f"Errors/Crashes ({len(errored)}):")
        chars_used += 25

        for test in errored[:max_errors]:
            if chars_used >= TESTER_BLOCK_MAX_CHARS:
                parts.append(f"  ... ({len(errored) - errored.index(test)} more errors)")
                break

            block = _format_error_test(test)
            parts.append(block)
            chars_used += len(block)

    # --- Regression sources ---
    if regression_sources:
        parts.append("")
        parts.append("Regression candidates:")
        for test_name, step_ids in list(regression_sources.items())[:5]:
            parts.append(f"  {test_name} → steps: {', '.join(step_ids[:3])}")

    return "\n".join(parts)


def _format_failed_test(test, *, compact: bool = False) -> str:
    """Format a single failed test for the tester block."""
    lines = [f"  ✗ {test.test_name}"]

    if compact:
        if test.errors:
            lines.append(f"    {test.errors[0][:120]}")
        return "\n".join(lines)

    # Full assertion details
    if test.assertions_total > 0:
        lines.append(
            f"    Assertions: {test.assertions_passed}/{test.assertions_total} passed"
        )

    for err in test.errors[:5]:
        # Cap individual error lines
        err_line = err[:250] if len(err) > 250 else err
        lines.append(f"    FAIL: {err_line}")

    if test.duration_ms:
        lines.append(f"    Duration: {test.duration_ms:.0f}ms")

    if test.chunk_ids:
        lines.append(f"    Related chunks: {', '.join(test.chunk_ids[:3])}")

    result = "\n".join(lines)
    if len(result) > FAIL_DETAIL_MAX_CHARS:
        result = result[:FAIL_DETAIL_MAX_CHARS] + "\n    ... [truncated]"
    return result


def _format_error_test(test) -> str:
    """Format an error/crash/timeout test for the tester block."""
    from runtime_test_harness import TestResult

    if test.result == TestResult.TIMEOUT:
        label = "TIMEOUT"
    else:
        label = "CRASH/ERROR"

    lines = [f"  ✗ {test.test_name} [{label}]"]

    for err in test.errors[:3]:
        err_line = err[:200] if len(err) > 200 else err
        lines.append(f"    {err_line}")

    # Include last few info messages as context for crashes
    if test.result == TestResult.ERROR and test.messages:
        last_msgs = test.messages[-3:]
        for msg in last_msgs:
            lines.append(f"    Context: {msg[:150]}")

    result = "\n".join(lines)
    if len(result) > ERROR_DETAIL_MAX_CHARS:
        result = result[:ERROR_DETAIL_MAX_CHARS] + "\n    ... [truncated]"
    return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _has_binary_content(text: str) -> bool:
    sample = text[:1000]
    if "\x00" in sample:
        return True
    non_printable = sum(1 for c in sample if ord(c) < 32 and c not in "\n\r\t")
    return non_printable / max(len(sample), 1) > 0.1


def _safe_truncate(text: str, max_chars: int) -> str:
    cleaned = text.replace("\x00", "")
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[:max_chars] + "\n... [truncated]"
