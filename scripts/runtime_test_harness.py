"""
runtime_test_harness.py — UE Automation Test generation and result parsing.

L8-S4 of the Level 8 "Autonomous Closed-Loop Execution" implementation.

Two responsibilities:
1. Generate deterministic UE automation test C++ files from structured
   test specifications (no LLM required — template-based generation)
2. Parse UE automation test output logs into structured results that
   integrate with the patch ledger and task context assembler

UE Automation Test output format (from Unreal Automation Tool):
    LogAutomationTest: Test Started: MyGame.Combat.DamageTest
    LogAutomationTest: Info: Setting up combat test...
    LogAutomationTest: Info: Applied 50.0 damage
    LogAutomationTest: Success: Health correctly reduced to 50.0
    LogAutomationTest: Test Completed: MyGame.Combat.DamageTest (Success)

    LogAutomationTest: Test Started: MyGame.Combat.DeathTest
    LogAutomationTest: Error: Expected Health to be 0.0, got 50.0
    LogAutomationTest: Test Completed: MyGame.Combat.DeathTest (Fail)

    LogAutomationController: X tests passed, Y tests failed

Usage:
    # Generate a test file
    python runtime_test_harness.py generate --spec test_spec.json --output TestCombat.cpp

    # Parse test results
    python runtime_test_harness.py parse --log automation_output.log [--json]

    # Generate + parse in one shot (for orchestrator)
    python runtime_test_harness.py report --log automation_output.log [--step-id S1]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import sys
import textwrap
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

import psycopg

LOGGER = logging.getLogger("openclaw.runtime_test_harness")
logging.basicConfig(
    level=os.environ.get("OPENCLAW_LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

DB_DSN = os.environ.get("OPENCLAW_MEMORY_DSN", "dbname=openclaw_memory")


# ---------------------------------------------------------------------------
# Test specification types
# ---------------------------------------------------------------------------

class TestResult(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    ERROR = "error"
    TIMEOUT = "timeout"
    SKIPPED = "skipped"


@dataclass
class TestAssertion:
    """A single assertion within a test."""
    description: str
    expression: str          # C++ expression to evaluate
    expected: str            # Expected value (as string)
    actual: str | None = None
    passed: bool | None = None
    message: str | None = None


@dataclass
class TestSpec:
    """Specification for a single automation test."""
    test_name: str           # e.g., "MyGame.Combat.DamageTest"
    category: str            # e.g., "Combat", "Movement", "AI"
    description: str
    setup_code: str          # C++ code to run before assertions
    assertions: list[TestAssertion] = field(default_factory=list)
    teardown_code: str = ""  # C++ cleanup code
    timeout_seconds: float = 30.0
    tags: list[str] = field(default_factory=list)


@dataclass
class TestRunResult:
    """Result of a single test execution."""
    test_name: str
    result: TestResult
    duration_ms: float | None = None
    assertions_total: int = 0
    assertions_passed: int = 0
    assertions_failed: int = 0
    messages: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    chunk_ids: list[str] = field(default_factory=list)  # Related code chunks


@dataclass
class TestSuiteResult:
    """Result of an entire test suite run."""
    tests: list[TestRunResult] = field(default_factory=list)
    total: int = 0
    passed: int = 0
    failed: int = 0
    errors: int = 0
    skipped: int = 0
    duration_ms: float = 0.0
    raw_log: str = ""
    step_id: str | None = None
    timestamp: str = ""

    @property
    def success(self) -> bool:
        return self.failed == 0 and self.errors == 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "passed": self.passed,
            "failed": self.failed,
            "errors": self.errors,
            "skipped": self.skipped,
            "success": self.success,
            "duration_ms": self.duration_ms,
            "step_id": self.step_id,
            "timestamp": self.timestamp,
            "tests": [
                {
                    "test_name": t.test_name,
                    "result": t.result.value,
                    "duration_ms": t.duration_ms,
                    "assertions_total": t.assertions_total,
                    "assertions_passed": t.assertions_passed,
                    "assertions_failed": t.assertions_failed,
                    "messages": t.messages[:5],
                    "errors": t.errors,
                    "chunk_ids": t.chunk_ids,
                }
                for t in self.tests
            ],
        }

    def summary(self) -> str:
        status = "PASS" if self.success else "FAIL"
        lines = [
            f"[{status}] {self.passed}/{self.total} passed, {self.failed} failed, {self.errors} errors",
        ]
        for t in self.tests:
            mark = "✓" if t.result == TestResult.PASS else "✗"
            lines.append(f"  {mark} {t.test_name}: {t.result.value}")
            for e in t.errors[:3]:
                lines.append(f"      ERROR: {e}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Test C++ code generation (template-based, deterministic)
# ---------------------------------------------------------------------------

# UE Automation Test template
UE_TEST_TEMPLATE = textwrap.dedent("""\
    // Auto-generated by runtime_test_harness.py — DO NOT EDIT
    // Test: {test_name}
    // Category: {category}
    // Description: {description}

    #include "CoreMinimal.h"
    #include "Misc/AutomationTest.h"
    {extra_includes}

    IMPLEMENT_SIMPLE_AUTOMATION_TEST(
        F{safe_name},
        "{test_name}",
        EAutomationTestFlags::ApplicationContextMask | EAutomationTestFlags::ProductFilter
    )

    bool F{safe_name}::RunTest(const FString& Parameters)
    {{
        // Setup
        {setup_code}

        // Assertions
        {assertion_code}

        // Teardown
        {teardown_code}

        return true;
    }}
""")

ASSERTION_TEMPLATE = textwrap.dedent("""\
    // Assertion: {description}
    {{
        auto _Actual = {expression};
        auto _Expected = {expected};
        if (_Actual == _Expected) {{
            AddInfo(FString::Printf(TEXT("PASS: {description} (got %s)"), *FString::SanitizeFloat((float)_Actual)));
        }} else {{
            AddError(FString::Printf(TEXT("FAIL: {description} — expected %s, got %s"),
                *FString::SanitizeFloat((float)_Expected), *FString::SanitizeFloat((float)_Actual)));
        }}
    }}
""")

BOOL_ASSERTION_TEMPLATE = textwrap.dedent("""\
    // Assertion: {description}
    if ({expression}) {{
        AddInfo(TEXT("PASS: {description}"));
    }} else {{
        AddError(TEXT("FAIL: {description}"));
    }}
""")


def generate_test_cpp(spec: TestSpec, extra_includes: list[str] | None = None) -> str:
    """Generate a UE automation test .cpp file from a TestSpec."""
    safe_name = re.sub(r"[^A-Za-z0-9]", "", spec.test_name)

    # Build assertion code
    assertion_lines = []
    for asrt in spec.assertions:
        if asrt.expected.lower() in ("true", "false"):
            code = BOOL_ASSERTION_TEMPLATE.format(
                description=asrt.description.replace('"', '\\"'),
                expression=asrt.expression,
            )
        else:
            code = ASSERTION_TEMPLATE.format(
                description=asrt.description.replace('"', '\\"'),
                expression=asrt.expression,
                expected=asrt.expected,
            )
        assertion_lines.append(code)

    includes_str = "\n".join(f'#include "{inc}"' for inc in (extra_includes or []))

    return UE_TEST_TEMPLATE.format(
        test_name=spec.test_name,
        safe_name=safe_name,
        category=spec.category,
        description=spec.description.replace('"', '\\"'),
        setup_code=textwrap.indent(spec.setup_code, "    ") if spec.setup_code else "// No setup",
        assertion_code=textwrap.indent("\n".join(assertion_lines), "    "),
        teardown_code=textwrap.indent(spec.teardown_code, "    ") if spec.teardown_code else "// No teardown",
        extra_includes=includes_str,
    )


def spec_from_dict(d: dict) -> TestSpec:
    """Create a TestSpec from a dictionary."""
    assertions = [
        TestAssertion(
            description=a["description"],
            expression=a["expression"],
            expected=a.get("expected", "true"),
        )
        for a in d.get("assertions", [])
    ]
    return TestSpec(
        test_name=d["test_name"],
        category=d.get("category", "General"),
        description=d.get("description", ""),
        setup_code=d.get("setup_code", ""),
        assertions=assertions,
        teardown_code=d.get("teardown_code", ""),
        timeout_seconds=d.get("timeout_seconds", 30.0),
        tags=d.get("tags", []),
    )


# ---------------------------------------------------------------------------
# Test result parsing
# ---------------------------------------------------------------------------

# UE automation log patterns
RE_TEST_START = re.compile(
    r"LogAutomationTest.*Test Started:\s*(?P<name>.+?)$", re.M
)
RE_TEST_COMPLETE = re.compile(
    r"LogAutomationTest.*Test Completed:\s*(?P<name>.+?)\s*\((?P<result>\w+)\)(?:\s*in\s*(?P<ms>[\d.]+)\s*ms)?",
    re.M,
)
RE_TEST_INFO = re.compile(
    r"LogAutomationTest.*(?:Info|Display):\s*(?P<msg>.+?)$", re.M
)
RE_TEST_ERROR = re.compile(
    r"LogAutomationTest.*Error:\s*(?P<msg>.+?)$", re.M
)
RE_TEST_WARNING = re.compile(
    r"LogAutomationTest.*Warning:\s*(?P<msg>.+?)$", re.M
)
RE_SUITE_SUMMARY = re.compile(
    r"(?P<passed>\d+)\s*tests?\s*passed.*?(?P<failed>\d+)\s*tests?\s*failed", re.I
)
RE_ASSERTION_FAIL = re.compile(
    r"FAIL:\s*(?P<desc>.+?)(?:\s*—\s*expected\s+(?P<expected>.+?),\s*got\s+(?P<actual>.+))?$"
)
RE_ASSERTION_PASS = re.compile(
    r"PASS:\s*(?P<desc>.+?)(?:\s*\(got\s+(?P<actual>.+?)\))?$"
)

# Crash/timeout patterns
RE_CRASH = re.compile(
    r"(?:Signal \d+|Assertion failed|Fatal error|Unhandled Exception|SIGABRT|SIGSEGV)", re.I
)
RE_TIMEOUT = re.compile(
    r"(?:Test timed out|Timeout after|exceeded time limit)", re.I
)


def parse_test_log(log_text: str, *, step_id: str | None = None) -> TestSuiteResult:
    """
    Parse UE automation test output into a TestSuiteResult.

    Handles:
    - Individual test start/complete markers
    - Info, error, and warning messages within each test
    - Assertion pass/fail extraction
    - Suite-level summary
    - Crash and timeout detection
    """
    suite = TestSuiteResult(
        raw_log=log_text,
        step_id=step_id,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )

    # Split log into per-test segments
    test_segments = _split_into_test_segments(log_text)

    for test_name, segment in test_segments.items():
        result = _parse_test_segment(test_name, segment)
        suite.tests.append(result)

    # Suite totals
    suite.total = len(suite.tests)
    suite.passed = sum(1 for t in suite.tests if t.result == TestResult.PASS)
    suite.failed = sum(1 for t in suite.tests if t.result == TestResult.FAIL)
    suite.errors = sum(1 for t in suite.tests if t.result == TestResult.ERROR)
    suite.skipped = sum(1 for t in suite.tests if t.result == TestResult.SKIPPED)
    suite.duration_ms = sum(t.duration_ms or 0 for t in suite.tests)

    # Check for crash outside any test
    if RE_CRASH.search(log_text) and suite.total == 0:
        suite.tests.append(TestRunResult(
            test_name="<process_crash>",
            result=TestResult.ERROR,
            errors=["Process crashed before any test completed"],
        ))
        suite.total = 1
        suite.errors = 1

    return suite


def _split_into_test_segments(log_text: str) -> dict[str, str]:
    """Split a log into per-test segments based on Start/Complete markers."""
    segments: dict[str, str] = {}
    current_test: str | None = None
    current_lines: list[str] = []

    for line in log_text.splitlines():
        m_start = RE_TEST_START.search(line)
        if m_start:
            if current_test and current_lines:
                segments[current_test] = "\n".join(current_lines)
            current_test = m_start.group("name").strip()
            current_lines = [line]
            continue

        m_complete = RE_TEST_COMPLETE.search(line)
        if m_complete and current_test:
            current_lines.append(line)
            segments[current_test] = "\n".join(current_lines)
            current_test = None
            current_lines = []
            continue

        if current_test:
            current_lines.append(line)

    # Handle unterminated test (crash mid-test)
    if current_test and current_lines:
        segments[current_test] = "\n".join(current_lines)

    return segments


def _parse_test_segment(test_name: str, segment: str) -> TestRunResult:
    """Parse a single test's log segment."""
    result = TestRunResult(test_name=test_name, result=TestResult.PASS)

    # Check completion status
    m_complete = RE_TEST_COMPLETE.search(segment)
    if m_complete:
        status = m_complete.group("result").lower()
        if status == "fail":
            result.result = TestResult.FAIL
        elif status in ("error", "critical"):
            result.result = TestResult.ERROR
        elif status == "skip":
            result.result = TestResult.SKIPPED

        ms_str = m_complete.group("ms")
        if ms_str:
            result.duration_ms = float(ms_str)
    else:
        # No completion marker — likely crash or timeout
        if RE_TIMEOUT.search(segment):
            result.result = TestResult.TIMEOUT
        elif RE_CRASH.search(segment):
            result.result = TestResult.ERROR
            result.errors.append("Test crashed without completion")
        else:
            result.result = TestResult.ERROR
            result.errors.append("Test did not complete (no completion marker)")

    # Extract messages
    for m in RE_TEST_INFO.finditer(segment):
        msg = m.group("msg").strip()
        result.messages.append(msg)

        # Check for assertion results
        m_pass = RE_ASSERTION_PASS.match(msg)
        if m_pass:
            result.assertions_total += 1
            result.assertions_passed += 1

    for m in RE_TEST_ERROR.finditer(segment):
        msg = m.group("msg").strip()
        result.errors.append(msg)

        m_fail = RE_ASSERTION_FAIL.match(msg)
        if m_fail:
            result.assertions_total += 1
            result.assertions_failed += 1

    return result


# ---------------------------------------------------------------------------
# Integration: resolve test failures to code chunks
# ---------------------------------------------------------------------------

def resolve_test_failures_to_chunks(
    conn: psycopg.Connection,
    suite_result: TestSuiteResult,
) -> int:
    """
    Map test failures to code chunks by matching test names and error context.

    Heuristic: test name "MyGame.Combat.DamageTest" → look for chunks in
    modules containing "Combat" or "Damage".
    """
    resolved = 0

    for test in suite_result.tests:
        if test.result in (TestResult.PASS, TestResult.SKIPPED):
            continue

        # Extract module/class hints from test name
        parts = test.test_name.replace(".", " ").replace("_", " ").split()
        keywords = [p for p in parts if len(p) > 3 and p[0].isupper()]

        if not keywords:
            continue

        # Search for matching chunks
        for keyword in keywords[:3]:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT id FROM code_chunks
                    WHERE (module_name ILIKE %s OR class_name ILIKE %s
                           OR qualified_name ILIKE %s)
                      AND chunk_type IN ('function', 'method', 'class')
                    LIMIT 3
                """, (f"%{keyword}%", f"%{keyword}%", f"%{keyword}%"))
                for row in cur.fetchall():
                    if row[0] not in test.chunk_ids:
                        test.chunk_ids.append(row[0])
                        resolved += 1

    return resolved


# ---------------------------------------------------------------------------
# Integration: link results to patch ledger
# ---------------------------------------------------------------------------

def link_results_to_ledger(
    conn: psycopg.Connection,
    suite_result: TestSuiteResult,
) -> dict[str, list[str]]:
    """
    For each failed test, find which recent patches might be responsible.

    Returns: {test_name: [step_ids]} mapping failures to candidate steps.
    """
    from patch_ledger import PatchLedger

    if not suite_result.step_id:
        return {}

    ledger = PatchLedger(conn)
    failure_sources: dict[str, list[str]] = {}

    for test in suite_result.tests:
        if test.result in (TestResult.PASS, TestResult.SKIPPED):
            continue

        # For each chunk associated with this failure, find recent patches
        candidate_steps = set()
        for chunk_id in test.chunk_ids:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT file_path FROM code_chunks WHERE id = %s",
                    (chunk_id,)
                )
                row = cur.fetchone()
                if row:
                    sources = ledger.find_regression_source(row[0])
                    for src in sources:
                        candidate_steps.add(src.step_id)

        if candidate_steps:
            failure_sources[test.test_name] = sorted(candidate_steps)

    return failure_sources


# ---------------------------------------------------------------------------
# Integration: format results for task context assembler
# ---------------------------------------------------------------------------

def format_for_context(suite_result: TestSuiteResult) -> str:
    """
    Format test results as a context section for the task context assembler.
    Suitable for injection into Builder agent prompts.
    """
    if suite_result.success:
        return f"All {suite_result.total} runtime tests passed."

    lines = [
        f"Runtime tests: {suite_result.passed}/{suite_result.total} passed, "
        f"{suite_result.failed} failed, {suite_result.errors} errors",
        "",
    ]

    for test in suite_result.tests:
        if test.result == TestResult.PASS:
            continue

        lines.append(f"  FAIL: {test.test_name}")
        for err in test.errors[:5]:
            lines.append(f"    - {err}")
        if test.chunk_ids:
            lines.append(f"    Related chunks: {', '.join(test.chunk_ids[:3])}")
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="UE Automation Test harness: generate tests and parse results."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # generate
    gen = sub.add_parser("generate", help="Generate a UE automation test file")
    gen.add_argument("--spec", required=True, help="JSON test spec file")
    gen.add_argument("--output", required=True, help="Output .cpp file path")

    # parse
    parse = sub.add_parser("parse", help="Parse UE automation test output")
    parse.add_argument("--log", required=True, help="Automation log file (or '-' for stdin)")
    parse.add_argument("--step-id", default=None)
    parse.add_argument("--json", action="store_true")
    parse.add_argument("--resolve-chunks", action="store_true")

    # report
    rpt = sub.add_parser("report", help="Full report: parse + resolve + ledger link")
    rpt.add_argument("--log", required=True)
    rpt.add_argument("--step-id", default=None)
    rpt.add_argument("--json", action="store_true")

    args = parser.parse_args()

    if args.command == "generate":
        spec_data = json.loads(Path(args.spec).read_text())
        if isinstance(spec_data, list):
            specs = [spec_from_dict(s) for s in spec_data]
        else:
            specs = [spec_from_dict(spec_data)]

        output_parts = [generate_test_cpp(s) for s in specs]
        Path(args.output).write_text("\n\n".join(output_parts))
        print(f"Generated {len(specs)} test(s) → {args.output}")

    elif args.command == "parse":
        if args.log == "-":
            log_text = sys.stdin.read()
        else:
            log_text = Path(args.log).read_text(encoding="utf-8", errors="replace")

        suite = parse_test_log(log_text, step_id=args.step_id)

        if args.resolve_chunks:
            conn = psycopg.connect(DB_DSN)
            try:
                resolve_test_failures_to_chunks(conn, suite)
            finally:
                conn.close()

        if args.json:
            print(json.dumps(suite.to_dict(), indent=2))
        else:
            print(suite.summary())

    elif args.command == "report":
        log_text = Path(args.log).read_text(encoding="utf-8", errors="replace")
        suite = parse_test_log(log_text, step_id=args.step_id)

        conn = psycopg.connect(DB_DSN)
        try:
            resolved = resolve_test_failures_to_chunks(conn, suite)
            sources = link_results_to_ledger(conn, suite)
        finally:
            conn.close()

        if args.json:
            result = suite.to_dict()
            result["regression_sources"] = sources
            result["chunks_resolved"] = resolved
            print(json.dumps(result, indent=2))
        else:
            print(suite.summary())
            if sources:
                print("\nRegression candidates:")
                for test_name, steps in sources.items():
                    print(f"  {test_name} → steps: {', '.join(steps)}")


if __name__ == "__main__":
    main()
