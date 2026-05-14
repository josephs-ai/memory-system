"""
compile_feedback_loop.py — Compile classifier integration for the Reviewer agent.

L9-S3 of the Level 9 "Orchestrator Fusion" implementation.

Intercepts raw Build.sh logs from the macOS execution bridge, pipes them
through the error classifier, and produces a token-budgeted evaluation
packet for the Reviewer agent.

Handles edge cases:
    - Completely unparseable logs → fallback to truncated raw snippet
    - Empty logs → clean "no output" response
    - Binary/corrupted logs → safe degradation
    - Logs with zero errors → clean success response

Token budgeting:
    - Root-cause errors: full detail (category, chunk, fix strategy, notes)
    - Cascade errors: compressed (category + "cascade of [root]" only)
    - Warnings: count only, no individual detail
    - Raw log snippet: only included on parse failure, capped at 2000 chars

Usage:
    from compile_feedback_loop import process_build_log

    # In unreal_delivery_loop.py, after receiving build output:
    feedback = process_build_log(
        build_log=stdout_text,
        step_id="work-123",
    )
    reviewer_packet["compile_feedback"] = feedback["reviewer_block"]
    reviewer_packet["compile_errors_json"] = feedback["structured"]
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger("openclaw.compile_feedback_loop")
logging.basicConfig(
    level=os.environ.get("OPENCLAW_LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

# Token budget constants
ROOT_ERROR_MAX_CHARS = 400       # Per root-cause error
CASCADE_ERROR_MAX_CHARS = 80     # Per cascade error
TOTAL_ERRORS_MAX_CHARS = 6000    # Total error section
RAW_FALLBACK_MAX_CHARS = 2000    # Raw log on parse failure
SUMMARY_MAX_CHARS = 500          # Summary section


def process_build_log(
    build_log: str,
    *,
    step_id: str | None = None,
    resolve_chunks: bool = True,
    max_errors_shown: int = 15,
    max_cascades_shown: int = 5,
) -> dict[str, Any]:
    """
    Process a Build.sh log into a Reviewer-ready feedback packet.

    Returns:
        {
            "success": bool,          — True if build had zero errors
            "reviewer_block": str,    — Token-budgeted text for Reviewer prompt
            "structured": dict,       — Full classified output (for programmatic use)
            "error_count": int,
            "warning_count": int,
            "root_causes": int,
            "cascades": int,
            "chunks_affected": list,
            "regression_sources": list,
            "processing_ms": float,
        }
    """
    t0 = time.monotonic()

    result: dict[str, Any] = {
        "success": True,
        "reviewer_block": "",
        "structured": {},
        "error_count": 0,
        "warning_count": 0,
        "root_causes": 0,
        "cascades": 0,
        "chunks_affected": [],
        "regression_sources": [],
        "processing_ms": 0.0,
    }

    # --- Edge case: empty log ---
    if not build_log or not build_log.strip():
        result["reviewer_block"] = "Build produced no output."
        result["processing_ms"] = round((time.monotonic() - t0) * 1000, 1)
        return result

    # --- Edge case: binary/corrupted log ---
    if not _is_text_log(build_log):
        snippet = _safe_truncate(build_log, RAW_FALLBACK_MAX_CHARS)
        result["success"] = False
        result["reviewer_block"] = (
            "Build log appears corrupted or binary. Raw snippet:\n"
            f"```\n{snippet}\n```"
        )
        result["processing_ms"] = round((time.monotonic() - t0) * 1000, 1)
        return result

    # --- Classify ---
    try:
        from classify_compile_errors import classify_build_log
        classified = classify_build_log(build_log, resolve_chunks=resolve_chunks)
    except Exception as e:
        LOGGER.error("Error classifier failed: %s", e)
        snippet = _safe_truncate(build_log, RAW_FALLBACK_MAX_CHARS)
        result["success"] = False
        result["reviewer_block"] = (
            f"Error classifier failed ({e}). Raw build log snippet:\n"
            f"```\n{snippet}\n```"
        )
        result["processing_ms"] = round((time.monotonic() - t0) * 1000, 1)
        return result

    result["structured"] = classified
    summary = classified.get("summary", {})
    groups = classified.get("groups", [])

    result["error_count"] = summary.get("errors", 0)
    result["warning_count"] = summary.get("warnings", 0)
    result["root_causes"] = summary.get("unique_root_causes", 0)
    result["cascades"] = summary.get("cascades", 0)
    result["chunks_affected"] = summary.get("chunks_affected", [])

    # --- Edge case: classifier found zero diagnostics ---
    if summary.get("total_diagnostics", 0) == 0:
        # Check if the log mentions success
        log_lower = build_log.lower()
        if "error" in log_lower or "fail" in log_lower:
            snippet = _safe_truncate(build_log, RAW_FALLBACK_MAX_CHARS)
            result["success"] = False
            result["reviewer_block"] = (
                "Build log mentions errors but classifier found no parseable diagnostics. "
                "Raw snippet:\n"
                f"```\n{snippet}\n```"
            )
        else:
            result["reviewer_block"] = "Build completed successfully. No errors or warnings."
        result["processing_ms"] = round((time.monotonic() - t0) * 1000, 1)
        return result

    # --- Build success/failure ---
    result["success"] = result["error_count"] == 0

    # --- Find regression sources via patch ledger ---
    if step_id and result["chunks_affected"]:
        result["regression_sources"] = _find_regressions(result["chunks_affected"])

    # --- Build token-budgeted reviewer block ---
    result["reviewer_block"] = _build_reviewer_block(
        summary, groups,
        regression_sources=result["regression_sources"],
        max_errors=max_errors_shown,
        max_cascades=max_cascades_shown,
    )

    result["processing_ms"] = round((time.monotonic() - t0) * 1000, 1)
    return result


# ---------------------------------------------------------------------------
# Token-budgeted reviewer block
# ---------------------------------------------------------------------------

def _build_reviewer_block(
    summary: dict,
    groups: list[dict],
    *,
    regression_sources: list[dict] | None = None,
    max_errors: int = 15,
    max_cascades: int = 5,
) -> str:
    """Build the text block injected into the Reviewer agent's prompt."""
    parts: list[str] = []
    chars_used = 0

    # --- Summary line ---
    error_count = summary.get("errors", 0)
    warning_count = summary.get("warnings", 0)
    root_causes = summary.get("unique_root_causes", 0)
    cascades = summary.get("cascades", 0)

    if error_count == 0:
        status = "BUILD SUCCESS"
    else:
        status = "BUILD FAILED"

    summary_line = (
        f"{status}: {error_count} errors ({root_causes} root causes, "
        f"{cascades} cascades), {warning_count} warnings"
    )
    parts.append(summary_line)

    # Category breakdown
    cat_counts = summary.get("category_breakdown", {})
    if cat_counts:
        cats = ", ".join(f"{cat}: {n}" for cat, n in sorted(cat_counts.items(), key=lambda x: -x[1]))
        parts.append(f"Categories: {cats}")

    parts.append("")
    chars_used += sum(len(p) for p in parts)

    # --- Root-cause errors (full detail) ---
    root_errors = [g for g in groups if not g.get("is_cascade") and g.get("root_error", {}).get("severity") in ("error", "fatal error")]
    cascade_errors = [g for g in groups if g.get("is_cascade")]

    shown_roots = 0
    for group in root_errors[:max_errors]:
        if chars_used >= TOTAL_ERRORS_MAX_CHARS:
            parts.append(f"  ... ({len(root_errors) - shown_roots} more root errors)")
            break

        block = _format_error_group(group, compact=False)
        if chars_used + len(block) > TOTAL_ERRORS_MAX_CHARS:
            block = _format_error_group(group, compact=True)

        parts.append(block)
        chars_used += len(block)
        shown_roots += 1

    # --- Cascade errors (compact) ---
    if cascade_errors:
        parts.append("")
        cascade_header = f"Cascades ({len(cascade_errors)} — likely caused by root errors above):"
        parts.append(cascade_header)
        chars_used += len(cascade_header)

        shown_cascades = 0
        for group in cascade_errors[:max_cascades]:
            if chars_used >= TOTAL_ERRORS_MAX_CHARS:
                break
            block = _format_error_group(group, compact=True)
            parts.append(block)
            chars_used += len(block)
            shown_cascades += 1

        remaining = len(cascade_errors) - shown_cascades
        if remaining > 0:
            parts.append(f"  ... ({remaining} more cascades)")

    # --- Regression sources ---
    if regression_sources:
        parts.append("")
        parts.append("Regression candidates:")
        for src in regression_sources[:5]:
            parts.append(f"  Step {src.get('step_id', '?')}: {src.get('operation', '?')} ({src.get('created_at', '?')})")

    return "\n".join(parts)


def _format_error_group(group: dict, *, compact: bool = False) -> str:
    """Format a single error group for the reviewer block."""
    root = group.get("root_error", {})
    category = group.get("category", "unknown")
    chunk_name = root.get("chunk_name")
    file_loc = f"{root.get('file', '?')}:{root.get('line', '?')}"

    if compact:
        chunk_info = f" → {chunk_name}" if chunk_name else ""
        return f"  [{category}] {file_loc}{chunk_info}: {root.get('message', '?')[:120]}"

    lines = [f"  [{category}] {file_loc}"]
    lines.append(f"    {root.get('message', '?')}")

    if chunk_name:
        lines.append(f"    Chunk: {chunk_name} ({root.get('chunk_type', '?')})")

    fix = group.get("fix_strategy")
    if fix:
        lines.append(f"    Fix: {fix}")

    notes = root.get("notes", [])
    for note in notes[:2]:
        lines.append(f"    Note: {note[:150]}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_text_log(text: str) -> bool:
    """Heuristic: check if the log is likely text (not binary)."""
    if not text:
        return True
    # Check first 1000 chars for null bytes or high non-printable ratio
    sample = text[:1000]
    if "\x00" in sample:
        return False
    non_printable = sum(1 for c in sample if ord(c) < 32 and c not in "\n\r\t")
    return non_printable / max(len(sample), 1) < 0.1


def _safe_truncate(text: str, max_chars: int) -> str:
    """Truncate text safely, replacing non-printable chars."""
    cleaned = text.replace("\x00", "")
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[:max_chars] + "\n... [truncated]"


def _find_regressions(chunk_ids: list[str]) -> list[dict]:
    """Query patch ledger for regression sources."""
    try:
        import psycopg
        DB_DSN = os.environ.get("OPENCLAW_MEMORY_DSN", "dbname=openclaw_memory")
        conn = psycopg.connect(DB_DSN)
        try:
            with conn.cursor() as cur:
                if not chunk_ids:
                    return []
                placeholders = ",".join(["%s"] * len(chunk_ids))
                cur.execute(f"""
                    SELECT DISTINCT pl.step_id, pl.operation, pl.created_at, pl.campaign_id
                    FROM patch_ledger pl
                    JOIN code_chunks cc ON cc.file_path = pl.file_path
                    WHERE cc.id IN ({placeholders})
                      AND pl.rolled_back = FALSE
                    ORDER BY pl.created_at DESC
                    LIMIT 5
                """, chunk_ids)
                return [
                    {
                        "step_id": row[0],
                        "operation": row[1],
                        "created_at": row[2].isoformat() if row[2] else None,
                        "campaign_id": row[3],
                    }
                    for row in cur.fetchall()
                ]
        finally:
            conn.close()
    except Exception as e:
        LOGGER.error("Regression lookup failed: %s", e)
        return []
