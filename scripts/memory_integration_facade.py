"""
memory_integration_facade.py — Clean integration API for the Campaign Orchestrator.

L9-S1 of the Level 9 "Orchestrator Fusion" implementation.

Exposes static, import-safe hooks that orchestrator adapters call without
importing deeply nested memory engine internals. Each hook returns plain
dicts/strings suitable for direct injection into agent packets.

Integration points:
    1. context_for_builder()   — Task-scoped context for Builder agent prompts
    2. classify_build_output() — Structured error classification from build logs
    3. record_file_change()    — Patch ledger recording for file modifications
    4. rollback_step()         — Atomic rollback of a campaign step
    5. parse_test_results()    — Runtime test result parsing
    6. index_new_files()       — Parse + embed + graph new/changed files
    7. find_regression()       — Identify which step broke a file

All hooks are stateless and create/close their own DB connections.
Thread-safe. Crash-resilient (each hook catches and logs exceptions,
returning safe defaults on failure).

Usage:
    from memory_integration_facade import context_for_builder, classify_build_output
    
    # In packet_builder.py, for the Builder role:
    builder_context = context_for_builder(
        task="implement dash ability",
        target_qualified_name="MyCharacter.AMyCharacter.Dash",
    )
    packet["memory_context_block"] = builder_context["prompt_block"]
    
    # In unreal_delivery_loop.py, after build:
    errors = classify_build_output(build_log_text)
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger("openclaw.memory_integration_facade")

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

DB_DSN = os.environ.get("OPENCLAW_MEMORY_DSN", "dbname=openclaw_memory")


def _get_conn():
    """Create a fresh database connection."""
    import psycopg
    return psycopg.connect(DB_DSN)


# ---------------------------------------------------------------------------
# 0. Builder write lifecycle hooks (L9-S2)
# ---------------------------------------------------------------------------

def before_builder_write(
    step_id: str,
    file_path: str,
    *,
    campaign_id: str | None = None,
    metadata: dict | None = None,
) -> dict[str, Any]:
    """Snapshot current file + record in patch ledger BEFORE Builder writes."""
    try:
        from builder_write_hook import before_write
        return before_write(step_id, file_path, campaign_id=campaign_id, metadata=metadata)
    except Exception as e:
        LOGGER.error("before_builder_write failed: %s", e)
        return {"patch_id": None, "success": False, "error": str(e)}


def after_builder_write(
    step_id: str,
    file_path: str,
    *,
    project_id: str | None = None,
) -> dict[str, Any]:
    """Parse + embed + graph + ledger update AFTER Builder writes."""
    try:
        from builder_write_hook import after_write
        return after_write(step_id, file_path, project_id=project_id)
    except Exception as e:
        LOGGER.error("after_builder_write failed: %s", e)
        return {"success": False, "error": str(e)}


def handle_builder_write(
    step_id: str,
    file_path: str,
    new_content: str,
    *,
    campaign_id: str | None = None,
    project_id: str | None = None,
    metadata: dict | None = None,
) -> dict[str, Any]:
    """Full atomic lifecycle: snapshot → write → index → ledger update."""
    try:
        from builder_write_hook import handle_builder_write as _handle
        return _handle(
            step_id, file_path, new_content,
            campaign_id=campaign_id, project_id=project_id, metadata=metadata,
        )
    except Exception as e:
        LOGGER.error("handle_builder_write failed: %s", e)
        return {"success": False, "error": str(e)}


# ---------------------------------------------------------------------------
# 1. Context for Builder agent
# ---------------------------------------------------------------------------

def context_for_builder(
    task: str,
    *,
    target_qualified_name: str | None = None,
    target_file: str | None = None,
    target_line: int | None = None,
    error_groups: list[dict] | None = None,
    total_budget_chars: int = 24000,
) -> dict[str, Any]:
    """
    Assemble a task-scoped context pack for the Builder agent.

    Returns a dict with:
        - prompt_block: str — Ready for injection into Builder packet
        - total_chars: int
        - total_tokens_approx: int
        - sections: list of {label, priority, char_count, source}
        - assembly_ms: float

    On failure, returns a minimal dict with empty prompt_block.
    """
    try:
        from task_context_assembler import TaskContextAssembler, TokenBudget

        budget = TokenBudget(total_max_chars=total_budget_chars)
        conn = _get_conn()
        try:
            assembler = TaskContextAssembler(conn, budget=budget)
            ctx = assembler.assemble(
                task,
                target_qualified_name=target_qualified_name,
                target_file=target_file,
                target_line=target_line,
                error_groups=error_groups,
            )
            return ctx.to_dict()
        finally:
            conn.close()

    except Exception as e:
        LOGGER.error("context_for_builder failed: %s", e)
        return {
            "prompt_block": "",
            "total_chars": 0,
            "total_tokens_approx": 0,
            "sections": [],
            "assembly_ms": 0.0,
            "error": str(e),
        }


# ---------------------------------------------------------------------------
# 1b. Compile feedback for Reviewer agent (L9-S3)
# ---------------------------------------------------------------------------

def process_build_for_reviewer(
    build_log: str,
    *,
    step_id: str | None = None,
    resolve_chunks: bool = True,
) -> dict[str, Any]:
    """
    Process a Build.sh log into a Reviewer-ready feedback packet.

    Returns a dict with reviewer_block (token-budgeted text), structured
    error data, regression sources, and success flag. Gracefully handles
    unparseable/empty/binary logs.
    """
    try:
        from compile_feedback_loop import process_build_log
        return process_build_log(build_log, step_id=step_id, resolve_chunks=resolve_chunks)
    except Exception as e:
        LOGGER.error("process_build_for_reviewer failed: %s", e)
        snippet = build_log[:2000] if build_log else "(empty)"
        return {
            "success": False,
            "reviewer_block": f"Build feedback processing failed ({e}). Raw snippet:\n{snippet}",
            "structured": {},
            "error_count": 0,
            "warning_count": 0,
            "error": str(e),
        }


# ---------------------------------------------------------------------------
# 2. Compile error classification (raw, for programmatic use)
# ---------------------------------------------------------------------------

def classify_build_output(
    build_log: str,
    *,
    resolve_chunks: bool = True,
) -> dict[str, Any]:
    """
    Classify a Build.sh log into structured error groups.

    Returns a dict with:
        - summary: {total_diagnostics, errors, warnings, cascades, ...}
        - groups: list of error groups with chunk IDs, categories, fix strategies

    On failure, returns a minimal dict with zero counts.
    """
    try:
        from classify_compile_errors import classify_build_log as _classify
        return _classify(build_log, resolve_chunks=resolve_chunks)

    except Exception as e:
        LOGGER.error("classify_build_output failed: %s", e)
        return {
            "summary": {
                "total_diagnostics": 0,
                "error_groups": 0,
                "errors": 0,
                "warnings": 0,
                "cascades": 0,
                "unique_root_causes": 0,
                "chunk_ids_resolved": 0,
                "chunks_affected": [],
                "category_breakdown": {},
            },
            "groups": [],
            "error": str(e),
        }


# ---------------------------------------------------------------------------
# 3. Patch ledger: record file change
# ---------------------------------------------------------------------------

def record_file_change(
    step_id: str,
    file_path: str,
    *,
    content_before: str | None = None,
    content_after: str | None = None,
    campaign_id: str | None = None,
    operation: str | None = None,
    metadata: dict | None = None,
) -> dict[str, Any]:
    """
    Record a file modification in the patch ledger.

    Returns:
        - patch_id: str
        - success: bool

    Auto-reads content_before from disk if not provided.
    Auto-detects operation from before/after state.
    """
    try:
        from patch_ledger import PatchLedger
        conn = _get_conn()
        try:
            ledger = PatchLedger(conn)
            patch_id = ledger.record_patch(
                step_id=step_id,
                file_path=file_path,
                content_before=content_before,
                content_after=content_after,
                campaign_id=campaign_id,
                operation=operation,
                metadata=metadata,
            )
            return {"patch_id": patch_id, "success": True}
        finally:
            conn.close()

    except Exception as e:
        LOGGER.error("record_file_change failed: %s", e)
        return {"patch_id": None, "success": False, "error": str(e)}


def update_after_parse(step_id: str, file_path: str) -> bool:
    """Update chunk_ids_after for a patch after the file has been re-parsed."""
    try:
        from patch_ledger import PatchLedger
        conn = _get_conn()
        try:
            PatchLedger(conn).update_after_parse(step_id, file_path)
            return True
        finally:
            conn.close()
    except Exception as e:
        LOGGER.error("update_after_parse failed: %s", e)
        return False


# ---------------------------------------------------------------------------
# 4. Patch ledger: rollback step
# ---------------------------------------------------------------------------

def rollback_step(
    step_id: str,
    *,
    reparse: bool = True,
    reembed: bool = True,
    regraph: bool = True,
) -> dict[str, Any]:
    """
    Atomically rollback all patches from a given step.

    Returns:
        - patches_rolled_back: int
        - files_restored: list[str]
        - chunks_reparsed: int
        - graph_updated: bool
        - errors: list[str]
    """
    try:
        from patch_ledger import PatchLedger
        conn = _get_conn()
        try:
            ledger = PatchLedger(conn)
            result = ledger.rollback_step(
                step_id,
                reparse=reparse,
                reembed=reembed,
                regraph=regraph,
            )
            return {
                "patches_rolled_back": result.patches_rolled_back,
                "files_restored": result.files_restored,
                "chunks_reparsed": result.chunks_reparsed,
                "chunks_reembedded": result.chunks_reembedded,
                "graph_updated": result.graph_updated,
                "errors": result.errors,
            }
        finally:
            conn.close()

    except Exception as e:
        LOGGER.error("rollback_step failed: %s", e)
        return {
            "patches_rolled_back": 0,
            "files_restored": [],
            "chunks_reparsed": 0,
            "chunks_reembedded": 0,
            "graph_updated": False,
            "errors": [str(e)],
        }


# ---------------------------------------------------------------------------
# 4b. Automation bridge for Tester agent (L9-S4)
# ---------------------------------------------------------------------------

def process_automation_for_tester(
    uat_log: str,
    *,
    step_id: str | None = None,
    resolve_chunks: bool = True,
    link_to_ledger: bool = True,
) -> dict[str, Any]:
    """
    Process a UE Automation Tool log into a Tester-ready feedback packet.

    Returns a dict with tester_block (token-budgeted text for the Tester
    agent prompt), structured results, regression sources, and success flag.
    Gracefully handles empty/binary/crash logs.
    """
    try:
        from automation_bridge import process_automation_log
        return process_automation_log(
            uat_log, step_id=step_id,
            resolve_chunks=resolve_chunks, link_to_ledger=link_to_ledger,
        )
    except Exception as e:
        LOGGER.error("process_automation_for_tester failed: %s", e)
        snippet = uat_log[:2000] if uat_log else "(empty)"
        return {
            "success": False,
            "tester_block": f"Automation feedback processing failed ({e}). Raw snippet:\n{snippet}",
            "structured": {},
            "total": 0, "passed": 0, "failed": 0, "errors": 0,
            "error": str(e),
        }


# ---------------------------------------------------------------------------
# 5. Runtime test result parsing (raw, for programmatic use)
# ---------------------------------------------------------------------------

def parse_test_results(
    automation_log: str,
    *,
    step_id: str | None = None,
    resolve_chunks: bool = True,
    link_to_ledger: bool = True,
) -> dict[str, Any]:
    """
    Parse UE automation test output and optionally link to patch ledger.

    Returns:
        - total, passed, failed, errors, success
        - tests: list of individual test results
        - regression_sources: {test_name: [step_ids]} (if link_to_ledger)
        - context_block: str — Ready for injection into Builder context
    """
    try:
        from runtime_test_harness import (
            parse_test_log,
            resolve_test_failures_to_chunks,
            link_results_to_ledger,
            format_for_context,
        )

        suite = parse_test_log(automation_log, step_id=step_id)
        regression_sources = {}

        if resolve_chunks or link_to_ledger:
            conn = _get_conn()
            try:
                if resolve_chunks:
                    resolve_test_failures_to_chunks(conn, suite)
                if link_to_ledger and step_id:
                    regression_sources = link_results_to_ledger(conn, suite)
            finally:
                conn.close()

        result = suite.to_dict()
        result["regression_sources"] = regression_sources
        result["context_block"] = format_for_context(suite)
        return result

    except Exception as e:
        LOGGER.error("parse_test_results failed: %s", e)
        return {
            "total": 0, "passed": 0, "failed": 0, "errors": 0,
            "success": True, "tests": [],
            "regression_sources": {},
            "context_block": "",
            "error": str(e),
        }


# ---------------------------------------------------------------------------
# 6. Index new/changed files
# ---------------------------------------------------------------------------

def index_files(
    file_paths: list[str],
    *,
    project_id: str | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """
    Parse, embed, and graph-sync a list of files.

    Handles both Python and C++ files automatically.
    Returns parse/embed/graph stats.
    """
    try:
        import psycopg
        from parse_code_ast import parse_file as parse_py, upsert_chunks, remove_stale_chunks
        from parse_code_cpp import CppParser, CPP_EXTENSIONS

        cpp_parser = CppParser()
        conn = psycopg.connect(DB_DSN)
        stats = {"parsed": 0, "upserted": 0, "removed": 0, "errors": 0, "files": 0}

        try:
            for fp in file_paths:
                try:
                    content = Path(fp).read_text(encoding="utf-8", errors="replace")
                    suffix = Path(fp).suffix.lower()

                    if suffix == ".py":
                        chunks = parse_py(fp, content)
                    elif suffix in CPP_EXTENSIONS:
                        chunks = cpp_parser.parse_file(fp, content)
                    else:
                        continue

                    if not chunks:
                        continue

                    for c in chunks:
                        c["project_id"] = project_id
                        c["subproject_id"] = None

                    stats["upserted"] += upsert_chunks(conn, chunks)
                    stats["removed"] += remove_stale_chunks(conn, fp, {c["id"] for c in chunks})
                    stats["parsed"] += len(chunks)
                    stats["files"] += 1

                except Exception as e:
                    LOGGER.error("index_files error for %s: %s", fp, e)
                    stats["errors"] += 1

            # Embed new chunks
            try:
                from embed_code_chunks import embed_unembedded_chunks
                stats["embedded"] = embed_unembedded_chunks(conn)
            except Exception as e:
                LOGGER.error("embed failed: %s", e)
                stats["embedded"] = 0

        finally:
            conn.close()

        # Graph sync
        try:
            from sync_code_graph import sync_graph
            sync_graph(force=force)
            stats["graph_synced"] = True
        except Exception as e:
            LOGGER.error("graph sync failed: %s", e)
            stats["graph_synced"] = False

        return stats

    except Exception as e:
        LOGGER.error("index_files failed: %s", e)
        return {"error": str(e)}


# ---------------------------------------------------------------------------
# 7. Find regression source
# ---------------------------------------------------------------------------

def find_regression(
    file_path: str,
    *,
    error_line: int | None = None,
) -> list[dict[str, Any]]:
    """
    Find which campaign step most recently modified a file (regression candidate).

    Returns list of {step_id, operation, created_at, campaign_id}.
    """
    try:
        from patch_ledger import PatchLedger
        conn = _get_conn()
        try:
            ledger = PatchLedger(conn)
            patches = ledger.find_regression_source(file_path, error_line=error_line)
            return [
                {
                    "step_id": p.step_id,
                    "operation": p.operation,
                    "created_at": p.created_at.isoformat() if p.created_at else None,
                    "campaign_id": p.campaign_id,
                    "patch_id": p.patch_id,
                }
                for p in patches
            ]
        finally:
            conn.close()

    except Exception as e:
        LOGGER.error("find_regression failed: %s", e)
        return []
