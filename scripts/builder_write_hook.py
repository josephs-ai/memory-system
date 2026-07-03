"""
builder_write_hook.py — Atomic file-write lifecycle for the Builder agent.

L9-S2 of the Level 9 "Orchestrator Fusion" implementation.

Called by universal_wrapper.py (or equivalent) when the Builder writes files.
Wraps the entire write lifecycle in an atomic sequence:

    1. BEFORE write: snapshot current content + record in patch ledger
    2. WRITE: the caller applies the file modification
    3. AFTER write: incremental re-parse → re-embed → graph sync
    4. UPDATE ledger: store post-parse chunk IDs

If step 3 fails (parse error, embed error), the file modification still
stands but the error is logged and returned — the caller can decide to
rollback via the facade.

Leverages the same incremental hash-skip from code_index_hook.py (S8):
    - File hash comparison prevents redundant re-parsing
    - SyntaxError resilience for partial writes
    - LEFT JOIN filter in re-embedding skips already-embedded chunks

Thread-safe: each call creates its own DB connection.
Idempotent: re-calling for the same (step_id, file_path) updates the record.

Usage by universal_wrapper.py:
    from builder_write_hook import before_write, after_write

    # Before applying the Builder's patch:
    pre = before_write(step_id="work-123", file_path="/path/to/file.cpp",
                       campaign_id="campaign-1")

    # Apply the Builder's file modification here...
    Path(file_path).write_text(new_content)

    # After the write:
    post = after_write(step_id="work-123", file_path="/path/to/file.cpp",
                       project_id="mygame")
"""

from __future__ import annotations

import hashlib
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

import psycopg

LOGGER = logging.getLogger("openclaw.builder_write_hook")
logging.basicConfig(
    level=os.environ.get("OPENCLAW_LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

DB_DSN = (os.environ.get("OPENCLAW_MEMORY_DSN") or os.environ.get("OPENCLAW_MEMORY_DB_DSN") or "dbname=openclaw_memory")  # honor both canonical DSN env names

# Extensions we can parse
PYTHON_EXTENSIONS = {".py"}
CPP_EXTENSIONS = {".cpp", ".h", ".hpp", ".cc", ".cxx", ".hh", ".hxx"}
PARSEABLE_EXTENSIONS = PYTHON_EXTENSIONS | CPP_EXTENSIONS


def _get_conn():
    return psycopg.connect(DB_DSN)


def _file_hash(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Phase 1: BEFORE WRITE — snapshot + ledger record
# ---------------------------------------------------------------------------

def before_write(
    step_id: str,
    file_path: str,
    *,
    campaign_id: str | None = None,
    metadata: dict | None = None,
) -> dict[str, Any]:
    """
    Call BEFORE applying a file modification.

    Snapshots current file content and records the pre-write state
    in the patch ledger. Returns pre-write info for the caller.

    Returns:
        {
            "patch_id": str | None,
            "file_existed": bool,
            "file_hash_before": str | None,
            "content_before_chars": int,
            "success": bool,
            "error": str | None,
        }
    """
    file_path = str(Path(file_path).resolve())
    result: dict[str, Any] = {
        "patch_id": None,
        "file_existed": False,
        "file_hash_before": None,
        "content_before_chars": 0,
        "success": False,
        "error": None,
    }

    try:
        # Read current content
        content_before = None
        if Path(file_path).exists():
            content_before = Path(file_path).read_text(encoding="utf-8", errors="replace")
            result["file_existed"] = True
            result["file_hash_before"] = _file_hash(content_before)
            result["content_before_chars"] = len(content_before)

        # Record in patch ledger (content_after=None at this point)
        from patch_ledger import PatchLedger
        conn = _get_conn()
        try:
            ledger = PatchLedger(conn)
            patch_id = ledger.record_patch(
                step_id=step_id,
                file_path=file_path,
                content_before=content_before,
                content_after=None,  # Will be updated in after_write
                campaign_id=campaign_id,
                metadata=metadata,
            )
            result["patch_id"] = patch_id
            result["success"] = True
        finally:
            conn.close()

    except Exception as e:
        LOGGER.error("before_write failed for %s: %s", file_path, e)
        result["error"] = str(e)

    return result


# ---------------------------------------------------------------------------
# Phase 2: AFTER WRITE — parse + embed + graph + ledger update
# ---------------------------------------------------------------------------

def after_write(
    step_id: str,
    file_path: str,
    *,
    project_id: str | None = None,
    force_reparse: bool = False,
) -> dict[str, Any]:
    """
    Call AFTER applying a file modification.

    Reads the new file content, updates the patch ledger with content_after,
    then runs the incremental indexing pipeline:
        1. Parse (Python AST or C++ tree-sitter)
        2. Embed new/changed chunks
        3. Sync Neo4j graph
        4. Update ledger with post-parse chunk IDs

    Returns:
        {
            "success": bool,
            "file_hash_after": str | None,
            "chunks_parsed": int,
            "chunks_upserted": int,
            "chunks_removed": int,
            "chunks_embedded": int,
            "graph_synced": bool,
            "parse_ms": float,
            "embed_ms": float,
            "total_ms": float,
            "error": str | None,
            "skipped": bool,  # True if file hash unchanged
        }
    """
    file_path = str(Path(file_path).resolve())
    t0 = time.monotonic()

    result: dict[str, Any] = {
        "success": False,
        "file_hash_after": None,
        "chunks_parsed": 0,
        "chunks_upserted": 0,
        "chunks_removed": 0,
        "chunks_embedded": 0,
        "graph_synced": False,
        "parse_ms": 0.0,
        "embed_ms": 0.0,
        "total_ms": 0.0,
        "error": None,
        "skipped": False,
    }

    suffix = Path(file_path).suffix.lower()
    if suffix not in PARSEABLE_EXTENSIONS:
        LOGGER.debug("Skipping non-parseable file: %s", file_path)
        result["success"] = True
        result["skipped"] = True
        return result

    try:
        # Read new content
        if not Path(file_path).exists():
            # File was deleted — clean up chunks
            result["success"] = True
            result["skipped"] = True
            return result

        content_after = Path(file_path).read_text(encoding="utf-8", errors="replace")
        result["file_hash_after"] = _file_hash(content_after)

        conn = _get_conn()
        try:
            # --- Update patch ledger with content_after ---
            from patch_ledger import PatchLedger
            ledger = PatchLedger(conn)

            # Update the existing record with new content
            patch_id = PatchLedger.make_patch_id(step_id, file_path)
            with conn.cursor() as cur:
                cur.execute("""
                    UPDATE patch_ledger
                    SET file_hash_after = %s,
                        content_after = %s
                    WHERE patch_id = %s
                """, (result["file_hash_after"], content_after, patch_id))
            conn.commit()

            # --- Phase 3a: Parse (incremental, hash-skip) ---
            t_parse = time.monotonic()

            if suffix in PYTHON_EXTENSIONS:
                from parse_code_ast import process_file
                stats = process_file(conn, file_path, project_id=project_id, force=force_reparse)
            elif suffix in CPP_EXTENSIONS:
                from parse_code_cpp import CppParser, process_cpp_file
                cpp_parser = CppParser()
                stats = process_cpp_file(conn, cpp_parser, file_path, project_id=project_id, force=force_reparse)
            else:
                stats = {"parsed": 0, "upserted": 0, "removed": 0, "skipped": 1}

            result["chunks_parsed"] = stats.get("parsed", 0)
            result["chunks_upserted"] = stats.get("upserted", 0)
            result["chunks_removed"] = stats.get("removed", 0)
            result["skipped"] = bool(stats.get("skipped", 0))
            result["parse_ms"] = round((time.monotonic() - t_parse) * 1000, 1)

            # --- Phase 3b: Re-embed changed chunks ---
            t_embed = time.monotonic()
            if not result["skipped"]:
                try:
                    from code_index_hook import reembed_changed_chunks
                    result["chunks_embedded"] = reembed_changed_chunks()
                except Exception as e:
                    LOGGER.error("Re-embed failed for %s: %s", file_path, e)

            result["embed_ms"] = round((time.monotonic() - t_embed) * 1000, 1)

            # --- Phase 3c: Graph sync ---
            if not result["skipped"]:
                try:
                    from sync_code_graph import sync_graph
                    sync_graph(force=False)
                    result["graph_synced"] = True
                except Exception as e:
                    LOGGER.error("Graph sync failed: %s", e)

            # --- Phase 4: Update ledger with post-parse chunk IDs ---
            ledger.update_after_parse(step_id, file_path)

            result["success"] = True

        finally:
            conn.close()

    except Exception as e:
        LOGGER.error("after_write failed for %s: %s", file_path, e)
        result["error"] = str(e)

    result["total_ms"] = round((time.monotonic() - t0) * 1000, 1)
    return result


# ---------------------------------------------------------------------------
# Combined: full write lifecycle (for simpler callers)
# ---------------------------------------------------------------------------

def handle_builder_write(
    step_id: str,
    file_path: str,
    new_content: str,
    *,
    campaign_id: str | None = None,
    project_id: str | None = None,
    metadata: dict | None = None,
) -> dict[str, Any]:
    """
    Full atomic lifecycle: snapshot → write → index → update.

    This is the simplest integration point. Call this with the new content
    and it handles everything.

    Returns combined before_write + after_write results.
    """
    result: dict[str, Any] = {"success": False}

    # Phase 1: Before
    pre = before_write(
        step_id, file_path,
        campaign_id=campaign_id,
        metadata=metadata,
    )
    result["before"] = pre

    if not pre["success"]:
        result["error"] = f"before_write failed: {pre.get('error')}"
        return result

    # Phase 2: Write the file
    try:
        fp = Path(file_path).resolve()
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(new_content, encoding="utf-8")
    except Exception as e:
        result["error"] = f"file write failed: {e}"
        return result

    # Phase 3 + 4: After
    post = after_write(
        step_id, str(fp),
        project_id=project_id,
    )
    result["after"] = post
    result["success"] = post["success"]
    if post.get("error"):
        result["error"] = post["error"]

    return result
