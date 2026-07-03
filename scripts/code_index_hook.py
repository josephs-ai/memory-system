"""
code_index_hook.py — File watcher hook for incremental code index updates.

Slice 8 (S8) of the Zero-Reread Architecture.

Called after the tracked_path_watcher flushes changed file paths.
Filters for .py files, re-parses them via parse_code_ast, re-embeds
changed chunks, and updates the Neo4j graph.

Designed to be called from the watcher's flush pipeline or as a standalone
script for manual re-indexing of specific files.

Debouncing strategy:
    - The tracked_path_watcher already debounces filesystem events
      (configurable debounce_seconds + max_wait_seconds)
    - This hook adds a second layer: file hash comparison in parse_code_ast
      ensures that even if called multiple times, unchanged files are skipped
    - Partial file writes are handled by try/except around ast.parse —
      SyntaxError files are skipped and logged, not crashed on

Usage:
    # Called by watcher pipeline with a list of changed paths:
    python code_index_hook.py /path/to/changed1.py /path/to/changed2.py

    # Re-index an entire directory:
    python code_index_hook.py --directory /path/to/scripts/

    # Dry-run (parse only, no writes):
    python code_index_hook.py --dry-run /path/to/file.py

Features:
    - Filters to .py files only (non-Python changes ignored)
    - Incremental: file hash skip in parse_code_ast prevents redundant work
    - Resilient to partial writes: SyntaxError → skip, not crash
    - Re-embeds only changed chunks (LEFT JOIN filter)
    - Updates Neo4j graph after re-parse
    - Detailed stats output for pipeline integration
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

import psycopg

LOGGER = logging.getLogger("openclaw.code_index_hook")
logging.basicConfig(
    level=os.environ.get("OPENCLAW_LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)

DB_DSN = (os.environ.get("OPENCLAW_MEMORY_DSN") or os.environ.get("OPENCLAW_MEMORY_DB_DSN") or "dbname=openclaw_memory")  # honor both canonical DSN env names
SCRIPT_DIR = Path(__file__).resolve().parent

# Ensure imports work
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))


def filter_python_files(paths: list[str]) -> list[str]:
    """Filter to only .py files that exist."""
    result = []
    for p in paths:
        path = Path(p).resolve()
        if path.suffix == ".py" and path.exists():
            result.append(str(path))
        elif path.suffix == ".py" and not path.exists():
            LOGGER.info("Skipping deleted file: %s", p)
    return result


def reparse_files(
    py_files: list[str],
    *,
    project_id: str | None = None,
    force: bool = False,
    dry_run: bool = False,
) -> dict[str, int]:
    """
    Re-parse changed Python files using parse_code_ast.

    Returns stats dict with parsed/upserted/removed/skipped/errors counts.
    """
    from parse_code_ast import process_file, parse_file

    stats = {"parsed": 0, "upserted": 0, "removed": 0, "skipped": 0, "errors": 0, "files": 0}

    if dry_run:
        for fp in py_files:
            try:
                content = Path(fp).read_text(encoding="utf-8")
                chunks = parse_file(fp, content)
                stats["parsed"] += len(chunks)
                stats["files"] += 1
                LOGGER.info("DRY-RUN %s: %d chunks", Path(fp).name, len(chunks))
            except Exception as e:
                LOGGER.error("Error parsing %s: %s", fp, e)
                stats["errors"] += 1
        return stats

    conn = psycopg.connect(DB_DSN)
    try:
        for fp in py_files:
            try:
                file_stats = process_file(
                    conn, fp,
                    project_id=project_id,
                    force=force,
                )
                stats["parsed"] += file_stats["parsed"]
                stats["upserted"] += file_stats["upserted"]
                stats["removed"] += file_stats["removed"]
                stats["skipped"] += file_stats["skipped"]
                stats["files"] += 1

                if file_stats["skipped"]:
                    LOGGER.debug("SKIP %s (unchanged)", Path(fp).name)
                else:
                    LOGGER.info("PARSED %s: %d chunks, %d stale removed",
                                Path(fp).name, file_stats["upserted"], file_stats["removed"])
            except Exception as e:
                LOGGER.error("Error processing %s: %s", fp, e)
                stats["errors"] += 1
    finally:
        conn.close()

    return stats


def reembed_changed_chunks() -> int:
    """
    Re-embed any code chunks that don't have embeddings yet.
    Uses the embed_code_chunks.py script's incremental mode (LEFT JOIN filter).
    Returns number of newly embedded chunks.
    """
    try:
        from embed_code_chunks import (
            ensure_code_chunks_collection,
            ensure_embedding_table,
            fetch_chunks_to_embed,
            build_embedding_text,
            upsert_pg_embedding,
            upsert_code_vectors_batch,
            code_chunk_point_id,
            get_qdrant_client,
            MODEL_NAME,
        )
        from qdrant_client.http import models as qdrant_models
        from sentence_transformers import SentenceTransformer

        conn = psycopg.connect(DB_DSN)
        try:
            ensure_embedding_table(conn)
            chunks = fetch_chunks_to_embed(conn, force=False)
            if not chunks:
                return 0

            LOGGER.info("Re-embedding %d new/changed code chunks", len(chunks))

            model = SentenceTransformer(MODEL_NAME, device="cpu")
            ensure_code_chunks_collection()
            qdrant = get_qdrant_client()

            texts = [build_embedding_text(c) for c in chunks]
            embeddings = model.encode(
                texts,
                batch_size=64,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )

            points = []
            # strict=True: chunk↔embedding alignment is load-bearing (upsert
            # keyed by chunk["id"]); never silently truncate/mis-pair.
            for chunk, emb in zip(chunks, embeddings, strict=True):
                vector = emb.tolist()
                upsert_pg_embedding(conn, chunk["id"], MODEL_NAME, vector)
                points.append(
                    qdrant_models.PointStruct(
                        id=code_chunk_point_id(chunk["id"]),
                        vector=vector,
                        payload={
                            "chunk_id": chunk["id"],
                            "file_path": chunk["file_path"],
                            "chunk_type": chunk["chunk_type"],
                            "qualified_name": chunk["qualified_name"],
                            "module_name": chunk["module_name"],
                            "signature": chunk["signature"],
                            "docstring": chunk["docstring"],
                            "project_id": chunk["project_id"],
                            "line_start": chunk["line_start"],
                            "line_end": chunk["line_end"],
                            "class_name": chunk["class_name"],
                        },
                    )
                )

            if points:
                upsert_code_vectors_batch(qdrant, points)

            conn.commit()
            return len(chunks)

        finally:
            conn.close()

    except Exception as e:
        LOGGER.error("Re-embedding failed: %s", e)
        return 0


def update_graph() -> dict[str, int]:
    """
    Re-sync the Neo4j code graph after re-parsing.
    Delegates to sync_code_graph.py which is fully idempotent.
    """
    try:
        from sync_code_graph import (
            fetch_all_chunks,
            sync_nodes,
            sync_relationships,
            get_neo4j_driver,
        )

        conn = psycopg.connect(DB_DSN)
        driver = get_neo4j_driver()
        stats = {
            "codefile_nodes": 0, "codemodule_nodes": 0,
            "function_nodes": 0, "class_nodes": 0,
            "defines_rels": 0, "has_method_rels": 0,
            "contains_rels": 0, "imports_rels": 0, "calls_rels": 0,
        }

        try:
            chunks = fetch_all_chunks(conn)
            if chunks:
                with driver.session() as session:
                    sync_nodes(session, chunks, stats)
                    sync_relationships(session, chunks, stats)
        finally:
            conn.close()
            driver.close()

        return stats

    except Exception as e:
        LOGGER.error("Graph update failed: %s", e)
        return {}


def handle_changed_paths(
    paths: list[str],
    *,
    project_id: str | None = None,
    force: bool = False,
    dry_run: bool = False,
    skip_embed: bool = False,
    skip_graph: bool = False,
) -> dict[str, Any]:
    """
    Full pipeline: filter → parse → embed → graph update.

    This is the main entry point for the watcher integration.
    """
    start = time.time()
    result: dict[str, Any] = {"steps": []}

    # Step 1: Filter to Python files
    py_files = filter_python_files(paths)
    if not py_files:
        result["skipped_reason"] = "no_python_files"
        result["elapsed_seconds"] = round(time.time() - start, 3)
        return result

    LOGGER.info("Processing %d Python files", len(py_files))

    # Step 2: Re-parse
    parse_stats = reparse_files(py_files, project_id=project_id, force=force, dry_run=dry_run)
    result["steps"].append({"name": "parse", **parse_stats})

    if dry_run:
        result["elapsed_seconds"] = round(time.time() - start, 3)
        return result

    # Step 3: Re-embed (only if new chunks were parsed)
    if not skip_embed and parse_stats["upserted"] > 0:
        embedded = reembed_changed_chunks()
        result["steps"].append({"name": "embed", "embedded": embedded})
    elif not skip_embed:
        result["steps"].append({"name": "embed", "embedded": 0, "note": "no_new_chunks"})

    # Step 4: Update graph (only if changes were made)
    if not skip_graph and parse_stats["upserted"] > 0:
        graph_stats = update_graph()
        result["steps"].append({"name": "graph", **graph_stats})
    elif not skip_graph:
        result["steps"].append({"name": "graph", "note": "no_changes_to_sync"})

    result["elapsed_seconds"] = round(time.time() - start, 3)
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Code index hook: re-parse, re-embed, and update graph for changed files."
    )
    parser.add_argument("paths", nargs="*", help="Changed file paths")
    parser.add_argument("--directory", help="Re-index all .py files in a directory")
    parser.add_argument("--project-id", default=None)
    parser.add_argument("--force", action="store_true", help="Force re-parse even if hash unchanged")
    parser.add_argument("--dry-run", action="store_true", help="Parse only, no writes")
    parser.add_argument("--skip-embed", action="store_true", help="Skip embedding step")
    parser.add_argument("--skip-graph", action="store_true", help="Skip graph update step")
    args = parser.parse_args()

    paths = list(args.paths or [])

    if args.directory:
        from parse_code_ast import discover_python_files
        paths.extend(discover_python_files(args.directory))

    if not paths:
        LOGGER.error("No paths provided")
        sys.exit(1)

    result = handle_changed_paths(
        paths,
        project_id=args.project_id,
        force=args.force,
        dry_run=args.dry_run,
        skip_embed=args.skip_embed,
        skip_graph=args.skip_graph,
    )

    import json
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
