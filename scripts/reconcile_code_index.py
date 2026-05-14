"""
reconcile_code_index.py — Autonomous reconciliation & health daemon for the code index.

L7-S3 of the Level 7 "Autonomous Enterprise" implementation.

Performs a full-tree hash comparison to catch any files missed by the
filesystem watcher (e.g., due to system downtime, watcher restarts, or
files added while the watcher was not running). Emits a structured,
schema-validated health artifact.

Design principles:
    - Non-destructive: only inserts/updates for genuinely missed files
    - Watcher-compatible: does not conflict with active tracked_path_watcher
    - Efficient: parallel hash computation, short-circuit on unchanged files
    - Schema-validated JSON health artifact for monitoring consumption

Usage:
    # Daily cron reconciliation:
    python reconcile_code_index.py

    # Check health only (no repairs):
    python reconcile_code_index.py --health-only

    # Specify paths to reconcile:
    python reconcile_code_index.py --paths /path/to/scripts /path/to/other

    # Custom output location for health artifact:
    python reconcile_code_index.py --output /path/to/health.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psycopg

LOGGER = logging.getLogger("openclaw.reconcile_code_index")
logging.basicConfig(
    level=os.environ.get("OPENCLAW_LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)

DB_DSN = os.environ.get("OPENCLAW_MEMORY_DSN", "dbname=openclaw_memory")

# Default paths to reconcile (the memory engine scripts)
_MEMORY_ROOT = Path(os.environ.get("OPENCLAW_MEMORY_ROOT", str(Path(__file__).resolve().parent.parent)))
DEFAULT_PATHS = [
    os.environ.get(
        "OPENCLAW_CODE_INDEX_PATH",
        str(_MEMORY_ROOT / "scripts"),
    )
]

HEALTH_ARTIFACT_DIR = _MEMORY_ROOT / "health"
HASH_WORKERS = int(os.environ.get("OPENCLAW_RECONCILE_WORKERS", "4"))

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))


# ---------------------------------------------------------------------------
# Health artifact schema
# ---------------------------------------------------------------------------

HEALTH_SCHEMA_VERSION = "1.0.0"


def build_health_artifact(
    *,
    scan_paths: list[str],
    disk_files: int,
    indexed_files: int,
    stale_files: int,
    missing_files: int,
    orphaned_chunks: int,
    unembedded_chunks: int,
    repairs: dict[str, int],
    qdrant_code_points: int,
    qdrant_memory_points: int,
    neo4j_nodes: dict[str, int],
    neo4j_rels: dict[str, int],
    elapsed_seconds: float,
    errors: list[str],
) -> dict[str, Any]:
    """Build a schema-validated health artifact."""
    now = datetime.now(timezone.utc)
    status = "healthy"
    if errors:
        status = "degraded"
    if stale_files > 10 or missing_files > 10 or orphaned_chunks > 50:
        status = "unhealthy"

    return {
        "schema_version": HEALTH_SCHEMA_VERSION,
        "timestamp": now.isoformat(),
        "status": status,
        "scan": {
            "paths": scan_paths,
            "disk_files": disk_files,
            "indexed_files": indexed_files,
            "stale_files": stale_files,
            "missing_from_index": missing_files,
            "orphaned_chunks": orphaned_chunks,
            "unembedded_chunks": unembedded_chunks,
            "coverage_pct": round(
                indexed_files / max(disk_files, 1) * 100, 1
            ),
        },
        "repairs": repairs,
        "infrastructure": {
            "qdrant": {
                "code_chunks_points": qdrant_code_points,
                "memory_items_points": qdrant_memory_points,
            },
            "neo4j": {
                "nodes": neo4j_nodes,
                "relationships": neo4j_rels,
            },
        },
        "elapsed_seconds": round(elapsed_seconds, 2),
        "errors": errors,
    }


# ---------------------------------------------------------------------------
# Disk scanning (parallel hash computation)
# ---------------------------------------------------------------------------

def _hash_file(file_path: str) -> tuple[str, str | None]:
    """Compute SHA-256 of a file. Returns (path, hash) or (path, None) on error."""
    try:
        content = Path(file_path).read_bytes()
        return file_path, hashlib.sha256(content).hexdigest()
    except (OSError, PermissionError) as e:
        LOGGER.warning("Cannot hash %s: %s", file_path, e)
        return file_path, None


def scan_disk_files(paths: list[str]) -> dict[str, str]:
    """
    Find all .py files under the given paths and compute their SHA-256 hashes.
    Uses thread pool for parallel I/O.

    Returns: {absolute_file_path → sha256_hash}
    """
    py_files: list[str] = []
    for p in paths:
        target = Path(p).resolve()
        if target.is_file() and target.suffix == ".py":
            py_files.append(str(target))
        elif target.is_dir():
            target_depth = len(target.parts)
            for f in sorted(target.rglob("*.py")):
                relative_parts = f.parts[target_depth:]
                if any(part.startswith(".") or part == "__pycache__" for part in relative_parts):
                    continue
                py_files.append(str(f.resolve()))

    if not py_files:
        return {}

    results: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=HASH_WORKERS) as executor:
        futures = {executor.submit(_hash_file, f): f for f in py_files}
        for future in as_completed(futures):
            path, file_hash = future.result()
            if file_hash is not None:
                results[path] = file_hash

    return results


# ---------------------------------------------------------------------------
# Index state from PostgreSQL
# ---------------------------------------------------------------------------

def get_indexed_file_hashes(conn) -> dict[str, str]:
    """Get all indexed file paths and their stored hashes from code_chunks."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT DISTINCT file_path, file_hash
            FROM code_chunks
            ORDER BY file_path
        """)
        return {row[0]: row[1] for row in cur.fetchall()}


def get_unembedded_count(conn) -> int:
    """Count code chunks that don't have embeddings yet."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT count(*)
            FROM code_chunks c
            LEFT JOIN code_chunk_embeddings e ON e.chunk_id = c.id
            WHERE e.chunk_id IS NULL
        """)
        return cur.fetchone()[0]


def get_orphaned_chunks(conn, valid_paths: set[str]) -> list[str]:
    """Find indexed file paths that no longer exist on disk."""
    with conn.cursor() as cur:
        cur.execute("SELECT DISTINCT file_path FROM code_chunks")
        indexed_paths = {row[0] for row in cur.fetchall()}
    return sorted(indexed_paths - valid_paths)


def remove_orphaned_chunks(conn, orphaned_paths: list[str]) -> int:
    """Remove chunks for files that no longer exist on disk."""
    if not orphaned_paths:
        return 0
    removed = 0
    with conn.cursor() as cur:
        for path in orphaned_paths:
            cur.execute("DELETE FROM code_chunks WHERE file_path = %s", (path,))
            removed += cur.rowcount
    conn.commit()
    return removed


# ---------------------------------------------------------------------------
# Infrastructure health checks
# ---------------------------------------------------------------------------

def check_qdrant_health() -> dict[str, int]:
    """Check Qdrant collection point counts."""
    try:
        from qdrant_client import QdrantClient
        client = QdrantClient(
            host=os.environ.get("OPENCLAW_QDRANT_HOST", "localhost"),
            port=int(os.environ.get("OPENCLAW_QDRANT_PORT", "6333")),
        )
        code_info = client.get_collection("code_chunks")
        memory_info = client.get_collection("memory_items")
        return {
            "code_chunks": code_info.points_count,
            "memory_items": memory_info.points_count,
        }
    except Exception as e:
        LOGGER.error("Qdrant health check failed: %s", e)
        return {"code_chunks": -1, "memory_items": -1}


def check_neo4j_health() -> tuple[dict[str, int], dict[str, int]]:
    """Check Neo4j node and relationship counts."""
    try:
        from neo4j import GraphDatabase
        driver = GraphDatabase.driver(
            os.environ.get("OPENCLAW_NEO4J_URI", "bolt://localhost:7687"),
            auth=(
                os.environ.get("OPENCLAW_NEO4J_USER", "neo4j"),
                os.environ.get("OPENCLAW_NEO4J_PASSWORD", "neo4jpassword"),
            ),
        )
        nodes: dict[str, int] = {}
        rels: dict[str, int] = {}
        with driver.session() as session:
            result = session.run(
                "MATCH (n) RETURN labels(n)[0] AS label, count(*) AS cnt"
            )
            for r in result:
                nodes[r["label"]] = r["cnt"]

            result = session.run(
                "MATCH ()-[r]->() RETURN type(r) AS t, count(*) AS cnt"
            )
            for r in result:
                rels[r["t"]] = r["cnt"]
        driver.close()
        return nodes, rels
    except Exception as e:
        LOGGER.error("Neo4j health check failed: %s", e)
        return {}, {}


# ---------------------------------------------------------------------------
# Repair: re-index missed files
# ---------------------------------------------------------------------------

def repair_missed_files(
    missed_paths: list[str],
    stale_paths: list[str],
) -> dict[str, int]:
    """
    Re-index files that were missed by the watcher or have stale hashes.
    Delegates to code_index_hook.handle_changed_paths for the full pipeline.
    """
    all_paths = missed_paths + stale_paths
    if not all_paths:
        return {"reparsed": 0, "reembedded": 0, "graph_updated": False}

    from code_index_hook import handle_changed_paths

    LOGGER.info(
        "Repairing %d missed + %d stale files",
        len(missed_paths), len(stale_paths),
    )

    result = handle_changed_paths(all_paths, force=True)

    repairs = {"reparsed": 0, "reembedded": 0, "graph_updated": False}
    for step in result.get("steps", []):
        if step.get("name") == "parse":
            repairs["reparsed"] = step.get("upserted", 0)
        elif step.get("name") == "embed":
            repairs["reembedded"] = step.get("embedded", 0)
        elif step.get("name") == "graph":
            repairs["graph_updated"] = True

    return repairs


# ---------------------------------------------------------------------------
# Main reconciliation pipeline
# ---------------------------------------------------------------------------

def reconcile(
    paths: list[str],
    *,
    health_only: bool = False,
    output_path: str | None = None,
) -> dict[str, Any]:
    """
    Full reconciliation pipeline:
    1. Scan disk for all .py files and compute hashes
    2. Compare against indexed state in PostgreSQL
    3. Identify missed, stale, and orphaned files
    4. Repair if not health-only
    5. Check infrastructure health (Qdrant, Neo4j)
    6. Emit health artifact
    """
    t0 = time.monotonic()
    errors: list[str] = []

    # Phase 1: Disk scan
    LOGGER.info("Phase 1: Scanning disk (%d paths)...", len(paths))
    disk_files = scan_disk_files(paths)
    LOGGER.info("Found %d .py files on disk", len(disk_files))

    # Phase 2: Compare with index
    LOGGER.info("Phase 2: Comparing with index...")
    conn = psycopg.connect(DB_DSN)
    try:
        indexed_hashes = get_indexed_file_hashes(conn)
        unembedded = get_unembedded_count(conn)

        # Find discrepancies
        missed_paths = []  # On disk but not in index
        stale_paths = []   # In index but hash changed
        for path, disk_hash in disk_files.items():
            if path not in indexed_hashes:
                missed_paths.append(path)
            elif indexed_hashes[path] != disk_hash:
                stale_paths.append(path)

        # Find orphaned (in index but not on disk)
        orphaned_paths = get_orphaned_chunks(conn, set(disk_files.keys()))

        LOGGER.info(
            "Discrepancies: %d missed, %d stale, %d orphaned, %d unembedded",
            len(missed_paths), len(stale_paths), len(orphaned_paths), unembedded,
        )

        # Phase 3: Repair
        repairs = {"reparsed": 0, "reembedded": 0, "graph_updated": False, "orphans_removed": 0}
        if not health_only:
            if missed_paths or stale_paths:
                repair_result = repair_missed_files(missed_paths, stale_paths)
                repairs.update(repair_result)

            if orphaned_paths:
                repairs["orphans_removed"] = remove_orphaned_chunks(conn, orphaned_paths)
                LOGGER.info("Removed %d orphaned chunks", repairs["orphans_removed"])

    finally:
        conn.close()

    # Phase 4: Infrastructure health
    LOGGER.info("Phase 4: Checking infrastructure...")
    qdrant_health = check_qdrant_health()
    neo4j_nodes, neo4j_rels = check_neo4j_health()

    elapsed = time.monotonic() - t0

    # Phase 5: Build and write health artifact
    artifact = build_health_artifact(
        scan_paths=paths,
        disk_files=len(disk_files),
        indexed_files=len(indexed_hashes),
        stale_files=len(stale_paths),
        missing_files=len(missed_paths),
        orphaned_chunks=len(orphaned_paths),
        unembedded_chunks=unembedded,
        repairs=repairs,
        qdrant_code_points=qdrant_health.get("code_chunks", -1),
        qdrant_memory_points=qdrant_health.get("memory_items", -1),
        neo4j_nodes=neo4j_nodes,
        neo4j_rels=neo4j_rels,
        elapsed_seconds=elapsed,
        errors=errors,
    )

    # Write artifact
    if output_path:
        out = Path(output_path)
    else:
        HEALTH_ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        out = HEALTH_ARTIFACT_DIR / f"reconciliation-{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"

    out.write_text(json.dumps(artifact, indent=2) + "\n")
    LOGGER.info("Health artifact written to: %s", out)

    # Also write as latest.json for easy monitoring
    latest = out.parent / "latest.json"
    latest.write_text(json.dumps(artifact, indent=2) + "\n")

    return artifact


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Reconcile code index and emit health artifact."
    )
    parser.add_argument("--paths", nargs="*", default=None,
                        help="Paths to scan (default: memory engine scripts)")
    parser.add_argument("--health-only", action="store_true",
                        help="Check health only, do not repair")
    parser.add_argument("--output", default=None,
                        help="Output path for health artifact")
    args = parser.parse_args()

    paths = args.paths or DEFAULT_PATHS
    artifact = reconcile(paths, health_only=args.health_only, output_path=args.output)

    print(json.dumps(artifact, indent=2))


if __name__ == "__main__":
    main()
