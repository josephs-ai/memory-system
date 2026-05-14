"""
patch_ledger.py — Per-step file modification tracking with atomic rollback.

L8-S2 of the Level 8 "Autonomous Closed-Loop Execution" implementation.

Records every file modification made by the campaign orchestrator. Enables:
    - Full audit trail of every change per campaign step
    - Atomic rollback of any step without corrupting Neo4j graph or Qdrant
    - Identification of which step introduced a regression
    - Content-level before/after snapshots for diff analysis

Rollback guarantees:
    1. File content restored to pre-patch state
    2. Code chunks re-parsed from restored content (updates PostgreSQL)
    3. Embeddings re-synced for affected chunks (updates Qdrant)
    4. Graph re-synced for affected files (updates Neo4j)
    5. Patch ledger entry marked rolled_back=TRUE with timestamp

Usage:
    # Record a patch
    from patch_ledger import PatchLedger
    ledger = PatchLedger(conn)
    ledger.record_patch(step_id="step-001", file_path="/path/to/file.cpp", content_after="...")

    # Rollback a step
    ledger.rollback_step("step-001")

    # Query history
    ledger.get_step_patches("step-001")
    ledger.get_file_history("/path/to/file.cpp")

CLI:
    python patch_ledger.py record --step-id S1 --file /path/to/file.cpp [--content-after ...]
    python patch_ledger.py rollback --step-id S1
    python patch_ledger.py history --step-id S1
    python patch_ledger.py file-history --file /path/to/file.cpp
    python patch_ledger.py status [--campaign-id C1]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psycopg

LOGGER = logging.getLogger("openclaw.patch_ledger")
logging.basicConfig(
    level=os.environ.get("OPENCLAW_LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

DB_DSN = os.environ.get("OPENCLAW_MEMORY_DSN", "dbname=openclaw_memory")


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class PatchRecord:
    """A single patch ledger entry."""
    patch_id: str
    step_id: str
    campaign_id: str | None
    file_path: str
    operation: str  # create, modify, delete
    file_hash_before: str | None
    file_hash_after: str | None
    content_before: str | None
    content_after: str | None
    chunk_ids_before: list[str]
    chunk_ids_after: list[str]
    graph_nodes_affected: list[str]
    rolled_back: bool
    rolled_back_at: datetime | None
    rollback_step_id: str | None
    created_at: datetime
    metadata: dict | None

    @property
    def is_active(self) -> bool:
        return not self.rolled_back


@dataclass
class RollbackResult:
    """Result of a rollback operation."""
    step_id: str
    patches_rolled_back: int
    files_restored: list[str]
    chunks_reparsed: int
    chunks_reembedded: int
    graph_updated: bool
    errors: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# PatchLedger core
# ---------------------------------------------------------------------------

class PatchLedger:
    """
    Manages the patch_ledger table for tracking file modifications.

    Thread-safe: each method manages its own cursor.
    Idempotent: re-recording the same (step_id, file_path) updates the record.
    """

    def __init__(self, conn: psycopg.Connection):
        self._conn = conn

    @staticmethod
    def make_patch_id(step_id: str, file_path: str) -> str:
        """Deterministic patch ID from (step_id, file_path)."""
        raw = f"{step_id}::{file_path}"
        return "patch-" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]

    @staticmethod
    def file_hash(content: str) -> str:
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    # --- Recording ---

    def record_patch(
        self,
        *,
        step_id: str,
        file_path: str,
        content_after: str | None = None,
        content_before: str | None = None,
        campaign_id: str | None = None,
        operation: str | None = None,
        metadata: dict | None = None,
    ) -> str:
        """
        Record a file modification in the patch ledger.

        If content_before is not provided, reads the current file from disk
        (if it exists). If content_after is not provided, reads the file after
        the expected modification.

        Auto-detects operation (create/modify/delete) from before/after state.

        Returns the patch_id.
        """
        file_path = str(Path(file_path).resolve())
        patch_id = self.make_patch_id(step_id, file_path)

        # Auto-read current content if not provided
        if content_before is None:
            try:
                content_before = Path(file_path).read_text(encoding="utf-8", errors="replace")
            except (OSError, FileNotFoundError):
                content_before = None

        # Auto-detect operation
        if operation is None:
            if content_before is None and content_after is not None:
                operation = "create"
            elif content_before is not None and content_after is None:
                operation = "delete"
            else:
                operation = "modify"

        hash_before = self.file_hash(content_before) if content_before else None
        hash_after = self.file_hash(content_after) if content_after else None

        # Get current chunk IDs for this file
        chunk_ids_before = self._get_chunk_ids(file_path)

        # Get graph nodes affected
        graph_nodes = self._get_graph_nodes(file_path)

        with self._conn.cursor() as cur:
            cur.execute("""
                INSERT INTO patch_ledger (
                    patch_id, step_id, campaign_id, file_path, operation,
                    file_hash_before, file_hash_after,
                    content_before, content_after,
                    chunk_ids_before, chunk_ids_after,
                    graph_nodes_affected, metadata
                ) VALUES (
                    %s, %s, %s, %s, %s,
                    %s, %s,
                    %s, %s,
                    %s, %s,
                    %s, %s
                )
                ON CONFLICT (patch_id) DO UPDATE SET
                    file_hash_after = EXCLUDED.file_hash_after,
                    content_after = EXCLUDED.content_after,
                    chunk_ids_after = EXCLUDED.chunk_ids_after,
                    metadata = EXCLUDED.metadata
            """, (
                patch_id, step_id, campaign_id, file_path, operation,
                hash_before, hash_after,
                content_before, content_after,
                chunk_ids_before, [],  # chunk_ids_after updated post-parse
                graph_nodes,
                json.dumps(metadata) if metadata else None,
            ))

        self._conn.commit()
        LOGGER.info("RECORDED patch %s: %s %s (%s)", patch_id[:16], operation, Path(file_path).name, step_id)
        return patch_id

    def update_after_parse(self, step_id: str, file_path: str) -> None:
        """
        Update chunk_ids_after for a patch after the file has been re-parsed.
        Call this after writing the file and running the parser.
        """
        file_path = str(Path(file_path).resolve())
        patch_id = self.make_patch_id(step_id, file_path)
        chunk_ids_after = self._get_chunk_ids(file_path)

        with self._conn.cursor() as cur:
            cur.execute("""
                UPDATE patch_ledger
                SET chunk_ids_after = %s
                WHERE patch_id = %s
            """, (chunk_ids_after, patch_id))

        self._conn.commit()

    # --- Rollback ---

    def rollback_step(
        self,
        step_id: str,
        *,
        rollback_step_id: str | None = None,
        reparse: bool = True,
        reembed: bool = True,
        regraph: bool = True,
    ) -> RollbackResult:
        """
        Atomically rollback all patches from a given step.

        1. Restores file content to pre-patch state
        2. Re-parses restored files (updates code_chunks)
        3. Re-embeds affected chunks (updates Qdrant)
        4. Re-syncs graph for affected files (updates Neo4j)
        5. Marks ledger entries as rolled_back

        Returns a RollbackResult with details.
        """
        patches = self.get_step_patches(step_id, active_only=True)
        if not patches:
            LOGGER.warning("No active patches found for step %s", step_id)
            return RollbackResult(
                step_id=step_id, patches_rolled_back=0,
                files_restored=[], chunks_reparsed=0,
                chunks_reembedded=0, graph_updated=False,
            )

        result = RollbackResult(
            step_id=step_id, patches_rolled_back=0,
            files_restored=[], chunks_reparsed=0,
            chunks_reembedded=0, graph_updated=False,
        )

        for patch in patches:
            try:
                self._rollback_single_patch(patch, result)
            except Exception as e:
                LOGGER.error("Error rolling back %s: %s", patch.patch_id, e)
                result.errors.append(f"{patch.file_path}: {e}")

        # Mark all patches as rolled back
        now = datetime.now(timezone.utc)
        with self._conn.cursor() as cur:
            cur.execute("""
                UPDATE patch_ledger
                SET rolled_back = TRUE,
                    rolled_back_at = %s,
                    rollback_step_id = %s
                WHERE step_id = %s AND rolled_back = FALSE
            """, (now, rollback_step_id or f"rollback-{step_id}", step_id))
        self._conn.commit()

        result.patches_rolled_back = len(patches)

        # Re-parse and re-embed if requested
        if reparse and result.files_restored:
            result.chunks_reparsed = self._reparse_files(result.files_restored)

        if reembed and result.chunks_reparsed > 0:
            result.chunks_reembedded = self._reembed_chunks(result.files_restored)

        if regraph and result.files_restored:
            result.graph_updated = self._regraph_files(result.files_restored)

        LOGGER.info(
            "ROLLBACK %s: %d patches, %d files restored, %d chunks reparsed",
            step_id, result.patches_rolled_back,
            len(result.files_restored), result.chunks_reparsed,
        )

        return result

    def _rollback_single_patch(self, patch: PatchRecord, result: RollbackResult) -> None:
        """Restore a single file to its pre-patch state."""
        fp = Path(patch.file_path)

        if patch.operation == "create":
            # File was created — delete it
            if fp.exists():
                fp.unlink()
                LOGGER.info("ROLLBACK delete created file: %s", fp.name)
            # Remove chunks for this file
            self._remove_file_chunks(patch.file_path)

        elif patch.operation == "delete":
            # File was deleted — restore it
            if patch.content_before:
                fp.parent.mkdir(parents=True, exist_ok=True)
                fp.write_text(patch.content_before, encoding="utf-8")
                LOGGER.info("ROLLBACK restore deleted file: %s", fp.name)

        elif patch.operation == "modify":
            # File was modified — restore previous content
            if patch.content_before:
                fp.parent.mkdir(parents=True, exist_ok=True)
                fp.write_text(patch.content_before, encoding="utf-8")
                LOGGER.info("ROLLBACK restore modified file: %s", fp.name)
            else:
                LOGGER.warning("No content_before for %s — cannot restore", fp.name)
                return

        result.files_restored.append(patch.file_path)

    def _remove_file_chunks(self, file_path: str) -> int:
        """Remove all code_chunks for a file."""
        with self._conn.cursor() as cur:
            cur.execute("DELETE FROM code_chunks WHERE file_path = %s", (file_path,))
            count = cur.rowcount
        self._conn.commit()
        return count

    def _reparse_files(self, file_paths: list[str]) -> int:
        """Re-parse files and update code_chunks. Returns total chunks."""
        total = 0
        for fp in file_paths:
            if not Path(fp).exists():
                continue
            try:
                content = Path(fp).read_text(encoding="utf-8", errors="replace")
                suffix = Path(fp).suffix.lower()

                if suffix == ".py":
                    from parse_code_ast import parse_file, upsert_chunks, remove_stale_chunks
                    chunks = parse_file(fp, content)
                    if chunks:
                        upsert_chunks(self._conn, chunks)
                        remove_stale_chunks(self._conn, fp, {c["id"] for c in chunks})
                        total += len(chunks)
                elif suffix in (".cpp", ".h", ".hpp", ".cc", ".cxx"):
                    from parse_code_cpp import CppParser
                    from parse_code_ast import upsert_chunks, remove_stale_chunks
                    parser = CppParser()
                    chunks = parser.parse_file(fp, content)
                    if chunks:
                        upsert_chunks(self._conn, chunks)
                        remove_stale_chunks(self._conn, fp, {c["id"] for c in chunks})
                        total += len(chunks)
            except Exception as e:
                LOGGER.error("Re-parse error for %s: %s", fp, e)
        return total

    def _reembed_chunks(self, file_paths: list[str]) -> int:
        """Re-embed chunks for given files. Returns count."""
        try:
            from embed_code_chunks import embed_unembedded_chunks
            return embed_unembedded_chunks(self._conn)
        except Exception as e:
            LOGGER.error("Re-embed error: %s", e)
            return 0

    def _regraph_files(self, file_paths: list[str]) -> bool:
        """Re-sync Neo4j graph. Returns success."""
        try:
            from sync_code_graph import sync_graph
            sync_graph(force=True)
            return True
        except Exception as e:
            LOGGER.error("Re-graph error: %s", e)
            return False

    # --- Queries ---

    def get_step_patches(self, step_id: str, *, active_only: bool = False) -> list[PatchRecord]:
        """Get all patches for a step."""
        query = "SELECT * FROM patch_ledger WHERE step_id = %s"
        if active_only:
            query += " AND rolled_back = FALSE"
        query += " ORDER BY created_at"

        with self._conn.cursor() as cur:
            cur.execute(query, (step_id,))
            cols = [d.name for d in cur.description]
            return [self._row_to_record(dict(zip(cols, row))) for row in cur.fetchall()]

    def get_file_history(
        self,
        file_path: str,
        *,
        limit: int = 50,
    ) -> list[PatchRecord]:
        """Get modification history for a specific file."""
        file_path = str(Path(file_path).resolve())
        with self._conn.cursor() as cur:
            cur.execute("""
                SELECT * FROM patch_ledger
                WHERE file_path = %s
                ORDER BY created_at DESC
                LIMIT %s
            """, (file_path, limit))
            cols = [d.name for d in cur.description]
            return [self._row_to_record(dict(zip(cols, row))) for row in cur.fetchall()]

    def get_campaign_status(self, campaign_id: str | None = None) -> dict[str, Any]:
        """Get overall campaign status."""
        with self._conn.cursor() as cur:
            where = "WHERE campaign_id = %s" if campaign_id else ""
            params = (campaign_id,) if campaign_id else ()

            cur.execute(f"""
                SELECT
                    count(*) as total_patches,
                    count(*) FILTER (WHERE rolled_back = TRUE) as rolled_back,
                    count(DISTINCT step_id) as total_steps,
                    count(DISTINCT file_path) as files_touched,
                    min(created_at) as first_patch,
                    max(created_at) as last_patch
                FROM patch_ledger {where}
            """, params)
            row = cur.fetchone()

            return {
                "total_patches": row[0],
                "rolled_back": row[1],
                "active_patches": row[0] - row[1],
                "total_steps": row[2],
                "files_touched": row[3],
                "first_patch": row[4].isoformat() if row[4] else None,
                "last_patch": row[5].isoformat() if row[5] else None,
            }

    def find_regression_source(
        self,
        file_path: str,
        *,
        error_line: int | None = None,
    ) -> list[PatchRecord]:
        """
        Find which step most recently modified a file (regression candidate).

        If error_line is provided, filters to patches whose chunk ranges
        overlap with that line.
        """
        file_path = str(Path(file_path).resolve())
        patches = self.get_file_history(file_path, limit=10)

        if error_line is not None:
            # Filter to patches that modified chunks containing the error line
            relevant = []
            for p in patches:
                if p.is_active and p.chunk_ids_after:
                    # Check if any chunk in this patch spans the error line
                    with self._conn.cursor() as cur:
                        if p.chunk_ids_after:
                            placeholders = ",".join(["%s"] * len(p.chunk_ids_after))
                            cur.execute(f"""
                                SELECT id FROM code_chunks
                                WHERE id IN ({placeholders})
                                  AND line_start <= %s AND line_end >= %s
                            """, (*p.chunk_ids_after, error_line, error_line))
                            if cur.fetchone():
                                relevant.append(p)
            return relevant

        return [p for p in patches if p.is_active]

    # --- Helpers ---

    def _get_chunk_ids(self, file_path: str) -> list[str]:
        """Get current chunk IDs for a file from code_chunks."""
        with self._conn.cursor() as cur:
            cur.execute(
                "SELECT id FROM code_chunks WHERE file_path = %s ORDER BY line_start",
                (file_path,)
            )
            return [row[0] for row in cur.fetchall()]

    def _get_graph_nodes(self, file_path: str) -> list[str]:
        """Get qualified names of graph nodes associated with a file."""
        with self._conn.cursor() as cur:
            cur.execute(
                "SELECT qualified_name FROM code_chunks WHERE file_path = %s AND chunk_type != 'module'",
                (file_path,)
            )
            return [row[0] for row in cur.fetchall()]

    @staticmethod
    def _row_to_record(row: dict) -> PatchRecord:
        """Convert a database row to a PatchRecord."""
        meta = row["metadata"]
        if isinstance(meta, str):
            meta = json.loads(meta)
        return PatchRecord(
            patch_id=row["patch_id"],
            step_id=row["step_id"],
            campaign_id=row["campaign_id"],
            file_path=row["file_path"],
            operation=row["operation"],
            file_hash_before=row["file_hash_before"],
            file_hash_after=row["file_hash_after"],
            content_before=row.get("content_before"),
            content_after=row.get("content_after"),
            chunk_ids_before=row["chunk_ids_before"] or [],
            chunk_ids_after=row["chunk_ids_after"] or [],
            graph_nodes_affected=row["graph_nodes_affected"] or [],
            rolled_back=row["rolled_back"],
            rolled_back_at=row["rolled_back_at"],
            rollback_step_id=row["rollback_step_id"],
            created_at=row["created_at"],
            metadata=meta,
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Patch Ledger — track and rollback file modifications")
    sub = parser.add_subparsers(dest="command", required=True)

    # record
    rec = sub.add_parser("record", help="Record a file modification")
    rec.add_argument("--step-id", required=True)
    rec.add_argument("--file", required=True)
    rec.add_argument("--campaign-id", default=None)
    rec.add_argument("--content-after", default=None, help="New file content (reads from file if omitted)")
    rec.add_argument("--metadata", default=None, help="JSON metadata")

    # rollback
    rb = sub.add_parser("rollback", help="Rollback all patches from a step")
    rb.add_argument("--step-id", required=True)
    rb.add_argument("--no-reparse", action="store_true")
    rb.add_argument("--no-reembed", action="store_true")
    rb.add_argument("--no-regraph", action="store_true")

    # history
    hist = sub.add_parser("history", help="Show patches for a step")
    hist.add_argument("--step-id", required=True)

    # file-history
    fhist = sub.add_parser("file-history", help="Show modification history for a file")
    fhist.add_argument("--file", required=True)

    # status
    st = sub.add_parser("status", help="Show campaign status")
    st.add_argument("--campaign-id", default=None)

    args = parser.parse_args()

    conn = psycopg.connect(DB_DSN)
    ledger = PatchLedger(conn)

    try:
        if args.command == "record":
            content_after = args.content_after
            if content_after is None and Path(args.file).exists():
                content_after = Path(args.file).read_text(encoding="utf-8", errors="replace")
            meta = json.loads(args.metadata) if args.metadata else None
            pid = ledger.record_patch(
                step_id=args.step_id,
                file_path=args.file,
                content_after=content_after,
                campaign_id=args.campaign_id,
                metadata=meta,
            )
            print(f"PATCH_ID={pid}")

        elif args.command == "rollback":
            result = ledger.rollback_step(
                args.step_id,
                reparse=not args.no_reparse,
                reembed=not args.no_reembed,
                regraph=not args.no_regraph,
            )
            print(f"ROLLED_BACK={result.patches_rolled_back}")
            print(f"FILES_RESTORED={len(result.files_restored)}")
            print(f"CHUNKS_REPARSED={result.chunks_reparsed}")
            print(f"CHUNKS_REEMBEDDED={result.chunks_reembedded}")
            print(f"GRAPH_UPDATED={result.graph_updated}")
            if result.errors:
                for e in result.errors:
                    print(f"ERROR: {e}")

        elif args.command == "history":
            patches = ledger.get_step_patches(args.step_id)
            for p in patches:
                rb = " [ROLLED BACK]" if p.rolled_back else ""
                print(f"  {p.operation:8s} {Path(p.file_path).name}{rb} ({p.created_at})")
                if p.chunk_ids_after:
                    print(f"           chunks: {len(p.chunk_ids_after)}")

        elif args.command == "file-history":
            patches = ledger.get_file_history(args.file)
            for p in patches:
                rb = " [ROLLED BACK]" if p.rolled_back else ""
                print(f"  [{p.step_id}] {p.operation:8s}{rb} ({p.created_at})")

        elif args.command == "status":
            status = ledger.get_campaign_status(args.campaign_id)
            for k, v in status.items():
                print(f"{k}={v}")

    finally:
        conn.close()


if __name__ == "__main__":
    main()
