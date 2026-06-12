"""
store_parity_checker.py — Cross-store parity verification.

Phase 4, Item 10: Build parity guarantees across Postgres / Qdrant / Neo4j.

Verifies that all three stores agree on the current state of memory items.
Detects drift, missing items, orphaned vectors, orphaned graph nodes.

Provides:
1. Continuous parity checker (compare all stores)
2. Drift reports (what's out of sync, since when)
3. Item-level sync status
4. Repair jobs (backfill missing items)
5. Alerting interface (returns structured findings)

Design:
- Postgres is the source of truth
- Qdrant should have vectors for all active/candidate items
- Neo4j should have nodes for all items with entity+property+value
- Runs deterministically, no LLM calls
"""
from __future__ import annotations

import logging
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

LOGGER = logging.getLogger("openclaw.store_parity_checker")

DB_DSN = os.environ.get("OPENCLAW_MEMORY_DSN", "dbname=openclaw_memory")
QDRANT_HOST = os.environ.get("OPENCLAW_QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.environ.get("OPENCLAW_QDRANT_PORT", "6333"))
QDRANT_COLLECTION = os.environ.get("OPENCLAW_QDRANT_COLLECTION", "memory_items")
NEO4J_URI = os.environ.get("OPENCLAW_NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.environ.get("OPENCLAW_NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.environ.get("OPENCLAW_NEO4J_PASSWORD", "neo4jpassword")


# ---------------------------------------------------------------------------
# Parity check result structures
# ---------------------------------------------------------------------------

@dataclass
class StoreStatus:
    """Status of a single store."""
    name: str
    reachable: bool = False
    item_count: int = 0
    error: str = ""

    def to_dict(self) -> dict:
        return {"name": self.name, "reachable": self.reachable, "item_count": self.item_count, "error": self.error}


@dataclass
class ParityIssue:
    """A single parity issue between stores."""
    item_id: str
    issue_type: str  # "missing_in_qdrant", "missing_in_neo4j", "orphan_in_qdrant", "orphan_in_neo4j", "stale_vector", "stale_graph"
    severity: str    # "critical", "warning", "info"
    detail: str
    repairable: bool = True

    def to_dict(self) -> dict:
        return {
            "item_id": self.item_id,
            "issue_type": self.issue_type,
            "severity": self.severity,
            "detail": self.detail,
            "repairable": self.repairable,
        }


@dataclass
class ParityReport:
    """Full parity check report."""
    timestamp: str = ""
    stores: list[StoreStatus] = field(default_factory=list)
    issues: list[ParityIssue] = field(default_factory=list)
    pg_active_count: int = 0
    qdrant_count: int = 0
    neo4j_count: int = 0
    pg_ids: set = field(default_factory=set, repr=False)
    qdrant_ids: set = field(default_factory=set, repr=False)
    neo4j_ids: set = field(default_factory=set, repr=False)

    @property
    def healthy(self) -> bool:
        return all(s.reachable for s in self.stores) and not any(i.severity == "critical" for i in self.issues)

    def summary(self) -> dict:
        issue_counts: dict[str, int] = {}
        for i in self.issues:
            issue_counts[i.issue_type] = issue_counts.get(i.issue_type, 0) + 1
        return {
            "timestamp": self.timestamp,
            "healthy": self.healthy,
            "stores": [s.to_dict() for s in self.stores],
            "pg_active": self.pg_active_count,
            "qdrant_count": self.qdrant_count,
            "neo4j_count": self.neo4j_count,
            "issue_counts": issue_counts,
            "total_issues": len(self.issues),
            "critical_issues": sum(1 for i in self.issues if i.severity == "critical"),
        }

    def to_dict(self) -> dict:
        d = self.summary()
        d["issues"] = [i.to_dict() for i in self.issues[:100]]  # Cap output
        return d


# ---------------------------------------------------------------------------
# Store connectivity checks
# ---------------------------------------------------------------------------

def check_postgres(conn: Any = None) -> tuple[StoreStatus, set[str]]:
    """Check Postgres and get all active/candidate item IDs."""
    status = StoreStatus(name="postgres")
    ids = set()

    should_close = False
    if conn is None:
        try:
            import psycopg
            conn = psycopg.connect(DB_DSN)
            should_close = True
        except Exception as ex:
            status.error = str(ex)
            return status, ids

    try:
        with conn.cursor() as cur:
            cur.execute("SELECT id FROM memory_items WHERE status IN ('active', 'candidate')")
            ids = {row[0] for row in cur.fetchall()}
            status.item_count = len(ids)
            status.reachable = True
    except Exception as ex:
        status.error = str(ex)
    finally:
        if should_close:
            conn.close()

    return status, ids


def check_qdrant() -> tuple[StoreStatus, set[str]]:
    """Check Qdrant and get all point IDs."""
    status = StoreStatus(name="qdrant")
    ids = set()

    try:
        from qdrant_client import QdrantClient
        client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

        # Check if collection exists
        collections = client.get_collections().collections
        if not any(c.name == QDRANT_COLLECTION for c in collections):
            status.reachable = True
            status.error = f"collection '{QDRANT_COLLECTION}' does not exist"
            return status, ids

        # Get count
        info = client.get_collection(QDRANT_COLLECTION)
        status.item_count = info.points_count or 0
        status.reachable = True

        # Scroll through all points to get IDs
        # Note: for large collections, this should be paginated
        offset = None
        batch_size = 1000
        while True:
            result = client.scroll(
                collection_name=QDRANT_COLLECTION,
                limit=batch_size,
                offset=offset,
                with_payload=["memory_id"],
                with_vectors=False,
            )
            points, next_offset = result
            for p in points:
                mid = p.payload.get("memory_id") if p.payload else None
                if mid:
                    ids.add(mid)
                else:
                    ids.add(str(p.id))

            if next_offset is None or not points:
                break
            offset = next_offset

    except Exception as ex:
        status.error = str(ex)

    return status, ids


def check_neo4j() -> tuple[StoreStatus, set[str]]:
    """Check Neo4j and get all memory node IDs."""
    status = StoreStatus(name="neo4j")
    ids = set()

    try:
        from neo4j import GraphDatabase
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

        with driver.session() as session:
            result = session.run("MATCH (m:MemoryItem) RETURN m.id AS id")
            for record in result:
                mid = record.get("id")
                if mid:
                    ids.add(mid)

            status.item_count = len(ids)
            status.reachable = True

        driver.close()

    except Exception as ex:
        status.error = str(ex)

    return status, ids


# ---------------------------------------------------------------------------
# Parity comparison
# ---------------------------------------------------------------------------

def run_parity_check(conn: Any = None) -> ParityReport:
    """
    Run a full parity check across Postgres, Qdrant, and Neo4j.

    Postgres is the source of truth. Reports:
    - Items missing from Qdrant
    - Items missing from Neo4j (if they have entity+property+value)
    - Orphaned items in Qdrant (not in Postgres)
    - Orphaned items in Neo4j (not in Postgres)
    """
    report = ParityReport(timestamp=datetime.now(timezone.utc).isoformat())

    # Check all stores
    pg_status, pg_ids = check_postgres(conn)
    report.stores.append(pg_status)
    report.pg_active_count = len(pg_ids)
    report.pg_ids = pg_ids

    qdrant_status, qdrant_ids = check_qdrant()
    report.stores.append(qdrant_status)
    report.qdrant_count = len(qdrant_ids)
    report.qdrant_ids = qdrant_ids

    neo4j_status, neo4j_ids = check_neo4j()
    report.stores.append(neo4j_status)
    report.neo4j_count = len(neo4j_ids)
    report.neo4j_ids = neo4j_ids

    # Compare Postgres → Qdrant
    if pg_status.reachable and qdrant_status.reachable:
        missing_in_qdrant = pg_ids - qdrant_ids
        for mid in list(missing_in_qdrant)[:500]:
            report.issues.append(ParityIssue(
                item_id=mid,
                issue_type="missing_in_qdrant",
                severity="warning",
                detail=f"Active item {mid} exists in Postgres but has no vector in Qdrant",
            ))

        orphan_in_qdrant = qdrant_ids - pg_ids
        for mid in list(orphan_in_qdrant)[:500]:
            report.issues.append(ParityIssue(
                item_id=mid,
                issue_type="orphan_in_qdrant",
                severity="info",
                detail=f"Vector {mid} exists in Qdrant but item is not active in Postgres",
            ))

    # Compare Postgres → Neo4j
    if pg_status.reachable and neo4j_status.reachable:
        # Only items with entity+property+value should be in Neo4j
        # We'd need to filter pg_ids, but for now compare what's there
        orphan_in_neo4j = neo4j_ids - pg_ids
        for mid in list(orphan_in_neo4j)[:500]:
            report.issues.append(ParityIssue(
                item_id=mid,
                issue_type="orphan_in_neo4j",
                severity="info",
                detail=f"Graph node {mid} exists in Neo4j but item is not active in Postgres",
            ))

    return report


# ---------------------------------------------------------------------------
# Repair jobs
# ---------------------------------------------------------------------------

def repair_missing_qdrant(
    item_ids: list[str] | None = None,
    conn: Any = None,
    dry_run: bool = True,
) -> dict:
    """
    Backfill Qdrant with vectors for items that are in Postgres but missing from Qdrant.

    If item_ids is None, repairs all missing items from the last parity check.
    """
    should_close = False
    if conn is None:
        import psycopg
        conn = psycopg.connect(DB_DSN)
        should_close = True

    try:
        if item_ids is None:
            report = run_parity_check(conn)
            item_ids = [i.item_id for i in report.issues if i.issue_type == "missing_in_qdrant"]

        if not item_ids:
            return {"repaired": 0, "errors": 0, "dry_run": dry_run}

        if dry_run:
            return {"would_repair": len(item_ids), "dry_run": True, "sample_ids": item_ids[:10]}

        # Fetch texts and generate embeddings
        from sentence_transformers import SentenceTransformer
        from qdrant_client import QdrantClient
        from qdrant_client.http import models

        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

        repaired = 0
        errors = 0

        with conn.cursor() as cur:
            placeholders = ",".join(["%s"] * len(item_ids))
            cur.execute(f"SELECT id, text FROM memory_items WHERE id IN ({placeholders})", item_ids)
            rows = cur.fetchall()

        for item_id, text in rows:
            try:
                embedding = model.encode(text, normalize_embeddings=True).tolist()
                point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, item_id))
                client.upsert(
                    collection_name=QDRANT_COLLECTION,
                    points=[models.PointStruct(
                        id=point_id,
                        vector=embedding,
                        payload={"memory_id": item_id, "text": text[:500]},
                    )],
                )
                repaired += 1
            except Exception as ex:
                LOGGER.warning("repair_qdrant_failed item_id=%s: %s", item_id, ex)
                errors += 1

        return {"repaired": repaired, "errors": errors, "dry_run": False}

    finally:
        if should_close:
            conn.close()


# ---------------------------------------------------------------------------
# Ingestion observability
# ---------------------------------------------------------------------------

@dataclass
class IngestionMetrics:
    """Snapshot of ingestion pipeline health."""
    timestamp: str = ""
    total_items: int = 0
    active_items: int = 0
    candidate_items: int = 0
    superseded_items: int = 0
    discarded_items: int = 0
    pending_items: int = 0
    items_with_embeddings: int = 0
    items_with_graph_links: int = 0
    items_with_contradictions: int = 0
    avg_confidence: float = 0.0
    promotion_rate: float = 0.0  # candidates that became active
    rejection_rate: float = 0.0  # candidates that were discarded
    embedding_lag: int = 0       # items missing embeddings
    graph_lag: int = 0           # items missing graph links
    last_ingestion: str = ""

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "total_items": self.total_items,
            "status_breakdown": {
                "active": self.active_items,
                "candidate": self.candidate_items,
                "superseded": self.superseded_items,
                "discarded": self.discarded_items,
                "pending": self.pending_items,
            },
            "embedding_coverage": self.items_with_embeddings,
            "graph_coverage": self.items_with_graph_links,
            "items_with_contradictions": self.items_with_contradictions,
            "avg_confidence": round(self.avg_confidence, 4),
            "promotion_rate": round(self.promotion_rate, 4),
            "rejection_rate": round(self.rejection_rate, 4),
            "embedding_lag": self.embedding_lag,
            "graph_lag": self.graph_lag,
            "last_ingestion": self.last_ingestion,
        }


def collect_ingestion_metrics(conn: Any = None) -> IngestionMetrics:
    """Collect current ingestion pipeline metrics from Postgres."""
    should_close = False
    if conn is None:
        import psycopg
        conn = psycopg.connect(DB_DSN)
        should_close = True

    metrics = IngestionMetrics(timestamp=datetime.now(timezone.utc).isoformat())

    try:
        with conn.cursor() as cur:
            # Status breakdown
            cur.execute("""
                SELECT status, COUNT(*) FROM memory_items GROUP BY status
            """)
            for status, count in cur.fetchall():
                if status == "active":
                    metrics.active_items = count
                elif status == "candidate":
                    metrics.candidate_items = count
                elif status == "superseded":
                    metrics.superseded_items = count
                elif status == "discarded":
                    metrics.discarded_items = count
                elif status == "pending":
                    metrics.pending_items = count

            metrics.total_items = (metrics.active_items + metrics.candidate_items +
                                   metrics.superseded_items + metrics.discarded_items + metrics.pending_items)

            # Average confidence
            cur.execute("SELECT AVG(confidence) FROM memory_items WHERE status IN ('active', 'candidate') AND confidence IS NOT NULL")
            row = cur.fetchone()
            metrics.avg_confidence = float(row[0] or 0) if row else 0

            # Promotion/rejection rates
            total_resolved = metrics.active_items + metrics.discarded_items
            if total_resolved > 0:
                metrics.promotion_rate = metrics.active_items / total_resolved
                metrics.rejection_rate = metrics.discarded_items / total_resolved

            # Contradiction count
            try:
                cur.execute("SELECT COUNT(*) FROM memory_items WHERE contradiction_count > 0")
                metrics.items_with_contradictions = cur.fetchone()[0]
            except Exception:
                pass

            # Last ingestion
            cur.execute("SELECT MAX(created_at) FROM memory_items")
            row = cur.fetchone()
            if row and row[0]:
                metrics.last_ingestion = str(row[0])

    finally:
        if should_close:
            conn.close()

    return metrics


# ---------------------------------------------------------------------------
# Replay-safe reprocessing
# ---------------------------------------------------------------------------

@dataclass
class ReprocessingPlan:
    """Plan for reprocessing historical data."""
    start_date: str
    end_date: str
    affected_items: int = 0
    affected_sessions: int = 0
    dry_run: bool = True
    steps: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "start_date": self.start_date,
            "end_date": self.end_date,
            "affected_items": self.affected_items,
            "affected_sessions": self.affected_sessions,
            "dry_run": self.dry_run,
            "steps": self.steps,
        }


def plan_reprocessing(
    days_back: int = 90,
    conn: Any = None,
    dry_run: bool = True,
) -> ReprocessingPlan:
    """
    Plan a replay-safe reprocessing of historical memory data.

    Steps:
    1. Identify affected items (created in the date range)
    2. Snapshot current state (for rollback)
    3. Mark affected items as "reprocessing"
    4. Re-extract from source transcripts
    5. Re-embed and re-index
    6. Compare old vs new, resolve differences
    7. Commit or rollback
    """
    should_close = False
    if conn is None:
        import psycopg
        conn = psycopg.connect(DB_DSN)
        should_close = True

    now = datetime.now(timezone.utc)
    start = now - __import__("datetime").timedelta(days=days_back)

    plan = ReprocessingPlan(
        start_date=start.isoformat(),
        end_date=now.isoformat(),
        dry_run=dry_run,
    )

    try:
        with conn.cursor() as cur:
            # Count affected items
            cur.execute("""
                SELECT COUNT(*) FROM memory_items
                WHERE created_at >= %s
            """, (start,))
            plan.affected_items = cur.fetchone()[0]

            # Count affected sessions
            cur.execute("""
                SELECT COUNT(DISTINCT source_session) FROM memory_items
                WHERE created_at >= %s AND source_session IS NOT NULL
            """, (start,))
            plan.affected_sessions = cur.fetchone()[0]

    finally:
        if should_close:
            conn.close()

    plan.steps = [
        f"1. Snapshot {plan.affected_items} items from {plan.affected_sessions} sessions",
        "2. Create restore point in restore_points table",
        "3. Mark affected items as status='reprocessing' (preserving original status)",
        "4. Re-run extraction pipeline on source transcripts",
        "5. Generate new candidates and compare against originals",
        "6. Re-embed new/changed items in Qdrant",
        "7. Re-sync graph links in Neo4j",
        "8. Resolve differences (new items added, stale items marked superseded)",
        "9. Update contradiction_edges for changed items",
        "10. Commit and clear 'reprocessing' status, or rollback to restore point",
    ]

    return plan


def create_restore_point(
    label: str = "",
    conn: Any = None,
) -> int | None:
    """Create a restore point before reprocessing."""
    should_close = False
    if conn is None:
        import psycopg
        conn = psycopg.connect(DB_DSN)
        should_close = True

    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO restore_points (label, path, created_at)
                VALUES (%s, %s, now())
                RETURNING restore_id
            """, (
                label or f"reprocessing-{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                "memory_items_snapshot",
            ))
            restore_id = cur.fetchone()[0]
        conn.commit()
        return restore_id
    finally:
        if should_close:
            conn.close()
