"""
contradiction_manager.py — First-class contradiction management system.

Phase 2, Item 6: Add contradiction management as a first-class system.

Not just "supersedes" — full contradiction handling with:
1. Semantic conflict detection (beyond keyword opposites)
2. Active invalidation of stale truths
3. Contradiction tagging in retrieval results
4. "Current truth vs historical truth" separation
5. Safe retrieval of changed beliefs

Design principles:
- Contradictions are tracked as edges (item_a contradicts item_b)
- Each contradiction has a resolution state
- Retrieval can surface contradictions as context
- No LLM calls for detection — deterministic + embedding similarity
- LLM-assisted resolution is optional (separate step)

DB schema additions (applied via migrate):
- contradiction_edges table
- contradiction state on memory_items
"""
from __future__ import annotations

import hashlib
import logging
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

LOGGER = logging.getLogger("openclaw.contradiction_manager")

DB_DSN = os.environ.get("OPENCLAW_MEMORY_DSN", "dbname=openclaw_memory")


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

MIGRATION_SQL = """
-- Contradiction edges table
CREATE TABLE IF NOT EXISTS contradiction_edges (
    edge_id BIGSERIAL PRIMARY KEY,
    item_a_id TEXT NOT NULL,
    item_b_id TEXT NOT NULL,
    detection_method TEXT NOT NULL,        -- 'slot_conflict', 'semantic', 'opposite_keywords', 'manual'
    confidence DOUBLE PRECISION DEFAULT 0.5,
    resolution_state TEXT NOT NULL DEFAULT 'unresolved',  -- 'unresolved', 'a_wins', 'b_wins', 'both_valid', 'both_invalid', 'merged'
    resolution_reason TEXT,
    resolved_by TEXT,                      -- agent or user who resolved
    resolved_at TIMESTAMPTZ,
    detected_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now(),
    UNIQUE(item_a_id, item_b_id)
);

CREATE INDEX IF NOT EXISTS idx_contradiction_edges_item_a ON contradiction_edges(item_a_id);
CREATE INDEX IF NOT EXISTS idx_contradiction_edges_item_b ON contradiction_edges(item_b_id);
CREATE INDEX IF NOT EXISTS idx_contradiction_edges_state ON contradiction_edges(resolution_state);

-- Add contradiction_count to memory_items if not present
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'memory_items' AND column_name = 'contradiction_count'
    ) THEN
        ALTER TABLE memory_items ADD COLUMN contradiction_count INT DEFAULT 0;
    END IF;
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'memory_items' AND column_name = 'is_current_truth'
    ) THEN
        ALTER TABLE memory_items ADD COLUMN is_current_truth BOOLEAN DEFAULT TRUE;
    END IF;
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'memory_items' AND column_name = 'superseded_reason'
    ) THEN
        ALTER TABLE memory_items ADD COLUMN superseded_reason TEXT;
    END IF;
END $$;
"""


class ResolutionState(str, Enum):
    UNRESOLVED = "unresolved"
    A_WINS = "a_wins"
    B_WINS = "b_wins"
    BOTH_VALID = "both_valid"       # e.g., scoped to different contexts
    BOTH_INVALID = "both_invalid"   # e.g., both outdated
    MERGED = "merged"               # Combined into a new item


class DetectionMethod(str, Enum):
    SLOT_CONFLICT = "slot_conflict"
    SEMANTIC = "semantic"
    OPPOSITE_KEYWORDS = "opposite_keywords"
    MANUAL = "manual"


@dataclass
class ContradictionEdge:
    """A contradiction between two memory items."""
    item_a_id: str
    item_b_id: str
    detection_method: DetectionMethod
    confidence: float = 0.5
    resolution_state: ResolutionState = ResolutionState.UNRESOLVED
    resolution_reason: str | None = None
    resolved_by: str | None = None
    item_a_text: str = ""
    item_b_text: str = ""
    edge_id: int | None = None

    def to_dict(self) -> dict:
        return {
            "edge_id": self.edge_id,
            "item_a_id": self.item_a_id,
            "item_b_id": self.item_b_id,
            "item_a_text": self.item_a_text[:200],
            "item_b_text": self.item_b_text[:200],
            "detection_method": self.detection_method.value,
            "confidence": round(self.confidence, 3),
            "resolution_state": self.resolution_state.value,
            "resolution_reason": self.resolution_reason,
        }


# ---------------------------------------------------------------------------
# Migration
# ---------------------------------------------------------------------------

def apply_migration(conn: Any = None):
    """Apply the contradiction_edges schema to the database."""
    should_close = False
    if conn is None:
        import psycopg
        conn = psycopg.connect(DB_DSN)
        should_close = True

    try:
        with conn.cursor() as cur:
            cur.execute(MIGRATION_SQL)
        conn.commit()
        LOGGER.info("contradiction_edges migration applied")
    finally:
        if should_close:
            conn.close()


# ---------------------------------------------------------------------------
# Detection: Slot conflicts
# ---------------------------------------------------------------------------

def detect_slot_conflicts(conn: Any = None) -> list[ContradictionEdge]:
    """
    Find items that occupy the same (entity, property, scope) slot but have
    different values. These are direct contradictions.
    """
    should_close = False
    if conn is None:
        import psycopg
        conn = psycopg.connect(DB_DSN)
        should_close = True

    edges = []
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT a.id, b.id, a.text, b.text, a.value, b.value,
                       a.entity, a.property, a.scope
                FROM memory_items a
                JOIN memory_items b
                  ON a.entity = b.entity
                 AND a.property = b.property
                 AND COALESCE(a.scope, '') = COALESCE(b.scope, '')
                 AND a.id < b.id
                WHERE a.status IN ('active', 'candidate')
                  AND b.status IN ('active', 'candidate')
                  AND a.value IS DISTINCT FROM b.value
                  AND a.entity IS NOT NULL
                  AND a.property IS NOT NULL
            """)

            for a_id, b_id, a_text, b_text, a_val, b_val, entity, prop, scope in cur.fetchall():
                edges.append(ContradictionEdge(
                    item_a_id=a_id,
                    item_b_id=b_id,
                    detection_method=DetectionMethod.SLOT_CONFLICT,
                    confidence=0.9,  # High confidence — same slot, different value
                    item_a_text=a_text or "",
                    item_b_text=b_text or "",
                ))

    finally:
        if should_close:
            conn.close()

    return edges


# ---------------------------------------------------------------------------
# Detection: Opposite keyword patterns
# ---------------------------------------------------------------------------

OPPOSITE_PAIRS = [
    ("use ", "avoid "),
    ("requires ", "does not require "),
    ("enabled", "disabled"),
    ("available", "unavailable"),
    ("supported", "not supported"),
    ("should", "should not"),
    ("must", "must not"),
    ("always", "never"),
    ("true", "false"),
    ("yes", "no"),
    ("can", "cannot"),
    ("allowed", "not allowed"),
    ("recommended", "not recommended"),
    ("deprecated", "active"),
    ("removed", "added"),
]


def detect_keyword_contradictions(
    items: list[dict] | None = None,
    conn: Any = None,
    limit: int = 500,
) -> list[ContradictionEdge]:
    """
    Find items with opposite keyword patterns that share the same entity.
    """
    should_close = False
    if items is None:
        if conn is None:
            import psycopg
            conn = psycopg.connect(DB_DSN)
            should_close = True

        try:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT id, text, entity, property, scope
                    FROM memory_items
                    WHERE status IN ('active', 'candidate')
                      AND text IS NOT NULL AND text != ''
                    ORDER BY updated_at DESC
                    LIMIT %s
                """, (limit,))
                items = [
                    {"id": r[0], "text": r[1], "entity": r[2], "property": r[3], "scope": r[4]}
                    for r in cur.fetchall()
                ]
        finally:
            if should_close:
                conn.close()

    edges = []
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            a = items[i]
            b = items[j]

            # Must share entity for keyword contradiction to be meaningful
            if a.get("entity") and b.get("entity") and a["entity"] == b["entity"]:
                a_text = (a.get("text") or "").lower()
                b_text = (b.get("text") or "").lower()

                for pos, neg in OPPOSITE_PAIRS:
                    if (pos in a_text and neg in b_text) or (neg in a_text and pos in b_text):
                        edges.append(ContradictionEdge(
                            item_a_id=a["id"],
                            item_b_id=b["id"],
                            detection_method=DetectionMethod.OPPOSITE_KEYWORDS,
                            confidence=0.6,
                            item_a_text=a.get("text", ""),
                            item_b_text=b.get("text", ""),
                        ))
                        break  # One match per pair is enough

    return edges


# ---------------------------------------------------------------------------
# Detection: Semantic similarity with value disagreement
# ---------------------------------------------------------------------------

def detect_semantic_contradictions(
    conn: Any = None,
    similarity_threshold: float = 0.85,
    limit: int = 200,
) -> list[ContradictionEdge]:
    """
    Find items with high text similarity but different values.
    Uses pgvector if embeddings exist, otherwise falls back to text overlap.

    Items that are semantically similar (talking about the same thing) but
    have different values are likely contradictions.
    """
    should_close = False
    if conn is None:
        import psycopg
        conn = psycopg.connect(DB_DSN)
        should_close = True

    edges = []
    try:
        with conn.cursor() as cur:
            # Check if we have a vector column available
            cur.execute("""
                SELECT EXISTS (
                    SELECT 1 FROM information_schema.columns
                    WHERE table_name = 'memory_items' AND column_name = 'embedding'
                )
            """)
            has_embeddings = cur.fetchone()[0]

            if has_embeddings:
                # Use vector similarity
                cur.execute("""
                    SELECT a.id, b.id, a.text, b.text, a.value, b.value,
                           1 - (a.embedding <=> b.embedding) AS similarity
                    FROM memory_items a
                    JOIN memory_items b ON a.id < b.id
                    WHERE a.status IN ('active', 'candidate')
                      AND b.status IN ('active', 'candidate')
                      AND a.embedding IS NOT NULL
                      AND b.embedding IS NOT NULL
                      AND a.value IS DISTINCT FROM b.value
                      AND a.entity = b.entity
                      AND 1 - (a.embedding <=> b.embedding) > %s
                    ORDER BY similarity DESC
                    LIMIT %s
                """, (similarity_threshold, limit))
            else:
                # Fallback: check if pg_trgm is available
                cur.execute("SELECT EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'pg_trgm')")
                has_trgm = cur.fetchone()[0]

                if has_trgm:
                    cur.execute("""
                        SELECT a.id, b.id, a.text, b.text, a.value, b.value,
                               similarity(LOWER(a.text), LOWER(b.text)) AS sim
                        FROM memory_items a
                        JOIN memory_items b ON a.id < b.id
                        WHERE a.status IN ('active', 'candidate')
                          AND b.status IN ('active', 'candidate')
                          AND a.value IS DISTINCT FROM b.value
                          AND a.entity = b.entity
                          AND a.entity IS NOT NULL
                          AND similarity(LOWER(a.text), LOWER(b.text)) > %s
                        ORDER BY sim DESC
                        LIMIT %s
                    """, (similarity_threshold, limit))
                else:
                    # No vector embeddings, no pg_trgm — skip semantic detection
                    LOGGER.info("semantic_detection_skipped: no embeddings column and no pg_trgm extension")
                    return edges  # Return empty, no fallback available

            for a_id, b_id, a_text, b_text, a_val, b_val, sim in cur.fetchall():
                edges.append(ContradictionEdge(
                    item_a_id=a_id,
                    item_b_id=b_id,
                    detection_method=DetectionMethod.SEMANTIC,
                    confidence=min(0.95, float(sim)),
                    item_a_text=a_text or "",
                    item_b_text=b_text or "",
                ))

    except Exception as ex:
        LOGGER.warning("semantic_contradiction_detection_failed: %s", ex)
        try:
            conn.rollback()
        except Exception:
            pass
    finally:
        if should_close:
            conn.close()

    return edges


# ---------------------------------------------------------------------------
# Store contradictions
# ---------------------------------------------------------------------------

def store_edges(edges: list[ContradictionEdge], conn: Any = None) -> int:
    """Insert detected contradiction edges into the database. Returns count inserted."""
    if not edges:
        return 0

    should_close = False
    if conn is None:
        import psycopg
        conn = psycopg.connect(DB_DSN)
        should_close = True

    inserted = 0
    try:
        with conn.cursor() as cur:
            for edge in edges:
                # Normalize order (smaller id first)
                a_id, b_id = sorted([edge.item_a_id, edge.item_b_id])
                cur.execute("""
                    INSERT INTO contradiction_edges (item_a_id, item_b_id, detection_method, confidence)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (item_a_id, item_b_id) DO UPDATE
                    SET confidence = GREATEST(contradiction_edges.confidence, EXCLUDED.confidence),
                        updated_at = now()
                    RETURNING edge_id
                """, (a_id, b_id, edge.detection_method.value, edge.confidence))
                if cur.fetchone():
                    inserted += 1

            # Update contradiction_count on affected items
            cur.execute("""
                UPDATE memory_items SET contradiction_count = sub.cnt
                FROM (
                    SELECT item_id, COUNT(*) AS cnt FROM (
                        SELECT item_a_id AS item_id FROM contradiction_edges WHERE resolution_state = 'unresolved'
                        UNION ALL
                        SELECT item_b_id AS item_id FROM contradiction_edges WHERE resolution_state = 'unresolved'
                    ) x
                    GROUP BY item_id
                ) sub
                WHERE memory_items.id = sub.item_id
            """)

        conn.commit()
    finally:
        if should_close:
            conn.close()

    return inserted


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------

def resolve_contradiction(
    edge_id: int | None = None,
    item_a_id: str | None = None,
    item_b_id: str | None = None,
    resolution: ResolutionState = ResolutionState.A_WINS,
    reason: str = "",
    resolved_by: str = "system",
    conn: Any = None,
) -> bool:
    """
    Resolve a contradiction edge and update the losing item(s).

    Resolution effects:
    - a_wins: b gets is_current_truth=False, superseded_reason set
    - b_wins: a gets is_current_truth=False, superseded_reason set
    - both_valid: no status changes (scoped to different contexts)
    - both_invalid: both get is_current_truth=False
    - merged: both get is_current_truth=False (new merged item should exist)
    """
    should_close = False
    if conn is None:
        import psycopg
        conn = psycopg.connect(DB_DSN)
        should_close = True

    try:
        with conn.cursor() as cur:
            # Find the edge
            if edge_id:
                cur.execute("SELECT edge_id, item_a_id, item_b_id FROM contradiction_edges WHERE edge_id = %s", (edge_id,))
            elif item_a_id and item_b_id:
                a, b = sorted([item_a_id, item_b_id])
                cur.execute("SELECT edge_id, item_a_id, item_b_id FROM contradiction_edges WHERE item_a_id = %s AND item_b_id = %s", (a, b))
            else:
                return False

            row = cur.fetchone()
            if not row:
                return False

            eid, a_id, b_id = row

            # Update edge
            cur.execute("""
                UPDATE contradiction_edges
                SET resolution_state = %s, resolution_reason = %s, resolved_by = %s, resolved_at = now(), updated_at = now()
                WHERE edge_id = %s
            """, (resolution.value, reason, resolved_by, eid))

            # Apply resolution effects
            now_str = datetime.now(timezone.utc).isoformat()

            if resolution == ResolutionState.A_WINS:
                cur.execute("""
                    UPDATE memory_items SET is_current_truth = FALSE, superseded_reason = %s, updated_at = now()
                    WHERE id = %s
                """, (f"Contradicted by {a_id}: {reason}", b_id))

            elif resolution == ResolutionState.B_WINS:
                cur.execute("""
                    UPDATE memory_items SET is_current_truth = FALSE, superseded_reason = %s, updated_at = now()
                    WHERE id = %s
                """, (f"Contradicted by {b_id}: {reason}", a_id))

            elif resolution == ResolutionState.BOTH_INVALID:
                for item_id in [a_id, b_id]:
                    cur.execute("""
                        UPDATE memory_items SET is_current_truth = FALSE, superseded_reason = %s, updated_at = now()
                        WHERE id = %s
                    """, (f"Both sides invalidated: {reason}", item_id))

            elif resolution == ResolutionState.MERGED:
                for item_id in [a_id, b_id]:
                    cur.execute("""
                        UPDATE memory_items SET is_current_truth = FALSE, superseded_reason = %s, updated_at = now()
                        WHERE id = %s
                    """, (f"Merged into new item: {reason}", item_id))

            # both_valid: no changes needed

            # Recount contradictions
            cur.execute("""
                UPDATE memory_items SET contradiction_count = COALESCE(sub.cnt, 0)
                FROM (
                    SELECT item_id, COUNT(*) AS cnt FROM (
                        SELECT item_a_id AS item_id FROM contradiction_edges WHERE resolution_state = 'unresolved'
                        UNION ALL
                        SELECT item_b_id AS item_id FROM contradiction_edges WHERE resolution_state = 'unresolved'
                    ) x
                    GROUP BY item_id
                ) sub
                WHERE memory_items.id = sub.item_id
                  AND memory_items.id IN (%s, %s)
            """, (a_id, b_id))

        conn.commit()
        return True

    finally:
        if should_close:
            conn.close()


# ---------------------------------------------------------------------------
# Retrieval integration
# ---------------------------------------------------------------------------

def get_contradictions_for_items(
    item_ids: list[str],
    include_resolved: bool = False,
    conn: Any = None,
) -> dict[str, list[ContradictionEdge]]:
    """
    For a set of retrieved items, find any active contradictions.
    Returns {item_id: [ContradictionEdge, ...]}.
    Used to tag retrieval results with contradiction context.
    """
    if not item_ids:
        return {}

    should_close = False
    if conn is None:
        import psycopg
        conn = psycopg.connect(DB_DSN)
        should_close = True

    result: dict[str, list[ContradictionEdge]] = {iid: [] for iid in item_ids}

    try:
        with conn.cursor() as cur:
            placeholders = ",".join(["%s"] * len(item_ids))
            state_filter = "" if include_resolved else "AND ce.resolution_state = 'unresolved'"

            cur.execute(f"""
                SELECT ce.edge_id, ce.item_a_id, ce.item_b_id,
                       ce.detection_method, ce.confidence, ce.resolution_state,
                       ce.resolution_reason,
                       a.text AS a_text, b.text AS b_text
                FROM contradiction_edges ce
                JOIN memory_items a ON a.id = ce.item_a_id
                JOIN memory_items b ON b.id = ce.item_b_id
                WHERE (ce.item_a_id IN ({placeholders}) OR ce.item_b_id IN ({placeholders}))
                {state_filter}
            """, item_ids + item_ids)

            for row in cur.fetchall():
                edge = ContradictionEdge(
                    item_a_id=row[1],
                    item_b_id=row[2],
                    detection_method=DetectionMethod(row[3]),
                    confidence=float(row[4]),
                    resolution_state=ResolutionState(row[5]),
                    resolution_reason=row[6],
                    item_a_text=row[7] or "",
                    item_b_text=row[8] or "",
                    edge_id=row[0],
                )

                if row[1] in result:
                    result[row[1]].append(edge)
                if row[2] in result:
                    result[row[2]].append(edge)

    finally:
        if should_close:
            conn.close()

    return result


# ---------------------------------------------------------------------------
# Full detection sweep
# ---------------------------------------------------------------------------

def run_detection_sweep(conn: Any = None) -> dict:
    """
    Run all detection methods and store new contradiction edges.

    Returns summary of findings.
    """
    should_close = False
    if conn is None:
        import psycopg
        conn = psycopg.connect(DB_DSN)
        should_close = True

    try:
        # Ensure schema exists
        apply_migration(conn)

        all_edges: list[ContradictionEdge] = []

        # Method 1: Slot conflicts
        try:
            slot_edges = detect_slot_conflicts(conn)
            all_edges.extend(slot_edges)
        except Exception as ex:
            LOGGER.warning("slot_conflict_detection_failed: %s", ex)
            slot_edges = []
            try:
                conn.rollback()
            except Exception:
                pass

        # Method 2: Keyword opposites
        try:
            keyword_edges = detect_keyword_contradictions(conn=conn)
            all_edges.extend(keyword_edges)
        except Exception as ex:
            LOGGER.warning("keyword_contradiction_detection_failed: %s", ex)
            keyword_edges = []
            try:
                conn.rollback()
            except Exception:
                pass

        # Method 3: Semantic similarity (uses its own error handling)
        semantic_edges = detect_semantic_contradictions(conn=conn)
        all_edges.extend(semantic_edges)

        # Deduplicate by (a_id, b_id)
        seen = set()
        unique_edges = []
        for edge in all_edges:
            key = tuple(sorted([edge.item_a_id, edge.item_b_id]))
            if key not in seen:
                seen.add(key)
                unique_edges.append(edge)

        # Store
        inserted = store_edges(unique_edges, conn)

        return {
            "slot_conflicts": len(slot_edges),
            "keyword_contradictions": len(keyword_edges),
            "semantic_contradictions": len(semantic_edges),
            "total_unique": len(unique_edges),
            "inserted": inserted,
        }

    finally:
        if should_close:
            conn.close()


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def get_contradiction_summary(conn: Any = None) -> dict:
    """Get a summary of all contradictions in the system."""
    should_close = False
    if conn is None:
        import psycopg
        conn = psycopg.connect(DB_DSN)
        should_close = True

    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT resolution_state, COUNT(*) FROM contradiction_edges
                GROUP BY resolution_state
            """)
            by_state = dict(cur.fetchall())

            cur.execute("SELECT COUNT(*) FROM contradiction_edges")
            total = cur.fetchone()[0]

            cur.execute("""
                SELECT detection_method, COUNT(*) FROM contradiction_edges
                WHERE resolution_state = 'unresolved'
                GROUP BY detection_method
            """)
            unresolved_by_method = dict(cur.fetchall())

            cur.execute("""
                SELECT COUNT(DISTINCT id) FROM memory_items WHERE contradiction_count > 0
            """)
            items_with_contradictions = cur.fetchone()[0]

            return {
                "total_edges": total,
                "by_state": by_state,
                "unresolved_by_method": unresolved_by_method,
                "items_with_contradictions": items_with_contradictions,
            }

    finally:
        if should_close:
            conn.close()
