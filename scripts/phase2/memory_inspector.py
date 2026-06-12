"""
memory_inspector.py — Operator trust tooling: inspect, audit, and score memory.

Phase 8, Items 22-24:
22. Memory inspector CLI (item detail, retrieval history, provenance, links)
23. Memory quality audits (duplicate rate, stale count, contradictions, orphans)
24. Memory trust score per item

Provides a single entry point for operators to understand and trust the
memory system's state.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Any

LOGGER = logging.getLogger("openclaw.memory_inspector")

DB_DSN = os.environ.get("OPENCLAW_MEMORY_DSN", "dbname=openclaw_memory")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)


# ---------------------------------------------------------------------------
# Trust score computation
# ---------------------------------------------------------------------------

@dataclass
class TrustScore:
    """Per-item trust score combining multiple signals."""
    item_id: str
    overall: float
    components: dict[str, float] = field(default_factory=dict)
    flags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "item_id": self.item_id,
            "overall": round(self.overall, 4),
            "components": {k: round(v, 4) for k, v in self.components.items()},
            "flags": self.flags,
            "tier": self._tier(),
        }

    def _tier(self) -> str:
        if self.overall >= 0.85:
            return "high_trust"
        if self.overall >= 0.60:
            return "moderate_trust"
        if self.overall >= 0.35:
            return "low_trust"
        return "untrusted"


def compute_trust_score(item: dict) -> TrustScore:
    """
    Compute a composite trust score for a memory item.

    Components:
    - source_trust: based on authority_basis and role
    - recency_trust: how recently confirmed
    - confirmation_trust: mention count / corroboration
    - contradiction_trust: penalized for active contradictions
    - confidence_trust: the item's own confidence score
    - user_confirmed: bonus for user-confirmed items
    """
    item_id = str(item.get("id", ""))
    components = {}
    flags = []

    # 1. Source trust (use authority_basis if available, fall back to source_quality or source_agent)
    authority = str(item.get("authority_basis", "") or item.get("source_quality", "") or "unknown").lower()
    source_map = {
        "user_explicit": 1.0,
        "developer_explicit": 0.95,
        "tester_explicit": 0.90,
        "system_tool_verified": 0.85,
        "high": 0.85,
        "medium": 0.65,
        "low": 0.45,
        "assistant_inferred": 0.50,
        "summary_inferred": 0.35,
        "unknown": 0.30,
    }
    components["source_trust"] = source_map.get(authority, 0.30)

    # 2. Recency trust
    last_confirmed = item.get("last_confirmed") or item.get("created_at")
    if last_confirmed:
        try:
            if isinstance(last_confirmed, str):
                last_confirmed = datetime.fromisoformat(str(last_confirmed).replace("Z", "+00:00"))
            now = datetime.now(timezone.utc)
            if last_confirmed.tzinfo is None:
                last_confirmed = last_confirmed.replace(tzinfo=timezone.utc)
            days_ago = (now - last_confirmed).total_seconds() / 86400
            # Gentle decay
            import math
            components["recency_trust"] = 1.0 / (1.0 + math.log(1.0 + days_ago / 30))
        except (ValueError, TypeError):
            components["recency_trust"] = 0.5
    else:
        components["recency_trust"] = 0.3
        flags.append("no_timestamp")

    # 3. Confirmation trust (mention count)
    mention_count = int(item.get("mention_count", 1) or 1)
    import math
    components["confirmation_trust"] = min(1.0, 0.4 + 0.2 * math.log(1 + mention_count))

    # 4. Contradiction trust
    contradiction_count = int(item.get("contradiction_count", 0) or 0)
    if contradiction_count == 0:
        components["contradiction_trust"] = 1.0
    elif contradiction_count == 1:
        components["contradiction_trust"] = 0.5
        flags.append("has_contradiction")
    else:
        components["contradiction_trust"] = max(0.1, 0.5 - 0.15 * (contradiction_count - 1))
        flags.append(f"multiple_contradictions ({contradiction_count})")

    # 5. Confidence trust
    confidence = float(item.get("confidence", 0.5) or 0.5)
    components["confidence_trust"] = confidence

    # 6. User confirmed bonus
    is_current_truth = item.get("is_current_truth", True)
    if not is_current_truth:
        components["current_truth"] = 0.0
        flags.append("not_current_truth")
    else:
        components["current_truth"] = 1.0

    # Weighted combination
    weights = {
        "source_trust": 0.25,
        "recency_trust": 0.15,
        "confirmation_trust": 0.15,
        "contradiction_trust": 0.20,
        "confidence_trust": 0.15,
        "current_truth": 0.10,
    }

    overall = sum(components.get(k, 0) * w for k, w in weights.items())

    return TrustScore(item_id=item_id, overall=overall, components=components, flags=flags)


# ---------------------------------------------------------------------------
# Quality audit
# ---------------------------------------------------------------------------

@dataclass
class QualityAudit:
    """Periodic quality audit of the memory system."""
    timestamp: str = ""
    total_items: int = 0
    active_items: int = 0

    # Duplicate analysis
    duplicate_count: int = 0
    duplicate_rate: float = 0.0

    # Staleness
    stale_durable_count: int = 0    # Active items not confirmed in 60+ days
    stale_rate: float = 0.0

    # Contradictions
    unresolved_contradictions: int = 0
    items_with_contradictions: int = 0

    # Low confidence
    low_confidence_durable: int = 0  # Active items with confidence < 0.5
    low_confidence_rate: float = 0.0

    # Trust distribution
    trust_high: int = 0
    trust_moderate: int = 0
    trust_low: int = 0
    trust_untrusted: int = 0

    # Provenance
    items_without_source: int = 0
    items_without_source_rate: float = 0.0

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "total_items": self.total_items,
            "active_items": self.active_items,
            "duplicates": {"count": self.duplicate_count, "rate": round(self.duplicate_rate, 4)},
            "staleness": {"stale_count": self.stale_durable_count, "rate": round(self.stale_rate, 4)},
            "contradictions": {"unresolved": self.unresolved_contradictions, "items_affected": self.items_with_contradictions},
            "low_confidence": {"count": self.low_confidence_durable, "rate": round(self.low_confidence_rate, 4)},
            "trust_distribution": {
                "high": self.trust_high,
                "moderate": self.trust_moderate,
                "low": self.trust_low,
                "untrusted": self.trust_untrusted,
            },
            "provenance": {"missing_source": self.items_without_source, "rate": round(self.items_without_source_rate, 4)},
        }


def run_quality_audit(conn: Any = None) -> QualityAudit:
    """Run a comprehensive quality audit of the memory system."""
    should_close = False
    if conn is None:
        import psycopg
        conn = psycopg.connect(DB_DSN)
        should_close = True

    audit = QualityAudit(timestamp=datetime.now(timezone.utc).isoformat())

    try:
        with conn.cursor() as cur:
            # Total and active counts
            cur.execute("SELECT COUNT(*) FROM memory_items")
            audit.total_items = cur.fetchone()[0]

            cur.execute("SELECT COUNT(*) FROM memory_items WHERE status = 'active'")
            audit.active_items = cur.fetchone()[0]

            if audit.active_items == 0:
                return audit

            # Duplicates (same entity+property+value, different IDs)
            cur.execute("""
                SELECT COUNT(*) FROM (
                    SELECT entity, property, value, COUNT(*) AS cnt
                    FROM memory_items
                    WHERE status = 'active' AND entity IS NOT NULL AND property IS NOT NULL
                    GROUP BY entity, property, value
                    HAVING COUNT(*) > 1
                ) dups
            """)
            audit.duplicate_count = cur.fetchone()[0]
            audit.duplicate_rate = audit.duplicate_count / max(audit.active_items, 1)

            # Stale items (not confirmed in 60+ days)
            stale_cutoff = datetime.now(timezone.utc) - timedelta(days=60)
            cur.execute("""
                SELECT COUNT(*) FROM memory_items
                WHERE status = 'active'
                  AND (last_confirmed IS NULL OR last_confirmed < %s)
                  AND (created_at IS NULL OR created_at < %s)
            """, (stale_cutoff, stale_cutoff))
            audit.stale_durable_count = cur.fetchone()[0]
            audit.stale_rate = audit.stale_durable_count / max(audit.active_items, 1)

            # Contradictions
            try:
                cur.execute("SELECT COUNT(*) FROM contradiction_edges WHERE resolution_state = 'unresolved'")
                audit.unresolved_contradictions = cur.fetchone()[0]
                cur.execute("SELECT COUNT(DISTINCT id) FROM memory_items WHERE contradiction_count > 0")
                audit.items_with_contradictions = cur.fetchone()[0]
            except Exception:
                pass

            # Low confidence
            cur.execute("SELECT COUNT(*) FROM memory_items WHERE status = 'active' AND confidence < 0.5")
            audit.low_confidence_durable = cur.fetchone()[0]
            audit.low_confidence_rate = audit.low_confidence_durable / max(audit.active_items, 1)

            # Missing source provenance
            cur.execute("""
                SELECT COUNT(*) FROM memory_items
                WHERE status = 'active'
                  AND (source_session IS NULL OR source_session = '' OR source_session = 'unknown')
            """)
            audit.items_without_source = cur.fetchone()[0]
            audit.items_without_source_rate = audit.items_without_source / max(audit.active_items, 1)

            # Trust distribution (sample 500 items for speed)
            cur.execute("""
                SELECT id, confidence, source_quality, created_at, last_confirmed,
                       COALESCE(contradiction_count, 0), COALESCE(is_current_truth, TRUE)
                FROM memory_items
                WHERE status = 'active'
                ORDER BY RANDOM()
                LIMIT 500
            """)

            for row in cur.fetchall():
                item = {
                    "id": row[0], "confidence": row[1], "authority_basis": row[2],
                    "created_at": row[3], "last_confirmed": row[4],
                    "contradiction_count": row[5], "is_current_truth": row[6],
                }
                score = compute_trust_score(item)
                tier = score._tier()
                if tier == "high_trust":
                    audit.trust_high += 1
                elif tier == "moderate_trust":
                    audit.trust_moderate += 1
                elif tier == "low_trust":
                    audit.trust_low += 1
                else:
                    audit.trust_untrusted += 1

    finally:
        if should_close:
            conn.close()

    return audit


# ---------------------------------------------------------------------------
# Item inspector
# ---------------------------------------------------------------------------

def inspect_item(item_id: str, conn: Any = None) -> dict:
    """
    Full inspection of a single memory item.
    Combines: item data, trust score, provenance, contradictions, retrieval history.
    """
    should_close = False
    if conn is None:
        import psycopg
        conn = psycopg.connect(DB_DSN)
        should_close = True

    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT id, text, memory_type, entity, property, value, scope, status,
                       confidence, source_quality, source_agent, source_session, source_chunk,
                       created_at, first_seen, last_confirmed, supersedes,
                       COALESCE(contradiction_count, 0), COALESCE(is_current_truth, TRUE)
                FROM memory_items WHERE id = %s
            """, (item_id,))

            row = cur.fetchone()
            if not row:
                return {"error": f"Item {item_id} not found"}

            item = {
                "id": row[0], "text": row[1], "memory_type": row[2],
                "entity": row[3], "property": row[4], "value": row[5],
                "scope": row[6], "status": row[7], "confidence": float(row[8] or 0),
                "source_quality": row[9], "source_agent": row[10],
                "source_session": row[11], "source_chunk": row[12],
                "created_at": str(row[13] or ""), "first_seen": str(row[14] or ""),
                "last_confirmed": str(row[15] or ""), "supersedes": row[16],
                "contradiction_count": int(row[17] or 0),
                "is_current_truth": bool(row[18]),
            }

            # Trust score
            trust = compute_trust_score(item)

            # Superseded by
            cur.execute("SELECT id, text FROM memory_items WHERE supersedes = %s AND status = 'active'", (item_id,))
            superseded_by = [{"id": r[0], "text": r[1][:100]} for r in cur.fetchall()]

            # Contradictions
            contradictions = []
            try:
                cur.execute("""
                    SELECT ce.edge_id, ce.item_a_id, ce.item_b_id,
                           ce.detection_method, ce.confidence, ce.resolution_state
                    FROM contradiction_edges ce
                    WHERE ce.item_a_id = %s OR ce.item_b_id = %s
                """, (item_id, item_id))
                for r in cur.fetchall():
                    other = r[2] if r[1] == item_id else r[1]
                    contradictions.append({
                        "edge_id": r[0], "other_item": other,
                        "method": r[3], "confidence": float(r[4]),
                        "state": r[5],
                    })
            except Exception:
                pass

            return {
                "item": item,
                "trust_score": trust.to_dict(),
                "superseded_by": superseded_by,
                "contradictions": contradictions,
            }

    finally:
        if should_close:
            conn.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Memory Inspector")
    sub = parser.add_subparsers(dest="command")

    inspect_p = sub.add_parser("inspect", help="Inspect a single item")
    inspect_p.add_argument("item_id")

    sub.add_parser("audit", help="Run quality audit")

    trust_p = sub.add_parser("trust", help="Compute trust score for an item")
    trust_p.add_argument("item_id")

    args = parser.parse_args()

    if args.command == "inspect":
        result = inspect_item(args.item_id)
        print(json.dumps(result, indent=2, default=str))

    elif args.command == "audit":
        audit = run_quality_audit()
        print(json.dumps(audit.to_dict(), indent=2))

    elif args.command == "trust":
        import psycopg
        conn = psycopg.connect(DB_DSN)
        with conn.cursor() as cur:
            cur.execute("""
                SELECT id, confidence, source_quality, created_at, last_confirmed,
                       COALESCE(contradiction_count, 0), COALESCE(is_current_truth, TRUE)
                FROM memory_items WHERE id = %s
            """, (args.item_id,))
            row = cur.fetchone()
            if row:
                item = {
                    "id": row[0], "confidence": row[1], "source_quality": row[2],
                    "created_at": row[3], "last_confirmed": row[4],
                    "contradiction_count": row[5], "is_current_truth": row[6],
                }
                score = compute_trust_score(item)
                print(json.dumps(score.to_dict(), indent=2))
            else:
                print(f"Item {args.item_id} not found")
        conn.close()

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
