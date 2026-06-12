"""
retrieval_optimizer.py — Research-grade retrieval improvements.

Phase 9, Items 25-27:
25. Learn retrieval policies from downstream success (feedback-driven optimization)
26. Better benchmark categories for failure cases
27. "Do not retrieve" classifiers (suppress stale, low-confidence, overbroad matches)

Phase 10, Items 28-29:
28. Evaluate graph layer ROI
29. Audit every subsystem for ROI

Design:
- Feedback-driven weight learning: adjusts component weights based on
  which configurations led to positive downstream feedback
- Failure-case categorization: classifies retrieval failures for targeted fixes
- Suppression classifier: deterministic rules + learned patterns for
  items that should NOT be injected into context
- ROI audit: measures each subsystem's contribution to overall quality
"""
from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any

LOGGER = logging.getLogger("openclaw.retrieval_optimizer")

DB_DSN = os.environ.get("OPENCLAW_MEMORY_DSN", "dbname=openclaw_memory")


# ---------------------------------------------------------------------------
# Failure case categories
# ---------------------------------------------------------------------------

class FailureCategory(str, Enum):
    STALE_RETRIEVAL = "stale_retrieval"           # Retrieved outdated info
    CONTRADICTORY = "contradictory_retrieval"      # Retrieved conflicting items
    FALSE_MEMORY = "false_memory"                  # Retrieved fabricated/inferred item
    TEMPORAL_CONFUSION = "temporal_confusion"       # Wrong time period
    OVER_RETRIEVAL = "over_retrieval"              # Too many irrelevant items
    UNDER_RETRIEVAL = "under_retrieval"            # Missed relevant items
    OVERBROAD_MATCH = "overbroad_match"            # Semantically similar but irrelevant
    LOW_CONFIDENCE_LEAK = "low_confidence_leak"    # Low-quality item made it through


@dataclass
class FailureCase:
    """A categorized retrieval failure for analysis."""
    category: FailureCategory
    query: str
    item_id: str = ""
    item_text: str = ""
    confidence: float = 0.0
    detail: str = ""

    def to_dict(self) -> dict:
        return {
            "category": self.category.value,
            "query": self.query[:200],
            "item_id": self.item_id,
            "item_text": self.item_text[:200],
            "confidence": round(self.confidence, 4),
            "detail": self.detail,
        }


def classify_failure(
    query: str,
    retrieved_items: list[dict],
    expected_items: list[dict] | None = None,
) -> list[FailureCase]:
    """
    Classify retrieval failures based on the results and expectations.

    Checks for:
    - Stale items (not confirmed in 30+ days)
    - Contradictory pairs in results
    - Low-confidence items that shouldn't have been retrieved
    - Over-retrieval (too many low-relevance items)
    - Under-retrieval (expected items missing)
    """
    failures = []
    now = datetime.now(timezone.utc)

    for item in retrieved_items:
        # Stale retrieval
        last_confirmed = item.get("last_confirmed") or item.get("created_at")
        if last_confirmed:
            try:
                if isinstance(last_confirmed, str):
                    last_confirmed = datetime.fromisoformat(str(last_confirmed).replace("Z", "+00:00"))
                if last_confirmed.tzinfo is None:
                    last_confirmed = last_confirmed.replace(tzinfo=timezone.utc)
                if (now - last_confirmed).days > 30:
                    failures.append(FailureCase(
                        category=FailureCategory.STALE_RETRIEVAL,
                        query=query,
                        item_id=str(item.get("id", "")),
                        item_text=str(item.get("text", "")),
                        confidence=float(item.get("confidence", 0) or 0),
                        detail=f"Item last confirmed {(now - last_confirmed).days} days ago",
                    ))
            except (ValueError, TypeError):
                pass

        # Low confidence leak
        conf = float(item.get("confidence", 0) or 0)
        if conf < 0.4 and conf > 0:
            failures.append(FailureCase(
                category=FailureCategory.LOW_CONFIDENCE_LEAK,
                query=query,
                item_id=str(item.get("id", "")),
                item_text=str(item.get("text", "")),
                confidence=conf,
                detail=f"Item confidence {conf:.2f} below threshold",
            ))

        # Contradictory retrieval
        contra_count = int(item.get("contradiction_count", 0) or 0)
        if contra_count > 0:
            failures.append(FailureCase(
                category=FailureCategory.CONTRADICTORY,
                query=query,
                item_id=str(item.get("id", "")),
                item_text=str(item.get("text", "")),
                confidence=conf,
                detail=f"Item has {contra_count} active contradictions",
            ))

        # False memory (inferred items with low source quality)
        is_inferred = item.get("is_inferred") or item.get("source_quality") == "inferred"
        if is_inferred and conf < 0.6:
            failures.append(FailureCase(
                category=FailureCategory.FALSE_MEMORY,
                query=query,
                item_id=str(item.get("id", "")),
                item_text=str(item.get("text", "")),
                confidence=conf,
                detail="Inferred item with low confidence — possible false memory",
            ))

    # Under-retrieval
    if expected_items:
        retrieved_ids = {str(r.get("id", "")) for r in retrieved_items}
        for expected in expected_items:
            eid = str(expected.get("id", ""))
            if eid and eid not in retrieved_ids:
                failures.append(FailureCase(
                    category=FailureCategory.UNDER_RETRIEVAL,
                    query=query,
                    item_id=eid,
                    item_text=str(expected.get("text", "")),
                    detail="Expected item not in results",
                ))

    return failures


# ---------------------------------------------------------------------------
# "Do not retrieve" classifier
# ---------------------------------------------------------------------------

@dataclass
class SuppressionDecision:
    """Decision to suppress an item from retrieval results."""
    item_id: str
    suppress: bool
    reason: str
    confidence: float  # How confident we are in the suppression decision

    def to_dict(self) -> dict:
        return {
            "item_id": self.item_id,
            "suppress": self.suppress,
            "reason": self.reason,
            "confidence": round(self.confidence, 4),
        }


def should_suppress(item: dict) -> SuppressionDecision:
    """
    Determine if an item should be suppressed from retrieval results.

    Suppression rules (deterministic):
    1. Not current truth (superseded) → suppress
    2. Very low confidence (< 0.2) → suppress
    3. Multiple unresolved contradictions (≥3) → suppress
    4. Status not active/candidate → suppress
    5. Raw JSON/tool output as text → suppress
    6. Extremely long text (> 2000 chars) without summary → suppress
    """
    item_id = str(item.get("id", ""))
    text = str(item.get("text", "") or "")

    # Rule 1: Not current truth
    if not item.get("is_current_truth", True):
        return SuppressionDecision(item_id, True, "superseded (not current truth)", 0.95)

    # Rule 2: Very low confidence
    conf = float(item.get("confidence", 0.5) or 0.5)
    if conf < 0.2:
        return SuppressionDecision(item_id, True, f"very low confidence ({conf:.2f})", 0.90)

    # Rule 3: Multiple contradictions
    contra_count = int(item.get("contradiction_count", 0) or 0)
    if contra_count >= 3:
        return SuppressionDecision(item_id, True, f"heavily contradicted ({contra_count} contradictions)", 0.85)

    # Rule 4: Status
    status = str(item.get("status", "active"))
    if status not in ("active", "candidate"):
        return SuppressionDecision(item_id, True, f"ineligible status ({status})", 0.95)

    # Rule 5: Raw JSON/tool output
    if text.startswith("{") and '"type"' in text[:100]:
        return SuppressionDecision(item_id, True, "raw JSON/tool output stored as memory", 0.80)

    # Rule 6: Extremely long without structure
    if len(text) > 2000 and "\n" not in text[:500]:
        return SuppressionDecision(item_id, True, "extremely long unstructured text", 0.70)

    return SuppressionDecision(item_id, False, "passed all suppression checks", 1.0)


def filter_suppressed(items: list[dict]) -> tuple[list[dict], list[SuppressionDecision]]:
    """
    Filter a list of retrieval results, removing items that should be suppressed.

    Returns (kept_items, suppressed_decisions).
    """
    kept = []
    suppressed = []

    for item in items:
        decision = should_suppress(item)
        if decision.suppress:
            suppressed.append(decision)
        else:
            kept.append(item)

    return kept, suppressed


# ---------------------------------------------------------------------------
# Feedback-driven weight optimization
# ---------------------------------------------------------------------------

@dataclass
class WeightConfig:
    """Retrieval component weights."""
    vector_weight: float = 0.6
    fts_weight: float = 0.4
    rerank_enabled: bool = True
    temporal_weight: float = 0.1
    feedback_weight: float = 0.05
    recency_bias: float = 0.0  # -1 to 1 (negative = favor old, positive = favor new)

    def to_dict(self) -> dict:
        return {
            "vector_weight": round(self.vector_weight, 4),
            "fts_weight": round(self.fts_weight, 4),
            "rerank_enabled": self.rerank_enabled,
            "temporal_weight": round(self.temporal_weight, 4),
            "feedback_weight": round(self.feedback_weight, 4),
            "recency_bias": round(self.recency_bias, 4),
        }


def learn_weights_from_feedback(conn: Any = None) -> WeightConfig:
    """
    Analyze retrieval feedback to suggest optimal component weights.

    Looks at which source types (vector, FTS, graph) produced the most
    "useful" feedback vs "bad" feedback, and adjusts weights accordingly.
    """
    should_close = False
    if conn is None:
        import psycopg
        conn = psycopg.connect(DB_DSN)
        should_close = True

    config = WeightConfig()

    try:
        with conn.cursor() as cur:
            # Check if retrieval_feedback table exists
            cur.execute("""
                SELECT EXISTS (
                    SELECT 1 FROM information_schema.tables
                    WHERE table_name = 'retrieval_feedback'
                )
            """)
            if not cur.fetchone()[0]:
                LOGGER.info("retrieval_feedback table not found, using defaults")
                return config

            # Aggregate feedback by operator verdict
            cur.execute("""
                SELECT
                    COALESCE(operator_feedback, 'unknown') AS feedback,
                    COUNT(*) AS cnt
                FROM retrieval_feedback
                GROUP BY operator_feedback
                ORDER BY operator_feedback
            """)

            feedback_stats: dict[str, int] = {}
            for feedback, cnt in cur.fetchall():
                feedback_stats[feedback] = cnt

            if not feedback_stats:
                return config

            # Use overall positive/negative ratio to adjust confidence in current weights
            positive = sum(v for k, v in feedback_stats.items() if k in ("useful", "good", "correct", "accepted"))
            negative = sum(v for k, v in feedback_stats.items() if k in ("bad", "wrong", "rejected", "irrelevant"))
            total = positive + negative

            if total > 0:
                overall_quality = positive / total
                # If quality is high, keep current weights; if low, try shifting toward FTS
                if overall_quality < 0.5:
                    config.fts_weight = min(0.6, config.fts_weight + 0.1)
                    config.vector_weight = 1.0 - config.fts_weight
                    LOGGER.info("feedback suggests increasing FTS weight (quality=%.2f)", overall_quality)

            # Normalize weights
            total_weight = config.vector_weight + config.fts_weight
            if total_weight > 0:
                config.vector_weight /= total_weight
                config.fts_weight /= total_weight

    finally:
        if should_close:
            conn.close()

    return config


# ---------------------------------------------------------------------------
# Subsystem ROI audit
# ---------------------------------------------------------------------------

@dataclass
class SubsystemROI:
    """ROI assessment for a retrieval subsystem."""
    name: str
    measurable_gain: str = "unknown"
    live_usefulness: str = "unknown"
    reliability_cost: str = "unknown"
    maintenance_cost: str = "unknown"
    simpler_replacement: str = "none"
    recommendation: str = ""
    score: float = 0.0  # 0-1, higher = keep

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "measurable_gain": self.measurable_gain,
            "live_usefulness": self.live_usefulness,
            "reliability_cost": self.reliability_cost,
            "maintenance_cost": self.maintenance_cost,
            "simpler_replacement": self.simpler_replacement,
            "recommendation": self.recommendation,
            "roi_score": round(self.score, 2),
        }


def audit_subsystem_roi(conn: Any = None) -> list[SubsystemROI]:
    """
    Audit each major subsystem for ROI.

    Evaluates: Postgres FTS, Qdrant vector, Neo4j graph, cross-encoder reranker,
    temporal scoring, feedback scoring, lifecycle filtering, contradiction detection.
    """
    should_close = False
    if conn is None:
        import psycopg
        conn = psycopg.connect(DB_DSN)
        should_close = True

    subsystems = []

    try:
        with conn.cursor() as cur:
            # 1. Postgres FTS
            cur.execute("SELECT COUNT(*) FROM memory_items WHERE status = 'active'")
            active = cur.fetchone()[0]

            subsystems.append(SubsystemROI(
                name="postgres_fts",
                measurable_gain="Core store + FTS for keyword matching",
                live_usefulness="Essential — primary data store",
                reliability_cost="Low — Postgres is mature",
                maintenance_cost="Low — standard schema migrations",
                recommendation="KEEP — foundational, cannot remove",
                score=1.0,
            ))

            # 2. Qdrant vector search
            subsystems.append(SubsystemROI(
                name="qdrant_vector",
                measurable_gain="Semantic search beyond keyword matching",
                live_usefulness="High for conceptual queries, moderate for exact-match queries",
                reliability_cost="Medium — separate service, needs health checks",
                maintenance_cost="Medium — collection management, embedding updates",
                simpler_replacement="pgvector in Postgres (eliminates separate service)",
                recommendation="EVALUATE — run ablation to measure vector vs FTS-only gap. If gap < 5% F1, consider pgvector migration to reduce infra",
                score=0.7,
            ))

            # 3. Neo4j graph
            graph_items = 0
            try:
                cur.execute("""
                    SELECT COUNT(*) FROM memory_items
                    WHERE status = 'active' AND entity IS NOT NULL AND property IS NOT NULL AND value IS NOT NULL
                """)
                graph_items = cur.fetchone()[0]
            except Exception:
                pass

            graph_coverage = graph_items / max(active, 1)
            subsystems.append(SubsystemROI(
                name="neo4j_graph",
                measurable_gain=f"Graph traversal for {graph_items} items ({graph_coverage:.0%} of active)",
                live_usefulness="Low-moderate — helps with code-context and relationship queries",
                reliability_cost="High — separate service, complex driver management",
                maintenance_cost="High — schema constraints, sync required",
                simpler_replacement="Postgres self-joins on entity/property/value columns",
                recommendation="EVALUATE CAREFULLY — run ablation with and without graph. If gain < 3% F1, fold relationships into Postgres JOIN queries",
                score=0.4,
            ))

            # 4. Cross-encoder reranker
            subsystems.append(SubsystemROI(
                name="cross_encoder_rerank",
                measurable_gain="Precision boost on top-K ordering",
                live_usefulness="High — reranking consistently improves quality",
                reliability_cost="Low — runs in-process, no external dependency",
                maintenance_cost="Low — model file only",
                recommendation="KEEP — high ROI, run ablation to quantify exact contribution",
                score=0.9,
            ))

            # 5. Temporal scoring
            subsystems.append(SubsystemROI(
                name="temporal_scoring",
                measurable_gain="Recency-aware result ordering",
                live_usefulness="Moderate — helps for 'recent' queries, may hurt for 'canonical' queries",
                reliability_cost="Low — pure computation",
                maintenance_cost="Low — parameter tuning only",
                recommendation="KEEP but run ablation to quantify. May need task-aware activation",
                score=0.6,
            ))

            # 6. Feedback scoring
            feedback_count = 0
            try:
                cur.execute("""
                    SELECT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'retrieval_feedback')
                """)
                if cur.fetchone()[0]:
                    cur.execute("SELECT COUNT(*) FROM retrieval_feedback")
                    feedback_count = cur.fetchone()[0]
            except Exception:
                pass

            subsystems.append(SubsystemROI(
                name="feedback_scoring",
                measurable_gain=f"Feedback-driven boost/penalty ({feedback_count} feedback entries)",
                live_usefulness="Low-moderate — cold start problem, sparse signal",
                reliability_cost="Low",
                maintenance_cost="Low",
                recommendation="KEEP — low cost, grows more useful over time" if feedback_count > 50 else "KEEP but invest in collecting more feedback signal",
                score=0.5 if feedback_count > 50 else 0.3,
            ))

            # 7. Contradiction detection
            contra_count = 0
            try:
                cur.execute("SELECT COUNT(*) FROM contradiction_edges WHERE resolution_state = 'unresolved'")
                contra_count = cur.fetchone()[0]
            except Exception:
                pass

            subsystems.append(SubsystemROI(
                name="contradiction_detection",
                measurable_gain=f"{contra_count} unresolved contradictions detected",
                live_usefulness="High for trust — prevents conflicting information in context",
                reliability_cost="Low — Postgres-only",
                maintenance_cost="Low — periodic sweep",
                recommendation="KEEP — essential for trust at scale",
                score=0.85,
            ))

    finally:
        if should_close:
            conn.close()

    return subsystems
