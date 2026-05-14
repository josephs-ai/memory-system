"""
feedback_score_engine.py — Computes per-item feedback-weighted score modifiers.

Part of P0: Closing the retrieval feedback loop.

Consumes retrieval_feedback rows to produce per-item boost/penalty signals
that are blended into retrieval scoring. All processing is deterministic
aggregation — no LLM calls.

Core function:
    feedback_boost(item_id, query_type=None) → float
        Returns a score modifier (positive = boost, negative = penalty)
        based on aggregated feedback history for the item.

Query type classification:
    classify_query_type(query_text) → str
        Heuristic classification into: code | memory | timeline | general

Design decisions:
    - On-the-fly GROUP BY, no materialized views (per tester review)
    - Recency-weighted: recent feedback counts more than old feedback
    - Conservative: with sparse data, boosts are small
    - Graceful degradation: returns 0.0 if no feedback exists or on error
    - Does NOT mutate memory_items (per tester review)
"""

from __future__ import annotations

import logging
import math
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psycopg

LOGGER = logging.getLogger("openclaw.feedback_score_engine")

DB_DSN = os.environ.get("OPENCLAW_MEMORY_DSN", "dbname=openclaw_memory")

# Scoring constants
USEFUL_WEIGHT = 1.0       # Base weight for a "useful" mark
BAD_WEIGHT = -1.5         # Base weight for a "bad" mark (penalize harder than reward)
RECENCY_HALF_LIFE_DAYS = 30  # Feedback signal halves in value after this many days
MAX_BOOST = 0.25          # Cap positive boost
MIN_BOOST = -0.40         # Cap negative penalty (don't fully suppress)
COLD_START_DAMPING = 3    # Minimum feedback count before boost reaches full strength
TYPED_COLD_START = 2      # Min typed feedback rows before type-specific signal activates
TYPED_BLEND_WEIGHT = 0.6  # How much typed signal blends in (vs global) when active


# ---------------------------------------------------------------------------
# Query type classification
# ---------------------------------------------------------------------------

# Heuristic patterns for query type detection
_CODE_PATTERNS = re.compile(
    r'\b('
    r'function|class|method|import|module|def |return |variable|parameter|'
    r'parse|compile|syntax|ast|bug|error|traceback|exception|'
    r'\.py\b|\.cpp\b|\.h\b|\.ts\b|\.js\b|'
    r'code|refactor|implement|api|endpoint|schema|migration'
    r')\b',
    re.IGNORECASE,
)

_TIMELINE_PATTERNS = re.compile(
    r'\b('
    r'when|yesterday|today|last week|last month|ago|timeline|'
    r'history|chronolog|sequence|before|after|'
    r'\d{4}-\d{2}-\d{2}|'
    r'january|february|march|april|may|june|july|august|'
    r'september|october|november|december'
    r')\b',
    re.IGNORECASE,
)

_MEMORY_PATTERNS = re.compile(
    r'\b('
    r'remember|decision|preference|lesson|rule|workflow|'
    r'convention|standard|policy|agreement|'
    r'what did|who said|why did|how do we'
    r')\b',
    re.IGNORECASE,
)


def classify_query_type(query_text: str) -> str:
    """
    Classify a query into one of: code, memory, timeline, general.

    Uses heuristic pattern matching. Returns the type with the highest
    match density. Falls back to 'general' if no clear signal.
    """
    if not query_text:
        return "general"

    text = query_text.strip()
    word_count = max(len(text.split()), 1)

    code_hits = len(_CODE_PATTERNS.findall(text))
    timeline_hits = len(_TIMELINE_PATTERNS.findall(text))
    memory_hits = len(_MEMORY_PATTERNS.findall(text))

    # Normalize by query length to avoid bias toward long queries
    code_density = code_hits / word_count
    timeline_density = timeline_hits / word_count
    memory_density = memory_hits / word_count

    # Threshold: at least 1 hit and density > 0.1
    candidates = []
    if code_hits >= 1 and code_density > 0.08:
        candidates.append(("code", code_density))
    if timeline_hits >= 1 and timeline_density > 0.08:
        candidates.append(("timeline", timeline_density))
    if memory_hits >= 1 and memory_density > 0.08:
        candidates.append(("memory", memory_density))

    if not candidates:
        return "general"

    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates[0][0]


# ---------------------------------------------------------------------------
# Feedback aggregation
# ---------------------------------------------------------------------------

def _recency_weight(feedback_at: datetime | None) -> float:
    """
    Compute a recency decay weight for a feedback row.

    Recent feedback is weighted ~1.0, feedback from RECENCY_HALF_LIFE_DAYS ago
    is weighted ~0.5, and very old feedback asymptotically approaches 0.
    """
    if feedback_at is None:
        return 0.5  # Unknown time, use moderate weight

    now = datetime.now(timezone.utc)
    if feedback_at.tzinfo is None:
        feedback_at = feedback_at.replace(tzinfo=timezone.utc)

    age_days = max((now - feedback_at).total_seconds() / 86400.0, 0.0)
    return math.pow(0.5, age_days / RECENCY_HALF_LIFE_DAYS)


def _cold_start_factor(total_feedback_count: int) -> float:
    """
    Damping factor that ramps up from 0 to 1 as feedback accumulates.

    With only 1-2 feedback rows, the boost is heavily damped to avoid
    over-rotating on sparse data. At COLD_START_DAMPING+ rows, full strength.
    """
    if total_feedback_count <= 0:
        return 0.0
    return min(total_feedback_count / COLD_START_DAMPING, 1.0)


def feedback_boost(
    item_id: str,
    *,
    query_type: str | None = None,
    conn: psycopg.Connection | None = None,
) -> float:
    """
    Compute a feedback-weighted score modifier for a memory item.

    Args:
        item_id: The memory item ID
        query_type: Optional query type for per-type weighting (future S4)
        conn: Optional existing connection (avoids opening a new one)

    Returns:
        Float score modifier. Positive = boost, negative = penalty.
        Clamped to [MIN_BOOST, MAX_BOOST]. Returns 0.0 on error or no data.
    """
    try:
        return _feedback_boost_impl(item_id, query_type=query_type, conn=conn)
    except Exception:
        LOGGER.exception("feedback_boost failed for item_id=%s", item_id)
        return 0.0


def _compute_weighted_sum(rows: list[tuple[str, Any]]) -> float:
    """Compute recency-weighted feedback sum from (feedback_type, feedback_at) rows."""
    weighted_sum = 0.0
    for feedback_type, feedback_at in rows:
        recency = _recency_weight(feedback_at)
        if feedback_type == "useful":
            weighted_sum += USEFUL_WEIGHT * recency
        elif feedback_type == "bad":
            weighted_sum += BAD_WEIGHT * recency
    return weighted_sum


def _feedback_boost_impl(
    item_id: str,
    *,
    query_type: str | None = None,
    conn: psycopg.Connection | None = None,
) -> float:
    """Internal implementation — may raise on DB errors."""
    close_conn = False
    if conn is None:
        conn = psycopg.connect(DB_DSN)
        close_conn = True

    try:
        with conn.cursor() as cur:
            # Fetch all feedback rows for this item with timestamps and query_type
            cur.execute(
                """
                SELECT operator_feedback, feedback_at, query_type
                FROM retrieval_feedback
                WHERE selected_id = %s
                  AND operator_feedback IS NOT NULL
                  AND operator_feedback != ''
                ORDER BY feedback_at DESC NULLS LAST
                """,
                (item_id,),
            )
            rows = cur.fetchall()

        if not rows:
            return 0.0

        # Split into global (all) and type-specific
        all_pairs = [(fb, ts) for fb, ts, qt in rows]
        global_score = _compute_weighted_sum(all_pairs)
        global_damped = global_score * _cold_start_factor(len(all_pairs))

        # Per-query-type scoring (S4)
        if query_type and query_type != "general":
            typed_pairs = [(fb, ts) for fb, ts, qt in rows if qt == query_type]
            if len(typed_pairs) >= TYPED_COLD_START:
                typed_score = _compute_weighted_sum(typed_pairs)
                typed_damped = typed_score * _cold_start_factor(len(typed_pairs))
                # Blend: typed signal gets TYPED_BLEND_WEIGHT, global gets the rest
                blended = (TYPED_BLEND_WEIGHT * typed_damped +
                           (1.0 - TYPED_BLEND_WEIGHT) * global_damped)
                return max(MIN_BOOST, min(MAX_BOOST, blended))

        # Fall back to global-only
        return max(MIN_BOOST, min(MAX_BOOST, global_damped))

    finally:
        if close_conn:
            conn.close()


def feedback_boost_batch(
    item_ids: list[str],
    *,
    query_type: str | None = None,
    conn: psycopg.Connection | None = None,
) -> dict[str, float]:
    """
    Compute feedback boosts for multiple items in a single DB round-trip.

    Returns dict mapping item_id → boost. Items with no feedback return 0.0.
    """
    if not item_ids:
        return {}

    try:
        return _feedback_boost_batch_impl(item_ids, query_type=query_type, conn=conn)
    except Exception:
        LOGGER.exception("feedback_boost_batch failed")
        return {item_id: 0.0 for item_id in item_ids}


def _feedback_boost_batch_impl(
    item_ids: list[str],
    *,
    query_type: str | None = None,
    conn: psycopg.Connection | None = None,
) -> dict[str, float]:
    """Internal batch implementation."""
    close_conn = False
    if conn is None:
        conn = psycopg.connect(DB_DSN)
        close_conn = True

    try:
        with conn.cursor() as cur:
            placeholders = ",".join(["%s"] * len(item_ids))
            cur.execute(
                f"""
                SELECT selected_id, operator_feedback, feedback_at, query_type
                FROM retrieval_feedback
                WHERE selected_id IN ({placeholders})
                  AND operator_feedback IS NOT NULL
                  AND operator_feedback != ''
                ORDER BY selected_id, feedback_at DESC NULLS LAST
                """,
                tuple(item_ids),
            )
            rows = cur.fetchall()

        # Group by item_id
        from collections import defaultdict
        by_item: dict[str, list[tuple[str, Any, str | None]]] = defaultdict(list)
        for selected_id, feedback_type, feedback_at, qt in rows:
            by_item[selected_id].append((feedback_type, feedback_at, qt))

        use_typed = query_type and query_type != "general"

        result = {}
        for item_id in item_ids:
            feedback_rows = by_item.get(item_id, [])
            if not feedback_rows:
                result[item_id] = 0.0
                continue

            # Global score
            all_pairs = [(fb, ts) for fb, ts, qt in feedback_rows]
            global_score = _compute_weighted_sum(all_pairs)
            global_damped = global_score * _cold_start_factor(len(all_pairs))

            # Per-query-type blending (S4)
            if use_typed:
                typed_pairs = [(fb, ts) for fb, ts, qt in feedback_rows if qt == query_type]
                if len(typed_pairs) >= TYPED_COLD_START:
                    typed_score = _compute_weighted_sum(typed_pairs)
                    typed_damped = typed_score * _cold_start_factor(len(typed_pairs))
                    blended = (TYPED_BLEND_WEIGHT * typed_damped +
                               (1.0 - TYPED_BLEND_WEIGHT) * global_damped)
                    result[item_id] = max(MIN_BOOST, min(MAX_BOOST, blended))
                    continue

            result[item_id] = max(MIN_BOOST, min(MAX_BOOST, global_damped))

        return result

    finally:
        if close_conn:
            conn.close()


# ---------------------------------------------------------------------------
# CLI for testing / inspection
# ---------------------------------------------------------------------------

def main():
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Feedback score engine CLI")
    sub = parser.add_subparsers(dest="command")

    # feedback_boost
    boost_p = sub.add_parser("boost", help="Get feedback boost for an item")
    boost_p.add_argument("item_id")
    boost_p.add_argument("--query-type", default=None)

    # classify
    classify_p = sub.add_parser("classify", help="Classify a query type")
    classify_p.add_argument("query")

    # batch
    batch_p = sub.add_parser("batch", help="Batch feedback boosts")
    batch_p.add_argument("item_ids", nargs="+")

    # stats — show all items with non-zero feedback
    sub.add_parser("stats", help="Show all items with feedback")

    args = parser.parse_args()

    if args.command == "boost":
        score = feedback_boost(args.item_id, query_type=args.query_type)
        print(json.dumps({"item_id": args.item_id, "boost": score}))

    elif args.command == "classify":
        qtype = classify_query_type(args.query)
        print(json.dumps({"query": args.query, "query_type": qtype}))

    elif args.command == "batch":
        boosts = feedback_boost_batch(args.item_ids)
        print(json.dumps(boosts, indent=2))

    elif args.command == "stats":
        conn = psycopg.connect(DB_DSN)
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    selected_id,
                    COUNT(*) AS total,
                    SUM(CASE WHEN operator_feedback = 'useful' THEN 1 ELSE 0 END) AS useful,
                    SUM(CASE WHEN operator_feedback = 'bad' THEN 1 ELSE 0 END) AS bad,
                    MAX(feedback_at) AS last_feedback
                FROM retrieval_feedback
                WHERE operator_feedback IS NOT NULL AND operator_feedback != ''
                GROUP BY selected_id
                ORDER BY total DESC
                """
            )
            for row in cur.fetchall():
                item_id, total, useful, bad, last_at = row
                boost = feedback_boost(item_id, conn=conn)
                print(json.dumps({
                    "item_id": item_id,
                    "total_feedback": total,
                    "useful": useful,
                    "bad": bad,
                    "last_feedback": str(last_at) if last_at else None,
                    "computed_boost": boost,
                }))
        conn.close()

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
