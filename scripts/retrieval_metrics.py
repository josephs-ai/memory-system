"""
retrieval_metrics.py — Computes and persists retrieval quality metrics.

Part of P0-S3: Closing the retrieval feedback loop.

Tracks:
- Per-day: total retrievals with feedback, useful/bad ratio, feedback rate
- Per-item: hit count, feedback count, useful rate
- Trending: precision@k direction over configurable windows

Outputs:
- JSON health artifact for monitoring/control panel
- Optional write to retrieval_metrics table (created if needed)

Usage:
    python retrieval_metrics.py                    # Print metrics to stdout
    python retrieval_metrics.py --output FILE      # Write to JSON file
    python retrieval_metrics.py --window 7         # 7-day window
    python retrieval_metrics.py --persist          # Write to DB table
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from datetime import datetime, timezone, timedelta
from pathlib import Path

import psycopg

LOGGER = logging.getLogger("openclaw.retrieval_metrics")

DB_DSN = (os.environ.get("OPENCLAW_MEMORY_DSN") or os.environ.get("OPENCLAW_MEMORY_DB_DSN") or "dbname=openclaw_memory")  # honor both canonical DSN env names

DEFAULT_WINDOW_DAYS = 30


def _ensure_metrics_table(conn: psycopg.Connection):
    """Create retrieval_metrics table if it doesn't exist."""
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS retrieval_metrics (
                metric_id BIGSERIAL PRIMARY KEY,
                computed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                window_days INTEGER NOT NULL,
                total_feedback INTEGER NOT NULL,
                useful_count INTEGER NOT NULL,
                bad_count INTEGER NOT NULL,
                null_count INTEGER NOT NULL,
                useful_rate DOUBLE PRECISION,
                bad_rate DOUBLE PRECISION,
                feedback_rate DOUBLE PRECISION,
                unique_items_with_feedback INTEGER NOT NULL,
                top_useful_items JSONB,
                top_bad_items JSONB,
                payload JSONB NOT NULL DEFAULT '{}'
            )
        """)
    conn.commit()


def compute_aggregate_metrics(
    *,
    window_days: int = DEFAULT_WINDOW_DAYS,
    conn: psycopg.Connection | None = None,
) -> dict:
    """
    Compute aggregate retrieval quality metrics over a time window.

    Returns a dict with:
        - window: the time window configuration
        - totals: total feedback counts by type
        - rates: useful/bad/feedback rates
        - top_useful: items most frequently marked useful
        - top_bad: items most frequently marked bad
        - by_query_type: breakdown by query type (code/memory/timeline/general)
    """
    close_conn = False
    if conn is None:
        conn = psycopg.connect(DB_DSN)
        close_conn = True

    try:
        cutoff = datetime.now(timezone.utc) - timedelta(days=window_days)

        with conn.cursor() as cur:
            # Overall counts
            cur.execute(
                """
                SELECT
                    COUNT(*) AS total,
                    SUM(CASE WHEN operator_feedback = 'useful' THEN 1 ELSE 0 END) AS useful,
                    SUM(CASE WHEN operator_feedback = 'bad' THEN 1 ELSE 0 END) AS bad,
                    SUM(CASE WHEN operator_feedback IS NULL OR operator_feedback = '' THEN 1 ELSE 0 END) AS null_fb,
                    COUNT(DISTINCT selected_id) AS unique_items
                FROM retrieval_feedback
                WHERE feedback_at >= %s OR feedback_at IS NULL
                """,
                (cutoff,),
            )
            total, useful, bad, null_fb, unique_items = cur.fetchone()

            # Rates
            feedback_with_opinion = useful + bad
            useful_rate = useful / feedback_with_opinion if feedback_with_opinion > 0 else None
            bad_rate = bad / feedback_with_opinion if feedback_with_opinion > 0 else None
            feedback_rate = feedback_with_opinion / total if total > 0 else None

            # Top useful items
            cur.execute(
                """
                SELECT selected_id, COUNT(*) AS cnt
                FROM retrieval_feedback
                WHERE operator_feedback = 'useful'
                  AND (feedback_at >= %s OR feedback_at IS NULL)
                GROUP BY selected_id
                ORDER BY cnt DESC
                LIMIT 10
                """,
                (cutoff,),
            )
            top_useful = [{"item_id": r[0], "useful_count": r[1]} for r in cur.fetchall()]

            # Top bad items
            cur.execute(
                """
                SELECT selected_id, COUNT(*) AS cnt
                FROM retrieval_feedback
                WHERE operator_feedback = 'bad'
                  AND (feedback_at >= %s OR feedback_at IS NULL)
                GROUP BY selected_id
                ORDER BY cnt DESC
                LIMIT 10
                """,
                (cutoff,),
            )
            top_bad = [{"item_id": r[0], "bad_count": r[1]} for r in cur.fetchall()]

            # By query type
            cur.execute(
                """
                SELECT
                    COALESCE(query_type, 'unknown') AS qt,
                    COUNT(*) AS total,
                    SUM(CASE WHEN operator_feedback = 'useful' THEN 1 ELSE 0 END) AS useful,
                    SUM(CASE WHEN operator_feedback = 'bad' THEN 1 ELSE 0 END) AS bad
                FROM retrieval_feedback
                WHERE feedback_at >= %s OR feedback_at IS NULL
                GROUP BY qt
                ORDER BY total DESC
                """,
                (cutoff,),
            )
            by_query_type = {}
            for qt, qt_total, qt_useful, qt_bad in cur.fetchall():
                qt_with_opinion = qt_useful + qt_bad
                by_query_type[qt] = {
                    "total": qt_total,
                    "useful": qt_useful,
                    "bad": qt_bad,
                    "useful_rate": qt_useful / qt_with_opinion if qt_with_opinion > 0 else None,
                }

        return {
            "type": "retrieval_quality_metrics",
            "computed_at": datetime.now(timezone.utc).isoformat(),
            "window": {
                "days": window_days,
                "cutoff": cutoff.isoformat(),
            },
            "totals": {
                "total_feedback": total,
                "useful": useful,
                "bad": bad,
                "null_feedback": null_fb,
                "unique_items_with_feedback": unique_items,
            },
            "rates": {
                "useful_rate": round(useful_rate, 4) if useful_rate is not None else None,
                "bad_rate": round(bad_rate, 4) if bad_rate is not None else None,
                "feedback_rate": round(feedback_rate, 4) if feedback_rate is not None else None,
            },
            "top_useful_items": top_useful,
            "top_bad_items": top_bad,
            "by_query_type": by_query_type,
        }

    finally:
        if close_conn:
            conn.close()


def persist_metrics(metrics: dict, conn: psycopg.Connection | None = None):
    """Write metrics snapshot to the retrieval_metrics table."""
    close_conn = False
    if conn is None:
        conn = psycopg.connect(DB_DSN)
        close_conn = True

    try:
        _ensure_metrics_table(conn)
        totals = metrics.get("totals", {})
        rates = metrics.get("rates", {})
        window = metrics.get("window", {})

        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO retrieval_metrics (
                    window_days, total_feedback, useful_count, bad_count, null_count,
                    useful_rate, bad_rate, feedback_rate,
                    unique_items_with_feedback, top_useful_items, top_bad_items, payload
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    window.get("days", DEFAULT_WINDOW_DAYS),
                    totals.get("total_feedback", 0),
                    totals.get("useful", 0),
                    totals.get("bad", 0),
                    totals.get("null_feedback", 0),
                    rates.get("useful_rate"),
                    rates.get("bad_rate"),
                    rates.get("feedback_rate"),
                    totals.get("unique_items_with_feedback", 0),
                    json.dumps(metrics.get("top_useful_items", [])),
                    json.dumps(metrics.get("top_bad_items", [])),
                    json.dumps(metrics),
                ),
            )
        conn.commit()
    finally:
        if close_conn:
            conn.close()


def main():
    parser = argparse.ArgumentParser(
        description="Compute and optionally persist retrieval quality metrics"
    )
    parser.add_argument("--window", type=int, default=DEFAULT_WINDOW_DAYS,
                        help=f"Window in days (default: {DEFAULT_WINDOW_DAYS})")
    parser.add_argument("--output", type=str, default=None,
                        help="Write metrics to this JSON file")
    parser.add_argument("--persist", action="store_true",
                        help="Also persist to retrieval_metrics DB table")
    args = parser.parse_args()

    metrics = compute_aggregate_metrics(window_days=args.window)

    if args.persist:
        persist_metrics(metrics)
        print("Metrics persisted to retrieval_metrics table")

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        print(f"Wrote metrics to {args.output}")
    else:
        print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
