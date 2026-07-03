"""
auto_demote_bad_items.py — Identifies consistently-irrelevant memory items for review.

Part of P0-S3: Closing the retrieval feedback loop.

Scans retrieval_feedback for items that have been repeatedly marked "bad"
across different queries. Produces a demotion shortlist for human review.

Design decisions (per tester review):
- Does NOT mutate the memory_items table directly
- Does NOT reduce confidence or change status
- Only produces a shortlist JSON artifact for human review
- Threshold: ≥3 "bad" marks across ≥2 distinct queries
- Items with net-positive feedback (more useful than bad) are excluded

Usage:
    python auto_demote_bad_items.py                    # Print shortlist to stdout
    python auto_demote_bad_items.py --output FILE      # Write to JSON file
    python auto_demote_bad_items.py --min-bad 5        # Require ≥5 bad marks
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

import psycopg

LOGGER = logging.getLogger("openclaw.auto_demote_bad_items")

DB_DSN = (os.environ.get("OPENCLAW_MEMORY_DSN") or os.environ.get("OPENCLAW_MEMORY_DB_DSN") or "dbname=openclaw_memory")  # honor both canonical DSN env names

# Default thresholds
DEFAULT_MIN_BAD = 3          # Minimum "bad" feedback count
DEFAULT_MIN_QUERIES = 2      # Bad across at least this many distinct queries
DEFAULT_NET_THRESHOLD = 0    # Only flag if bad > useful (net negative)


def scan_demotion_candidates(
    *,
    min_bad: int = DEFAULT_MIN_BAD,
    min_distinct_queries: int = DEFAULT_MIN_QUERIES,
    conn: psycopg.Connection | None = None,
) -> list[dict]:
    """
    Scan retrieval_feedback for items that are consistently marked bad.

    Returns a list of candidate dicts with:
        - item_id: the memory item ID
        - bad_count: total "bad" marks
        - useful_count: total "useful" marks
        - net_score: useful - bad (negative = more bad than useful)
        - distinct_bad_queries: number of distinct queries where marked bad
        - last_bad_at: timestamp of most recent bad mark
        - sample_queries: up to 3 query_text values where it was marked bad
    """
    close_conn = False
    if conn is None:
        conn = psycopg.connect(DB_DSN)
        close_conn = True

    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                WITH item_stats AS (
                    SELECT
                        selected_id,
                        SUM(CASE WHEN operator_feedback = 'bad' THEN 1 ELSE 0 END) AS bad_count,
                        SUM(CASE WHEN operator_feedback = 'useful' THEN 1 ELSE 0 END) AS useful_count,
                        COUNT(DISTINCT CASE
                            WHEN operator_feedback = 'bad' AND query_text IS NOT NULL
                            THEN query_text
                        END) AS distinct_bad_queries,
                        MAX(CASE
                            WHEN operator_feedback = 'bad' THEN feedback_at
                        END) AS last_bad_at
                    FROM retrieval_feedback
                    WHERE operator_feedback IS NOT NULL
                      AND operator_feedback != ''
                    GROUP BY selected_id
                    HAVING
                        SUM(CASE WHEN operator_feedback = 'bad' THEN 1 ELSE 0 END) >= %s
                )
                SELECT
                    selected_id, bad_count, useful_count,
                    distinct_bad_queries, last_bad_at
                FROM item_stats
                WHERE bad_count > useful_count
                ORDER BY bad_count DESC, last_bad_at DESC
                """,
                (min_bad,),
            )
            candidates_raw = cur.fetchall()

            candidates = []
            for selected_id, bad_count, useful_count, distinct_bad_q, last_bad_at in candidates_raw:
                # Filter by distinct query count
                if distinct_bad_q < min_distinct_queries:
                    # If no query_text stored yet, relax this check
                    # (backward compat for pre-S2 feedback rows)
                    if distinct_bad_q > 0:
                        continue

                # Fetch sample bad query texts
                cur.execute(
                    """
                    SELECT DISTINCT query_text
                    FROM retrieval_feedback
                    WHERE selected_id = %s
                      AND operator_feedback = 'bad'
                      AND query_text IS NOT NULL
                    ORDER BY query_text
                    LIMIT 3
                    """,
                    (selected_id,),
                )
                sample_queries = [r[0] for r in cur.fetchall()]

                candidates.append({
                    "item_id": selected_id,
                    "bad_count": bad_count,
                    "useful_count": useful_count,
                    "net_score": useful_count - bad_count,
                    "distinct_bad_queries": distinct_bad_q,
                    "last_bad_at": last_bad_at.isoformat() if last_bad_at else None,
                    "sample_queries": sample_queries,
                })

            return candidates

    finally:
        if close_conn:
            conn.close()


def generate_shortlist_artifact(
    candidates: list[dict],
) -> dict:
    """
    Generate a structured shortlist artifact for human review.

    Returns a dict suitable for writing to JSON or feeding into a control panel.
    """
    return {
        "type": "demotion_shortlist",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "candidate_count": len(candidates),
        "candidates": candidates,
        "action_required": "Review each candidate. Items on this list have been "
                          "repeatedly marked as irrelevant during retrieval. "
                          "Consider archiving, refining, or adding exclusion tags.",
    }


def main():
    parser = argparse.ArgumentParser(
        description="Generate demotion shortlist for consistently-bad retrieval items"
    )
    parser.add_argument("--min-bad", type=int, default=DEFAULT_MIN_BAD,
                        help=f"Minimum 'bad' marks to flag (default: {DEFAULT_MIN_BAD})")
    parser.add_argument("--min-queries", type=int, default=DEFAULT_MIN_QUERIES,
                        help=f"Minimum distinct bad queries (default: {DEFAULT_MIN_QUERIES})")
    parser.add_argument("--output", type=str, default=None,
                        help="Write shortlist to this JSON file")
    args = parser.parse_args()

    candidates = scan_demotion_candidates(
        min_bad=args.min_bad,
        min_distinct_queries=args.min_queries,
    )

    artifact = generate_shortlist_artifact(candidates)

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(artifact, f, indent=2)
        print(f"Wrote {len(candidates)} candidates to {args.output}")
    else:
        print(json.dumps(artifact, indent=2))


if __name__ == "__main__":
    main()
