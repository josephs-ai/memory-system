"""
expand_memory_summary.py — Drill-down into consolidated memory summaries.

Part of P2-S3: Retrieval Integration.

When a summary is retrieved but the agent needs the original granular items,
this tool fetches the constituent items by their IDs from PostgreSQL.

Usage:
    python expand_memory_summary.py --summary-id summary-abc123
    python expand_memory_summary.py --summary-id summary-abc123 --limit 5
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import psycopg

LOGGER = logging.getLogger("openclaw.expand_memory_summary")

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

DB_DSN = os.environ.get("OPENCLAW_MEMORY_DSN", "dbname=openclaw_memory")


def expand_summary(
    summary_id: str,
    *,
    limit: int | None = None,
    conn: psycopg.Connection | None = None,
) -> dict:
    """
    Fetch the constituent items of a memory summary.

    Returns:
        {
            "summary_id": str,
            "summary_text": str,
            "resolution_level": str,
            "constituent_count": int,
            "items": [{"id": str, "text": str, "memory_type": str, ...}],
        }
    """
    close_conn = False
    if conn is None:
        conn = psycopg.connect(DB_DSN)
        close_conn = True

    try:
        with conn.cursor() as cur:
            # Fetch summary metadata
            cur.execute(
                """
                SELECT text, resolution_level, constituent_item_ids
                FROM memory_items
                WHERE id = %s AND memory_type = 'summary'
                """,
                (summary_id,),
            )
            row = cur.fetchone()
            if not row:
                return {
                    "summary_id": summary_id,
                    "error": "Summary not found",
                    "items": [],
                }

            summary_text, resolution, constituent_ids_raw = row
            constituent_ids = constituent_ids_raw if isinstance(constituent_ids_raw, list) else []

            if not constituent_ids:
                return {
                    "summary_id": summary_id,
                    "summary_text": summary_text,
                    "resolution_level": resolution,
                    "constituent_count": 0,
                    "items": [],
                }

            # Apply limit
            fetch_ids = constituent_ids[:limit] if limit else constituent_ids

            # Fetch constituent items
            placeholders = ",".join(["%s"] * len(fetch_ids))
            cur.execute(
                f"""
                SELECT id, text, memory_type, scope, entity, property, confidence, first_seen
                FROM memory_items
                WHERE id IN ({placeholders})
                ORDER BY first_seen ASC NULLS LAST
                """,
                tuple(fetch_ids),
            )
            cols = [d[0] for d in cur.description]
            items = [dict(zip(cols, r)) for r in cur.fetchall()]

        return {
            "summary_id": summary_id,
            "summary_text": summary_text,
            "resolution_level": resolution,
            "constituent_count": len(constituent_ids),
            "items_returned": len(items),
            "items": items,
        }

    finally:
        if close_conn:
            conn.close()


def main():
    parser = argparse.ArgumentParser(description="Expand a memory summary into constituent items")
    parser.add_argument("--summary-id", required=True)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    result = expand_summary(args.summary_id, limit=args.limit)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
