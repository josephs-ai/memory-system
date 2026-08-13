"""
Synchronize memory to neo4j.

Key functions: main
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from memory_db import fetch_memory_items_for_embedding, close_pool, get_conn
from graph_store_neo4j import ensure_memory_constraints, upsert_memory_graph


WATERMARK_KEY = "neo4j_sync:watermark"


def read_watermark():
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT value->>'updated_through' FROM memory_runtime_state WHERE key = %s",
                (WATERMARK_KEY,),
            )
            row = cur.fetchone()
    return row[0] if row and row[0] else None


def write_watermark(value: str) -> None:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO memory_runtime_state (key, value, updated_at)
                VALUES (%s, jsonb_build_object('updated_through', %s::text), now())
                ON CONFLICT (key) DO UPDATE
                    SET value = EXCLUDED.value, updated_at = now()
                """,
                (WATERMARK_KEY, value),
            )
        conn.commit()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--full",
        action="store_true",
        help="Ignore the watermark and re-sync every active item.",
    )
    args = parser.parse_args()

    ensure_memory_constraints()

    # Without a watermark this re-sent all ~18.5k items on every heartbeat, so a
    # single cycle took ~65 minutes and the 90-second worker was permanently
    # behind. Only items touched since the last successful sync need sending.
    since = None if args.full else read_watermark()
    items = fetch_memory_items_for_embedding("active", updated_since=since)
    if not items:
        print("NONE")
        print(f"SINCE={since}")
        close_pool()
        return

    synced = 0
    skipped = 0
    high_water = since

    for item in items:
        updated_at = item.get("updated_at")
        if updated_at and (high_water is None or str(updated_at) > str(high_water)):
            high_water = str(updated_at)

        if not item.get("entity") or not item.get("property") or item.get("value") is None:
            skipped += 1
            continue

        upsert_memory_graph(item)
        synced += 1

    # Advanced only after every item above was sent, so a crash mid-sync retries
    # the same window instead of stepping over it.
    if high_water:
        write_watermark(high_water)

    print(f"SYNCED={synced}")
    print(f"SKIPPED={skipped}")
    print(f"WATERMARK={high_water}")

    close_pool()


if __name__ == "__main__":
    main()
