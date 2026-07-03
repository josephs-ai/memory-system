"""
Backfill the effective_ts_epoch payload field onto EXISTING Qdrant memory_items
points, without re-embedding.

Reads (id, last_confirmed, first_seen, created_at) from Postgres, resolves the
effective timestamp to epoch seconds (same COALESCE order as temporal.py), and
calls Qdrant set_payload on the corresponding point id (uuid5 mapping). Batched,
idempotent, resumable by id cursor.

Usage:
    python backfill_qdrant_timestamps.py [--status active] [--batch 500] [--dry-run]

Run under the memory-db venv (has qdrant_client + psycopg):
    ~/.openclaw/venvs/memory-db/bin/python backfill_qdrant_timestamps.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import psycopg
from qdrant_client.http import models

from vector_store_qdrant import (
    EFFECTIVE_TS_FIELD,
    QDRANT_COLLECTION,
    effective_ts_epoch,
    ensure_qdrant_collection,
    get_qdrant_client,
    qdrant_point_id,
)

DB_DSN = "dbname=openclaw_memory"


def fetch_timestamp_rows(status: str, batch: int, after_id: str | None):
    """Page through memory_items by id, returning (id, last_confirmed, first_seen, created_at)."""
    where = ["status = %s"]
    params: list = [status]
    if after_id is not None:
        where.append("id > %s")
        params.append(after_id)
    params.append(batch)
    sql = f"""
        SELECT id, last_confirmed, first_seen, created_at
        FROM memory_items
        WHERE {' AND '.join(where)}
        ORDER BY id
        LIMIT %s
    """
    with psycopg.connect(DB_DSN) as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            cols = [d[0] for d in cur.description]
            return [dict(zip(cols, rec)) for rec in cur.fetchall()]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--status", default="active")
    ap.add_argument("--batch", type=int, default=500)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument(
        "--missing-out",
        default=None,
        help="Optional path to write ids of PG items with no Qdrant point (404s).",
    )
    args = ap.parse_args()

    ensure_qdrant_collection()  # also ensures the payload index exists
    client = get_qdrant_client()

    after_id: str | None = None
    total = 0
    updated = 0
    undated = 0
    missing = 0
    errored = 0

    # PG rows can reference items that were never embedded into Qdrant (e.g. the
    # benchmark dataset uses a different id scheme), so set_payload returns 404
    # for those. That is expected and non-fatal — we count them and (optionally)
    # record the ids for later coverage reconciliation, rather than spamming a
    # multi-line traceback per miss across a 200k+ run.
    miss_fh = None
    if args.missing_out:
        miss_fh = open(args.missing_out, "w")

    while True:
        rows = fetch_timestamp_rows(args.status, args.batch, after_id)
        if not rows:
            break

        for r in rows:
            after_id = r["id"]
            total += 1
            eff = effective_ts_epoch(r)
            if eff is None:
                undated += 1
                continue
            pid = qdrant_point_id(r["id"])
            if args.dry_run:
                updated += 1
                continue
            try:
                client.set_payload(
                    collection_name=QDRANT_COLLECTION,
                    payload={EFFECTIVE_TS_FIELD: eff},
                    points=[pid],
                )
                updated += 1
            except Exception as e:
                msg = str(e)
                if "404" in msg or "Not found" in msg or "No point" in msg:
                    # Point absent in Qdrant (never embedded) — expected, count quietly.
                    missing += 1
                    if miss_fh:
                        miss_fh.write(f"{r['id']}\n")
                else:
                    # A real error (network/schema) — surface it but keep going.
                    errored += 1
                    print(f"ERROR set_payload id={r['id']} pid={pid}: {e}", file=sys.stderr)

        print(
            f"PROGRESS total={total} updated={updated} undated={undated} "
            f"missing={missing} errored={errored} cursor={after_id}"
        )

    if miss_fh:
        miss_fh.close()

    print(
        f"DONE total={total} updated={updated} undated={undated} "
        f"missing={missing} errored={errored} dry_run={args.dry_run}"
    )
    if missing:
        where = f" (ids -> {args.missing_out})" if args.missing_out else ""
        print(
            f"NOTE {missing} PG items had no Qdrant point (never embedded); "
            f"skipped, not an error{where}.",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
