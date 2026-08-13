"""
Seed the episodic lane from history that already exists.

memory_checkpoints has run unbroken since 2026-03-12 and records, per run,
when it happened and which files moved. That is an activity log that nobody
was reading as one. Replaying it into memory_episodes gives the lane real
history on day one instead of only recording from now on.

Idempotent: episodes carry origin='checkpoint:<id>', and rows whose origin is
already present are skipped, so re-running adds only what is new.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from memory_db import get_conn, record_episode, close_pool


def summarize(trigger_type, trigger_reason, counts, delta_meta) -> str:
    """A searchable one-line description of what the run touched.

    Filenames are the useful text here: they are what someone actually recalls
    ("the heartbeat work", "phase 4"), so they belong in the FTS column rather
    than buried in the artifacts JSON.
    """
    counts = counts or {}
    delta_meta = delta_meta or {}
    bits = [f"{trigger_type or 'run'}/{trigger_reason or 'unknown'}"]

    for key in ("files_added", "files_changed", "files_removed", "delta_count"):
        value = counts.get(key)
        if value:
            bits.append(f"{key}={value}")

    names = []
    for key in ("added", "changed", "removed"):
        for path in (delta_meta.get(key) or [])[:12]:
            names.append(Path(str(path)).name)
    names = list(dict.fromkeys(names))
    if names:
        bits.append("files: " + ", ".join(names))
        # Postgres indexes "2026-07-01-heartbeat.md" as one token, so searching
        # "heartbeat" would never match it. Append the constituent words so the
        # terms people actually recall are individually searchable.
        words = dict.fromkeys(
            w.lower()
            for name in names
            for w in re.split(r"[^A-Za-z]+", name)
            if len(w) > 2
        )
        if words:
            bits.append("terms: " + " ".join(words))

    return " ".join(bits)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="0 = all")
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Rebuild checkpoint-derived episodes (e.g. after changing summarize()).",
    )
    args = parser.parse_args()

    with get_conn() as conn:
        with conn.cursor() as cur:
            if args.refresh:
                # Safe to drop: every one of these rows is derived from
                # memory_checkpoints and is rebuilt below. Episodes recorded
                # live (origin not 'checkpoint:%') are never touched.
                cur.execute(
                    "DELETE FROM memory_episodes WHERE origin LIKE 'checkpoint:%'"
                )
                print(f"REFRESH_DELETED={cur.rowcount}")
                conn.commit()

            # Highest checkpoint already recorded. Pulling every row instead
            # meant dragging 2.6k large delta_meta blobs across the wire on
            # each run -- 25s for a no-op, on a step that runs every 90s.
            cur.execute(
                """
                SELECT COALESCE(MAX(NULLIF(regexp_replace(origin, '^checkpoint:', ''), '')::bigint), 0)
                  FROM memory_episodes
                 WHERE origin LIKE 'checkpoint:%'
                """
            )
            high_water = cur.fetchone()[0] or 0

            cur.execute(
                """
                SELECT id, trigger_type, trigger_reason, status,
                       started_at, finished_at, delta_meta, counts
                  FROM memory_checkpoints
                 WHERE id > %s
                 ORDER BY started_at
                """,
                (high_water,),
            )
            rows = cur.fetchall()
            seen: set[str] = set()

    added = skipped = 0
    for row in rows:
        cid, ttype, treason, status, started, finished, delta_meta, counts = row
        origin = f"checkpoint:{cid}"
        if origin in seen or not started:
            skipped += 1
            continue

        record_episode(
            agent="memory-index",
            summary=summarize(ttype, treason, counts, delta_meta),
            started_at=started,
            ended_at=finished,
            artifacts={"counts": counts or {}, "delta": delta_meta or {}, "status": status},
            item_count=int((counts or {}).get("delta_count") or 0),
            origin=origin,
        )
        added += 1
        if args.limit and added >= args.limit:
            break

    print(f"BACKFILLED={added}")
    print(f"SKIPPED={skipped}")
    close_pool()


if __name__ == "__main__":
    main()
