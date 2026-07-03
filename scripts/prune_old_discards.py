"""
Prune old discarded memory items beyond the retention window.

The rejection gate (REJECT_WINDOW_DAYS, default 30) only uses recent discards.
Rows older than the retention window are dead weight — they consume disk,
slow VACUUM, and inflate pg_stat. Safe to delete.

Run periodically (e.g. weekly via cron or maintenance cycle).
"""
import os
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from memory_db import POOL

# Retain discards for 2x the reject window as a safety margin.
REJECT_WINDOW_DAYS = int(os.environ.get("OPENCLAW_REJECT_WINDOW_DAYS", "30"))
RETENTION_DAYS = int(os.environ.get("OPENCLAW_DISCARD_RETENTION_DAYS", str(REJECT_WINDOW_DAYS * 2)))


def prune():
    with POOL.connection() as conn:
        cur = conn.execute(
            "DELETE FROM memory_discarded WHERE discarded_at < now() - make_interval(days => %s)",
            (RETENTION_DAYS,),
        )
        deleted = cur.rowcount
        conn.execute("COMMIT")
    print(f"Pruned {deleted} discarded items older than {RETENTION_DAYS} days.")
    return deleted


if __name__ == "__main__":
    prune()
