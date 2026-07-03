"""
One-time migration: promote eligible inbox items to durable memory.

Targets items that meet the AUTO criteria (confidence >= 0.85,
authority_basis = user_explicit, allowed memory_type) but were stuck
in inbox because the approval gate was too strict.

Run once after fixing the promotion gate, then delete this script.
"""
import argparse
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from memory_db import POOL, upsert_memory_item, close_pool

ALLOWED_TYPES = {"fact", "decision", "learned_fix", "preference", "observation", "rule",
                 "implementation_pattern", "architecture_rule"}
MIN_CONFIDENCE = 0.85


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Show what would be promoted without writing")
    args = parser.parse_args()

    with POOL.connection() as conn:
        rows = conn.execute("""
            SELECT id, payload FROM memory_inbox
            WHERE (payload->>'confidence')::float >= %s
              AND payload->>'authority_basis' = 'user_explicit'
        """, (MIN_CONFIDENCE,)).fetchall()

    print(f"Found {len(rows)} eligible inbox items.")

    promoted_ids = []
    skipped = 0
    failed = 0

    for item_id, payload in rows:
        memory_type = (payload.get("memory_type") or payload.get("claim_type") or "").strip().lower()
        if memory_type not in ALLOWED_TYPES:
            skipped += 1
            continue

        if args.dry_run:
            text = (payload.get("text") or "")[:100]
            print(f"  [DRY-RUN] Would promote {item_id}: {memory_type} — {text}")
            promoted_ids.append(item_id)
            continue

        payload.setdefault("id", item_id)
        payload.setdefault("status", "active")
        payload.setdefault("approved_by", "auto_migration")
        payload.setdefault("approval_source", "inbox_backlog_promotion")

        try:
            upsert_memory_item(payload)
            promoted_ids.append(item_id)
        except Exception as e:
            print(f"  Failed to promote {item_id}: {e}")
            failed += 1

    if args.dry_run:
        print(f"\nDRY-RUN: Would promote {len(promoted_ids)}, skip {skipped}")
        close_pool()
        return

    # Remove ONLY successfully promoted items from inbox
    if promoted_ids:
        with POOL.connection() as conn:
            with conn.transaction():
                conn.execute("DELETE FROM memory_inbox WHERE id = ANY(%s)", (promoted_ids,))

    print(f"Promoted: {len(promoted_ids)}, Skipped (wrong type): {skipped}, Failed: {failed}")
    close_pool()


if __name__ == "__main__":
    main()
