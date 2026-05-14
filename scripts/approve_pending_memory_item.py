"""
Approve pending memory item for promotion to durable status.

Key functions: now_iso, append_log, main
"""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from memory_db import (
    fetch_pending_stable_item,
    delete_pending_stable_item,
    upsert_memory_item,
    close_pool,
)

WORKSPACE = Path.home() / ".openclaw" / "workspace"
DECISIONS_LOG = WORKSPACE / "memory" / "review" / "decisions.log"


def now_iso():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def append_log(line: str):
    with open(DECISIONS_LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--item-id", required=True)
    parser.add_argument("--approved-by", default="human")
    args = parser.parse_args()

    approved = fetch_pending_stable_item(args.item_id)

    if approved is None:
        print("NOT_FOUND")
        close_pool()
        return

    approved = dict(approved)
    approved["status"] = "active"
    approved["approved_at"] = now_iso()
    approved["approved_by"] = args.approved_by
    approved["approval_source"] = "pending-stable"

    upsert_memory_item(approved)
    delete_pending_stable_item(args.item_id)

    append_log(
        f"{now_iso()} approve_pending id={approved.get('id')} "
        f"approved_by={args.approved_by} target=memory_items"
    )

    print("APPROVED")
    print(json.dumps(approved, ensure_ascii=False, indent=2))
    close_pool()


if __name__ == "__main__":
    main()
