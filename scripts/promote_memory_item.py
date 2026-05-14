"""
Promote memory item in the memory lifecycle.

Key functions: now_iso, append_log, key_of, target_file_for_item
"""
import argparse
import json
import sys
from pathlib import Path
from datetime import datetime, timezone

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from memory_db import (
    fetch_memory_items,
    upsert_memory_item,
    close_pool,
    get_conn,
)

WORKSPACE = Path.home() / ".openclaw" / "workspace"
REVIEW_DIR = WORKSPACE / "memory" / "review"
DECISIONS_LOG = REVIEW_DIR / "decisions.log"


def now_iso():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def append_log(line: str):
    with open(DECISIONS_LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def key_of(item: dict):
    return (
        item.get("memory_type"),
        item.get("entity"),
        item.get("property"),
        item.get("value"),
    )


def target_file_for_item(item: dict) -> str:
    entity = item.get("entity")
    memory_type = item.get("memory_type")

    if entity == "browser":
        return "browser.md"
    if entity == "user_preference":
        return "preferences.md"
    if entity == "checkpoint_pipeline" and memory_type == "decision":
        return "memory-system.md"
    if memory_type == "learned_fix":
        return "learned-fixes.md"
    return "memory-system.md"


def fetch_queue_item(table_name: str, item_id: str):
    allowed = {"memory_inbox"}
    if table_name not in allowed:
        raise ValueError(f"unsupported table: {table_name}")

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(f"SELECT payload FROM {table_name} WHERE id = %s", (item_id,))
            row = cur.fetchone()
            return row[0] if row else None


def delete_queue_item(table_name: str, item_id: str):
    allowed = {"memory_inbox"}
    if table_name not in allowed:
        raise ValueError(f"unsupported table: {table_name}")

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(f"DELETE FROM {table_name} WHERE id = %s", (item_id,))
        conn.commit()


def promote_from_queue(table_name: str, item_id: str):
    target = fetch_queue_item(table_name, item_id)
    canonical_items = fetch_memory_items(["active", "superseded"])

    if target is None:
        print("NOT_FOUND")
        close_pool()
        return

    target = dict(target)

    canon_keys = {key_of(x) for x in canonical_items}
    if key_of(target) in canon_keys:
        print("ALREADY_CANONICAL")
        delete_queue_item(table_name, item_id)
        close_pool()
        return

    target["target_file"] = target_file_for_item(target)
    target["target_section"] = "Active"
    target["status"] = "active"
    target["last_confirmed"] = now_iso()
    target["suggested_route"] = None

    upsert_memory_item(target)
    delete_queue_item(table_name, item_id)

    append_log(
        f"{now_iso()} promote id={target.get('id')} from={table_name} "
        f"to=memory_items target_file={target.get('target_file')}"
    )
    print("PROMOTED")
    print(json.dumps(target, indent=2, default=str))
    close_pool()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--item-id", required=True)
    parser.add_argument("--from-queue", choices=["inbox", "auto"], required=True)
    args = parser.parse_args()

    if args.from_queue == "auto":
        print("AUTO_QUEUE_NOT_SUPPORTED_IN_DB_FIRST_MODE")
        close_pool()
        return

    table_name = "memory_inbox"
    promote_from_queue(table_name, args.item_id)


if __name__ == "__main__":
    main()
