import argparse
import json
import sys
from pathlib import Path
from datetime import datetime, timezone

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from memory_db import fetch_memory_items, upsert_memory_item, close_pool

WORKSPACE = Path.home() / ".openclaw" / "workspace"
REVIEW_DIR = WORKSPACE / "memory" / "review"
DECISIONS_LOG = REVIEW_DIR / "decisions.log"


def now_iso():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def json_safe(value):
    if hasattr(value, "isoformat"):
        return value.isoformat()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def append_log(line: str):
    with open(DECISIONS_LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def fetch_item_by_id(item_id: str):
    items = fetch_memory_items(["active", "superseded", "archived", "uncertain"])
    for item in items:
        if item.get("id") == item_id:
            return dict(item)
    return None


def restore_item(item_id: str):
    target = fetch_item_by_id(item_id)

    if target is None:
        print("NOT_FOUND")
        close_pool()
        return

    current_status = target.get("status")

    if current_status == "active":
        print("ALREADY_ACTIVE")
        close_pool()
        return

    target["status"] = "active"
    target["last_confirmed"] = now_iso()

    # Restored items should not keep stale queue-routing hints.
    if target.get("suggested_route") in {"inbox", "discard", "daily", "project"}:
        target["suggested_route"] = None

    upsert_memory_item(target)

    append_log(
        f"{now_iso()} restore id={target.get('id')} "
        f"from_status={current_status} to_status=active"
    )
    print("RESTORED")
    print(json.dumps(target, indent=2, default=json_safe))
    close_pool()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--item-id", required=True)
    args = parser.parse_args()
    restore_item(args.item_id)


if __name__ == "__main__":
    main()
