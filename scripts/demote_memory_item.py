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


def append_log(line: str):
    with open(DECISIONS_LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def fetch_item_by_id(item_id: str):
    items = fetch_memory_items(["active", "superseded", "archived", "uncertain"])
    for item in items:
        if item.get("id") == item_id:
            return dict(item)
    return None


def demote_item(item_id: str, new_status: str):
    target = fetch_item_by_id(item_id)

    if target is None:
        print("NOT_FOUND")
        close_pool()
        return

    old_status = target.get("status")
    target["status"] = new_status
    target["last_confirmed"] = now_iso()

    upsert_memory_item(target)

    append_log(
        f"{now_iso()} demote id={target.get('id')} "
        f"from_status={old_status} to_status={new_status}"
    )
    print("DEMOTED")
    print(json.dumps(target, indent=2, default=str))
    close_pool()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--item-id", required=True)
    parser.add_argument("--status", choices=["superseded", "archived", "uncertain"], default="archived")
    args = parser.parse_args()

    demote_item(args.item_id, args.status)


if __name__ == "__main__":
    main()
