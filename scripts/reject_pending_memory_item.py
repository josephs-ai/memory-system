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
    insert_discarded_item,
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
    parser.add_argument("--rejected-by", default="human")
    parser.add_argument("--reason", default="manual_rejection")
    args = parser.parse_args()

    rejected = fetch_pending_stable_item(args.item_id)

    if rejected is None:
        print("NOT_FOUND")
        close_pool()
        return

    rejected = dict(rejected)
    rejected["status"] = "discarded"
    rejected["rejected_at"] = now_iso()
    rejected["rejected_by"] = args.rejected_by
    rejected["rejection_reason"] = args.reason
    rejected["rejection_source"] = "pending-stable"

    insert_discarded_item(rejected)
    delete_pending_stable_item(args.item_id)

    append_log(
        f"{now_iso()} reject_pending id={rejected.get('id')} "
        f"rejected_by={args.rejected_by} reason={args.reason} target=memory_discarded"
    )

    print("REJECTED")
    print(json.dumps(rejected, ensure_ascii=False, indent=2))
    close_pool()


if __name__ == "__main__":
    main()
