"""
Auto Promote Safe Items — utility for the OpenClaw memory system.

Key functions: now_iso, append_log, identity_key, target_file_for_item
"""
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


def identity_key(item: dict):
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


def safe_to_promote(item: dict, canonical_items: list[dict]) -> bool:
    if item.get("scope") != "stable":
        return False
    if float(item.get("confidence", 0.0) or 0.0) < 0.90:
        return False
    if float(item.get("importance", 0.0) or 0.0) < 0.85:
        return False
    if item.get("memory_type") not in {"decision", "preference", "learned_fix", "fact"}:
        return False

    ikey = identity_key(item)

    for existing in canonical_items:
        if existing.get("status") != "active":
            continue

        if identity_key(existing) == ikey:
            return False

        if (
            existing.get("memory_type") == item.get("memory_type")
            and existing.get("entity") == item.get("entity")
            and existing.get("property") == item.get("property")
            and existing.get("scope") == item.get("scope")
            and existing.get("value") != item.get("value")
        ):
            return False

    return True


def fetch_queue_rows(table_name: str):
    allowed = {"memory_inbox"}
    if table_name not in allowed:
        raise ValueError(f"unsupported table: {table_name}")

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(f"SELECT id, payload FROM {table_name} ORDER BY queued_at, id")
            return [{"id": row[0], "payload": row[1]} for row in cur.fetchall()]


def delete_queue_item(table_name: str, item_id: str):
    allowed = {"memory_inbox"}
    if table_name not in allowed:
        raise ValueError(f"unsupported table: {table_name}")

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(f"DELETE FROM {table_name} WHERE id = %s", (item_id,))
        conn.commit()


def promote_inbox_queue(canonical_items: list[dict]):
    queue_rows = fetch_queue_rows("memory_inbox")
    promoted = 0

    for row in queue_rows:
        item = dict(row["payload"])
        item_id = item.get("id")

        if not safe_to_promote(item, canonical_items):
            continue

        item["status"] = "active"
        item["target_file"] = target_file_for_item(item)
        item["target_section"] = "Active"
        item["last_confirmed"] = now_iso()
        item["suggested_route"] = None

        upsert_memory_item(item)
        canonical_items.append(item)
        delete_queue_item("memory_inbox", item_id)

        promoted += 1
        append_log(
            f"{now_iso()} auto_promote id={item_id} from=memory_inbox "
            f"target_file={item.get('target_file')}"
        )

    return promoted


def main():
    canonical_items = fetch_memory_items(["active", "superseded"])

    promoted_inbox = promote_inbox_queue(canonical_items)

    total = promoted_inbox
    print(f"AUTO_PROMOTED={total}")
    print("FROM_AUTO=0")
    print(f"FROM_INBOX={promoted_inbox}")
    close_pool()


if __name__ == "__main__":
    main()
