import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from memory_db import close_pool, get_conn


def norm(value):
    if value is None:
        return ""
    return str(value).strip().lower()


def fetch_queue_payloads(table_name: str):
    allowed = {
        "memory_inbox",
        "memory_pending_stable",
    }
    if table_name not in allowed:
        raise ValueError(f"unsupported table: {table_name}")

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(f"SELECT payload FROM {table_name} ORDER BY id")
            return [row[0] for row in cur.fetchall()]


def canonical_key_exists(item: dict) -> bool:
    memory_type = norm(item.get("memory_type"))
    entity = norm(item.get("entity"))
    prop = norm(item.get("property"))
    value = norm(item.get("value"))

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT 1
                FROM memory_items
                WHERE status = 'active'
                  AND LOWER(COALESCE(memory_type, '')) = %s
                  AND LOWER(COALESCE(entity, '')) = %s
                  AND LOWER(COALESCE(property, '')) = %s
                  AND LOWER(COALESCE(value, '')) = %s
                LIMIT 1
                """,
                (memory_type, entity, prop, value),
            )
            return cur.fetchone() is not None


def preview_item(item: dict):
    keep = {
        "id",
        "memory_type",
        "scope",
        "entity",
        "property",
        "value",
        "confidence",
        "importance",
        "suggested_route",
        "text",
    }
    return json.dumps(
        {k: item.get(k) for k in keep},
        ensure_ascii=False,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=50)
    args = parser.parse_args()

    inbox_items = fetch_queue_payloads("memory_inbox")
    pending_items = fetch_queue_payloads("memory_pending_stable")

    print("=== PROMOTION CANDIDATE REPORT ===")
    print()

    print("=== INBOX ITEMS WORTH PROMOTION REVIEW ===")
    found_inbox = False
    count = 0

    for item in inbox_items:
        confidence = float(item.get("confidence", 0.0) or 0.0)
        stable_scope = norm(item.get("scope")) == "stable"
        strong_type = norm(item.get("memory_type")) in {"decision", "learned_fix", "preference", "fact"}

        if (
            confidence >= 0.65
            and stable_scope
            and strong_type
            and not canonical_key_exists(item)
        ):
            found_inbox = True
            print(preview_item(item))
            count += 1
            if count >= args.limit:
                break

    if not found_inbox:
        print("NONE")
    print()

    print("=== PENDING STABLE ITEMS READY FOR CANONICAL MEMORY ===")
    found_pending = False
    count = 0

    for item in pending_items:
        if not canonical_key_exists(item):
            found_pending = True
            print(preview_item(item))
            count += 1
            if count >= args.limit:
                break

    if not found_pending:
        print("NONE")
    print()

    close_pool()


if __name__ == "__main__":
    main()
