import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from memory_db import close_pool, get_conn


def fetch_queue_rows(table_name: str, limit: int):
    allowed = {
        "memory_inbox",
        "memory_pending_stable",
        "memory_discarded",
    }
    if table_name not in allowed:
        raise ValueError(f"unsupported table: {table_name}")

    with get_conn() as conn:
        with conn.cursor() as cur:
            order_col = "queued_at" if table_name != "memory_discarded" else "discarded_at"
            cur.execute(
                f"SELECT id, payload FROM {table_name} ORDER BY {order_col} DESC, id LIMIT %s",
                (limit,),
            )
            return [{"id": row[0], "payload": row[1]} for row in cur.fetchall()]


def preview_text(text: str, max_len: int = 160) -> str:
    s = (text or "").replace("\n", " ").strip()
    if len(s) <= max_len:
        return s
    return s[:max_len - 3] + "..."


def show_section(title: str, rows):
    print(f"=== {title} ===")
    print()

    if not rows:
        print("NONE")
        print()
        return

    for row in rows:
        item = row["payload"]
        print(f"id={item.get('id')}")
        print(f"status={item.get('status')}")
        print(f"memory_type={item.get('memory_type')} scope={item.get('scope')}")
        print(f"entity={item.get('entity')} property={item.get('property')} value={item.get('value')}")
        print(f"confidence={item.get('confidence')} importance={item.get('importance')}")
        print(f"suggested_route={item.get('suggested_route')}")
        print(f"text_preview={preview_text(item.get('text'))}")
        print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()

    inbox = fetch_queue_rows("memory_inbox", args.limit)
    pending = fetch_queue_rows("memory_pending_stable", args.limit)
    discarded = fetch_queue_rows("memory_discarded", args.limit)

    show_section("INBOX QUEUE", inbox)
    show_section("PENDING STABLE QUEUE", pending)
    show_section("DISCARDED QUEUE", discarded)

    close_pool()


if __name__ == "__main__":
    main()
