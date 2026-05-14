"""
Display promotion candidates from the memory system for inspection and debugging.

Key functions: fetch_queue_payloads, preview, main
"""
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from memory_db import close_pool, get_conn


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


def preview(text, n=180):
    s = (text or "").replace("\n", " ").strip()
    if len(s) <= n:
        return s
    return s[: n - 3] + "..."


def main():
    inbox = fetch_queue_payloads("memory_inbox")
    pending = fetch_queue_payloads("memory_pending_stable")

    print("=== PROMOTION CANDIDATES ===")
    print()

    found = False
    for item in inbox + pending:
        confidence = float(item.get("confidence", 0.0) or 0.0)
        importance = float(item.get("importance", 0.0) or 0.0)
        scope = item.get("scope")
        memory_type = item.get("memory_type")

        if (
            scope == "stable"
            and confidence >= 0.85
            and importance >= 0.80
            and memory_type in {"decision", "preference", "learned_fix", "fact"}
        ):
            found = True
            print(f"id={item.get('id')}")
            print(f"memory_type={memory_type} scope={scope}")
            print(f"confidence={confidence} importance={importance}")
            print(
                f"entity={item.get('entity')} "
                f"property={item.get('property')} "
                f"value={item.get('value')}"
            )
            print(f"suggested_route={item.get('suggested_route')}")
            print(f"text={preview(item.get('text'))}")
            print()

    if not found:
        print("NONE")

    close_pool()


if __name__ == "__main__":
    main()
