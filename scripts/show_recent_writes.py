"""
Display recent writes from the memory system for inspection and debugging.

Key functions: parse_ts, preview, main
"""
import argparse
import sys
from datetime import datetime
from pathlib import Path
from datetime import datetime, timezone

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from memory_db import close_pool, fetch_memory_items


def parse_ts(item):
    ts = item.get("last_confirmed") or item.get("first_seen")
    if not ts:
        return datetime.min.replace(tzinfo=timezone.utc)

    if isinstance(ts, datetime):
        if ts.tzinfo is None:
            return ts.replace(tzinfo=timezone.utc)
        return ts

    try:
        parsed = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed
    except Exception:
        return datetime.min.replace(tzinfo=timezone.utc)

def preview(text, n=180):
    s = (text or "").replace("\n", " ").strip()
    if len(s) <= n:
        return s
    return s[: n - 3] + "..."


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()

    items = fetch_memory_items(["active", "superseded", "archived", "uncertain"])
    items.sort(key=parse_ts, reverse=True)

    print("=== RECENT WRITES ===")
    print()

    if not items:
        print("NONE")
        close_pool()
        return

    for item in items[: args.limit]:
        print(f"id={item.get('id')}")
        print(f"status={item.get('status')}")
        print(f"target_file={item.get('target_file')}")
        print(
            f"entity={item.get('entity')} "
            f"property={item.get('property')} "
            f"value={item.get('value')}"
        )
        print(f"last_confirmed={item.get('last_confirmed')}")
        print(f"first_seen={item.get('first_seen')}")
        print(f"text={preview(item.get('text'))}")
        print()

    close_pool()


if __name__ == "__main__":
    main()
