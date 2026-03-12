import json
import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from memory_db import close_pool, fetch_memory_items


def json_safe(value):
    if hasattr(value, "isoformat"):
        return value.isoformat()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("item_id")
    args = parser.parse_args()

    items = fetch_memory_items(["active", "superseded", "archived", "uncertain"])

    for item in items:
        if item.get("id") == args.item_id:
            print("=== WHY THIS WAS WRITTEN ===")
            print()
            print(json.dumps(item, ensure_ascii=False, indent=2, default=json_safe))
            close_pool()
            return

    print("NOT FOUND")
    close_pool()


if __name__ == "__main__":
    main()
