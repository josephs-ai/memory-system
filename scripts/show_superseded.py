"""
Display superseded from the memory system for inspection and debugging.

Key functions: preview, main
"""
import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from memory_db import close_pool, fetch_memory_items


def preview(text, n=180):
    s = (text or "").replace("\n", " ").strip()
    if len(s) <= n:
        return s
    return s[: n - 3] + "..."


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=50)
    args = parser.parse_args()

    items = fetch_memory_items(["superseded", "archived", "uncertain"])

    print("=== SUPERSEDED / ARCHIVED ITEMS ===")
    print()

    if not items:
        print("NONE")
        close_pool()
        return

    for item in items[: args.limit]:
        print(f"id={item.get('id')}")
        print(f"status={item.get('status')}")
        print(
            f"entity={item.get('entity')} "
            f"property={item.get('property')} "
            f"value={item.get('value')}"
        )
        print(f"target_file={item.get('target_file')}")
        print(f"supersedes={item.get('supersedes')}")
        print(f"text={preview(item.get('text'))}")
        print()

    close_pool()


if __name__ == "__main__":
    main()
