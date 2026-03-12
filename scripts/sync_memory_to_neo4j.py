from __future__ import annotations

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from memory_db import fetch_memory_items_for_embedding, close_pool
from graph_store_neo4j import ensure_memory_constraints, upsert_memory_graph


def main():
    ensure_memory_constraints()

    items = fetch_memory_items_for_embedding("active")
    if not items:
        print("NONE")
        close_pool()
        return

    synced = 0
    skipped = 0

    for item in items:
        if not item.get("entity") or not item.get("property") or item.get("value") is None:
            skipped += 1
            continue

        upsert_memory_graph(item)
        synced += 1

    print(f"SYNCED={synced}")
    print(f"SKIPPED={skipped}")

    close_pool()


if __name__ == "__main__":
    main()
