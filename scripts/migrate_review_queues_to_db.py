"""
Data migration: review queues to db.

Key functions: iter_jsonl, batched, migrate_one, main
"""
import json
from pathlib import Path

from memory_db import (
    upsert_pending_stable,
    upsert_inbox,
    upsert_discarded,
    close_pool,
)

WORKSPACE = Path.home() / ".openclaw" / "workspace"
REVIEW_DIR = WORKSPACE / "memory" / "review"


def iter_jsonl(path: Path):
    if not path.exists():
        return
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for lineno, line in enumerate(f, start=1):
            s = line.strip()
            if not s:
                continue
            try:
                yield json.loads(s)
            except Exception as e:
                print(f"PARSE_ERROR file={path} line={lineno} error={e}")


def batched(iterable, size: int):
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


def migrate_one(path: Path, upsert_fn, label: str):
    total = 0
    batches = 0
    print(f"MIGRATING_QUEUE label={label} file={path}")
    for batch in batched(iter_jsonl(path), 200):
        upsert_fn(batch)
        total += len(batch)
        batches += 1
    print(f"QUEUE_DONE label={label} migrated={total} batches={batches}")
    return total, batches


def main():
    total = 0
    batches = 0

    n, b = migrate_one(REVIEW_DIR / "pending-stable.jsonl", upsert_pending_stable, "pending_stable")
    total += n
    batches += b

    n, b = migrate_one(REVIEW_DIR / "inbox.jsonl", upsert_inbox, "inbox")
    total += n
    batches += b

    n, b = migrate_one(REVIEW_DIR / "discarded.jsonl", upsert_discarded, "discarded")
    total += n
    batches += b

    print(f"MIGRATED={total}")
    print(f"BATCHES={batches}")

    close_pool()


if __name__ == "__main__":
    main()
