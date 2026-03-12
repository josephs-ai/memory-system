import json
from pathlib import Path

from memory_db import upsert_memory_items, count_memory_items, close_pool

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


def main():
    files = [
        REVIEW_DIR / "canonical-items.jsonl",
        REVIEW_DIR / "canonical-superseded.jsonl",
    ]

    total_migrated = 0
    total_batches = 0

    for path in files:
        file_count = 0
        print(f"MIGRATING file={path}")

        for batch in batched(iter_jsonl(path), 200):
            upsert_memory_items(batch)
            file_count += len(batch)
            total_migrated += len(batch)
            total_batches += 1

        print(f"FILE_DONE file={path} migrated={file_count}")

    print(f"MIGRATED={total_migrated}")
    print(f"BATCHES={total_batches}")
    print(f"DB_COUNT={count_memory_items()}")
    close_pool()

if __name__ == "__main__":
    main()
