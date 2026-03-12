import json
from pathlib import Path

from memory_db import (
    insert_retrieval_feedback,
    upsert_global_insights,
    close_pool,
)

WORKSPACE = Path.home() / ".openclaw" / "workspace"
MEMORY_DIR = WORKSPACE / "memory"
REVIEW_DIR = MEMORY_DIR / "review"
GLOBAL_INSIGHTS_DIR = MEMORY_DIR / "global-insights"


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


def migrate_feedback():
    path = REVIEW_DIR / "retrieval-feedback.jsonl"
    total = 0
    batches = 0
    print(f"MIGRATING_FEEDBACK file={path}")
    for batch in batched(iter_jsonl(path), 200):
        insert_retrieval_feedback(batch)
        total += len(batch)
        batches += 1
    print(f"FEEDBACK_DONE migrated={total} batches={batches}")
    return total, batches


def parse_insight_line(line: str):
    s = line.strip()
    if not s.startswith("- "):
        return None

    text = s[2:]
    source_project = None

    marker = " [source_project="
    if marker in text and text.endswith("]"):
        left, right = text.rsplit(marker, 1)
        source_project = right[:-1]
        text = left.strip()

    return {
        "text": text,
        "source_project": source_project,
    }


def migrate_global_insights():
    total = 0
    batches = 0
    rows = []

    if GLOBAL_INSIGHTS_DIR.exists():
        for path in sorted(GLOBAL_INSIGHTS_DIR.glob("*.md")):
            topic = path.stem
            for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
                parsed = parse_insight_line(line)
                if not parsed:
                    continue
                rows.append({
                    "topic": topic,
                    "text": parsed["text"],
                    "source_project": parsed["source_project"],
                    "promoted_by": "migration",
                })

    print(f"MIGRATING_GLOBAL_INSIGHTS rows={len(rows)}")
    for batch in batched(rows, 200):
        upsert_global_insights(batch)
        total += len(batch)
        batches += 1

    print(f"GLOBAL_INSIGHTS_DONE migrated={total} batches={batches}")
    return total, batches


def main():
    total = 0
    batches = 0

    n, b = migrate_feedback()
    total += n
    batches += b

    n, b = migrate_global_insights()
    total += n
    batches += b

    print(f"MIGRATED={total}")
    print(f"BATCHES={batches}")

    close_pool()


if __name__ == "__main__":
    main()
