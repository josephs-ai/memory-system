import json
import sys
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from memory_db import (
    upsert_inbox,
    upsert_discarded,
    upsert_memory_item,
    queue_item_exists_anywhere,
    candidate_matches_rejected,
    close_pool,
)

WORKSPACE = Path.home() / ".openclaw" / "workspace"
REVIEW_DIR = WORKSPACE / "memory" / "review"
DECISIONS_LOG = REVIEW_DIR / "decisions.log"

AUTO_THRESHOLD = 0.90
STABLE_AUTO_THRESHOLD = 0.93
INBOX_THRESHOLD = 0.65

def now_iso():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def append_log(line: str):
    with open(DECISIONS_LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def main():
    raw = sys.stdin.read().strip()
    if not raw or raw == "NONE":
        print("NONE")
        return

    item = json.loads(raw)

    confidence = float(item.get("confidence", 0.0) or 0.0)
    importance = float(item.get("importance", 0.0) or 0.0)
    scope = item.get("scope", "stable")
    item_id = item.get("id")
    now = now_iso()

    entity = (item.get("entity") or "").strip()
    prop = (item.get("property") or "").strip()
    value = (item.get("value") or "").strip()
    text = (item.get("text") or "").strip()
    memory_type = (item.get("memory_type") or "").strip().lower()

    has_full_structure = bool(entity and prop and value)
    text_word_count = len(text.split())
    looks_atomic = bool(text) and text_word_count <= 30

    stable_safe_auto = (
        scope == "stable"
        and confidence >= STABLE_AUTO_THRESHOLD
        and importance >= 0.90
        and has_full_structure
        and looks_atomic
        and memory_type in {"fact", "decision", "preference"}
    )

    if item_id and queue_item_exists_anywhere(item_id):
        append_log(f"{now} route=SKIP_DUPLICATE id={item_id} target=db_queue")
        print("SKIP_DUPLICATE db")
        close_pool()
        return

    matched_rejected, rejected_item = candidate_matches_rejected(item)
    if matched_rejected:
        upsert_discarded([item])
        append_log(
            f"{now} route=DISCARDED id={item_id} target=memory_discarded "
            f"reason=previously_rejected_memory "
            f"matched_rejected_id={rejected_item.get('id') if rejected_item else 'unknown'}"
        )
        print("DISCARDED db:memory_discarded")
        close_pool()
        return

    if stable_safe_auto:
        upsert_memory_item(item)
        append_log(
            f"{now} route=AUTO id={item_id} confidence={confidence:.2f} "
            f"importance={importance:.2f} scope={scope} target=memory_items reason=stable_safe_auto"
        )
        print("AUTO db:memory_items")
    elif scope == "stable" and confidence >= AUTO_THRESHOLD and importance >= 0.90:
        upsert_inbox([item])
        append_log(
            f"{now} route=INBOX id={item_id} confidence={confidence:.2f} "
            f"importance={importance:.2f} scope={scope} target=memory_inbox reason=stable_needs_review"
        )
        print("INBOX db:memory_inbox")
    elif confidence >= AUTO_THRESHOLD:
        upsert_memory_item(item)
        append_log(f"{now} route=AUTO id={item_id} confidence={confidence:.2f} target=memory_items")
        print("AUTO db:memory_items")
    elif confidence >= INBOX_THRESHOLD:
        upsert_inbox([item])
        append_log(f"{now} route=INBOX id={item_id} confidence={confidence:.2f} target=memory_inbox")
        print("INBOX db:memory_inbox")
    else:
        upsert_discarded([item])
        append_log(f"{now} route=DISCARDED id={item_id} confidence={confidence:.2f} target=memory_discarded")
        print("DISCARDED db:memory_discarded")

    close_pool()


if __name__ == "__main__":
    main()
