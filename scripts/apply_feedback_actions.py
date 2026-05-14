"""
Apply Feedback Actions — utility for the OpenClaw memory system.

Key functions: now_iso, append_log, main
"""
import sys
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from memory_db import (
    all_feedback_item_ids,
    feedback_stats_db,
    fetch_memory_status_map,
    update_memory_item_ranking,
    close_pool,
)

WORKSPACE = Path.home() / ".openclaw" / "workspace"
DECISIONS_LOG = WORKSPACE / "memory" / "review" / "decisions.log"


def now_iso():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def append_log(line: str):
    with open(DECISIONS_LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def main():
    status_map = fetch_memory_status_map()
    ids = all_feedback_item_ids()

    changed = False
    ts = now_iso()

    for item_id in ids:
        useful, bad = feedback_stats_db(item_id)
        status, item = status_map.get(item_id, ("unknown", None))
        if item is None:
            continue

        current_bonus = float(item.get("ranking_bonus") or 0)
        current_penalty = float(item.get("ranking_penalty") or 0)

        if status == "active" and useful >= 2 and bad == 0:
            new_bonus = max(current_bonus, useful * 4)
            update_memory_item_ranking(
                item_id,
                ranking_bonus=new_bonus,
                feedback_last_applied_at=ts,
            )
            append_log(
                f"{ts} feedback_action=KEEP_AS_IS id={item_id} useful={useful} bad={bad} bonus={new_bonus}"
            )
            changed = True

        elif bad >= 2 and bad >= useful:
            new_penalty = max(current_penalty, bad * 40)
            update_memory_item_ranking(
                item_id,
                ranking_penalty=new_penalty,
                feedback_last_applied_at=ts,
            )
            append_log(
                f"{ts} feedback_action=DOWNRANK_CANDIDATE id={item_id} status={status} useful={useful} bad={bad} penalty={new_penalty}"
            )
            changed = True

    print("APPLIED" if changed else "NO_CHANGES")
    close_pool()


if __name__ == "__main__":
    main()
