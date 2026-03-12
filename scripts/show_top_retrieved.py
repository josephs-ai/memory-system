import argparse
import sys
from collections import Counter
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from memory_db import close_pool, fetch_memory_items, get_conn


def feedback_counts():
    selected = Counter()
    useful = Counter()
    bad = Counter()

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT selected_id, operator_feedback
                FROM retrieval_feedback
                WHERE selected_id IS NOT NULL
                ORDER BY feedback_id
                """
            )
            rows = cur.fetchall()

    for sid, fb in rows:
        selected[sid] += 1
        if fb == "useful":
            useful[sid] += 1
        elif fb == "bad":
            bad[sid] += 1

    return selected, useful, bad


def build_text_map():
    items = fetch_memory_items(["active", "superseded", "archived", "uncertain"])
    return {item.get("id"): item for item in items}


def preview(text, n=140):
    s = (text or "").replace("\n", " ").strip()
    if len(s) <= n:
        return s
    return s[: n - 3] + "..."


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()

    selected, useful, bad = feedback_counts()
    item_map = build_text_map()

    print("=== TOP RETRIEVED ITEMS ===")
    print()

    if not selected:
        print("NONE")
        close_pool()
        return

    for item_id, n in selected.most_common(args.limit):
        item = item_map.get(item_id)
        status = item.get("status") if item else "unknown"
        text = preview(item.get("text")) if item else ""
        print(
            f"{item_id}: selected={n} useful={useful[item_id]} "
            f"bad={bad[item_id]} status={status}"
        )
        if text:
            print(f"  text={text}")

    close_pool()


if __name__ == "__main__":
    main()
