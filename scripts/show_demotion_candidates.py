"""
Display demotion candidates from the memory system for inspection and debugging.

Key functions: feedback_counts, preview, main
"""
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


def preview(text, n=180):
    s = (text or "").replace("\n", " ").strip()
    if len(s) <= n:
        return s
    return s[: n - 3] + "..."


def main():
    selected, useful, bad = feedback_counts()
    active_items = fetch_memory_items(["active"])

    print("=== DEMOTION CANDIDATES ===")
    print()

    found = False
    for item in active_items:
        iid = item.get("id")
        sel = selected[iid]
        use = useful[iid]
        b = bad[iid]

        if sel >= 2 and b >= 2 and b >= use:
            found = True
            print(f"id={iid}")
            print(f"selected={sel} useful={use} bad={b}")
            print(
                f"ranking_bonus={item.get('ranking_bonus', 0)} "
                f"ranking_penalty={item.get('ranking_penalty', 0)}"
            )
            print(
                f"entity={item.get('entity')} "
                f"property={item.get('property')} "
                f"value={item.get('value')}"
            )
            print(f"text={preview(item.get('text'))}")
            print()

    if not found:
        print("NONE")

    close_pool()


if __name__ == "__main__":
    main()
