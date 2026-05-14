"""
Generate report on feedback actions.

Key functions: build_status_map, feedback_rows, preview, main
"""
import sys
from collections import Counter, defaultdict
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from memory_db import close_pool, fetch_memory_items, get_conn


def build_status_map():
    status = {}
    items = fetch_memory_items(["active", "superseded", "archived", "uncertain"])
    for item in items:
        status[item.get("id")] = (item.get("status", "unknown"), item)
    return status


def feedback_rows():
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT selected_id, operator_feedback, payload
                FROM retrieval_feedback
                ORDER BY feedback_id
                """
            )
            return cur.fetchall()


def preview(text, n=140):
    s = (text or "").replace("\n", " ").strip()
    if len(s) <= n:
        return s
    return s[: n - 3] + "..."


def main():
    rows = feedback_rows()
    status_map = build_status_map()

    selected = Counter()
    useful = Counter()
    bad = Counter()
    considered = Counter()
    avg_score_sum = defaultdict(float)
    avg_score_n = defaultdict(int)

    for sid, fb, payload in rows:
        if sid:
            selected[sid] += 1
            if fb == "useful":
                useful[sid] += 1
            elif fb == "bad":
                bad[sid] += 1

        payload = payload or {}
        for item in payload.get("considered", []):
            iid = item.get("id")
            if not iid:
                continue
            considered[iid] += 1
            avg_score_sum[iid] += float(item.get("score", 0) or 0)
            avg_score_n[iid] += 1

    print("=== FEEDBACK-DRIVEN ACTION SUGGESTIONS ===")
    print()

    found = False
    all_ids = sorted(set(list(selected.keys()) + list(considered.keys())))

    for iid in all_ids:
        sel = selected[iid]
        use = useful[iid]
        b = bad[iid]
        cons = considered[iid]
        avg = avg_score_sum[iid] / max(avg_score_n[iid], 1)
        status, item = status_map.get(iid, ("unknown", None))

        freshness_class = item.get("freshness_class") if item else None
        text = preview(item.get("text")) if item else ""

        if status == "active" and sel >= 2 and b >= 2 and b >= use:
            found = True
            print(f"DEMOTE_CANDIDATE id={iid} status={status} selected={sel} useful={use} bad={b} avg_score={avg:.1f} text={text}")

        elif sel >= 1 and b >= 2 and b >= use:
            found = True
            print(f"DOWNRANK_CANDIDATE id={iid} status={status} selected={sel} useful={use} bad={b} avg_score={avg:.1f} text={text}")

        if status in {"superseded", "archived"} and cons >= 2 and sel == 0:
            found = True
            print(f"ARCHIVE_CANDIDATE id={iid} status={status} considered={cons} selected={sel} avg_score={avg:.1f} text={text}")

        if status == "active" and freshness_class == "volatile" and sel == 0 and use == 0 and cons == 0:
            found = True
            print(f"ARCHIVE_CANDIDATE id={iid} status={status} freshness_class={freshness_class} selected={sel} useful={use} bad={b} text={text}")

        if status == "active" and sel >= 2 and use >= 2 and b == 0:
            found = True
            print(f"KEEP_AS_IS id={iid} status={status} selected={sel} useful={use} bad={b} text={text}")

    if not found:
        print("NONE")

    close_pool()


if __name__ == "__main__":
    main()
