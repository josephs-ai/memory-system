"""
Generate report on memory maintenance.

Key functions: parse_iso, age_days, same_slot, retrieval_stats
"""
import sys
from collections import Counter, defaultdict
from pathlib import Path
from datetime import datetime, timezone

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from memory_db import close_pool, fetch_memory_items, get_conn

WORKSPACE = Path.home() / ".openclaw" / "workspace"


def parse_iso(ts):
    if not ts:
        return None
    if isinstance(ts, datetime):
        return ts
    try:
        return datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
    except Exception:
        return None


def age_days(item):
    ts = parse_iso(item.get("last_confirmed"))
    if ts is None:
        return None
    return (datetime.now(timezone.utc) - ts).days


def same_slot(a, b):
    return (
        a.get("memory_type") == b.get("memory_type")
        and a.get("entity") == b.get("entity")
        and a.get("scope") == b.get("scope")
        and a.get("property") == b.get("property")
    )


def retrieval_stats():
    selected = Counter()
    considered = Counter()
    avg_score_sum = defaultdict(float)
    avg_score_n = defaultdict(int)

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT selected_id, payload
                FROM retrieval_feedback
                ORDER BY feedback_id
                """
            )
            rows = cur.fetchall()

    for selected_id, payload in rows:
        if selected_id:
            selected[selected_id] += 1

        payload = payload or {}
        for item in payload.get("considered", []):
            iid = item.get("id")
            if not iid:
                continue
            considered[iid] += 1
            avg_score_sum[iid] += float(item.get("score", 0) or 0)
            avg_score_n[iid] += 1

    avg_score = {}
    for iid in considered:
        avg_score[iid] = avg_score_sum[iid] / max(avg_score_n[iid], 1)

    return selected, considered, avg_score


def fetch_queue_payloads(table_name: str):
    allowed = {
        "memory_inbox",
        "memory_pending_stable",
        "memory_discarded",
    }
    if table_name not in allowed:
        raise ValueError(f"unsupported table: {table_name}")

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(f"SELECT payload FROM {table_name}")
            return [row[0] for row in cur.fetchall()]


def preview(text, n=180):
    s = (text or "").replace("\n", " ").strip()
    if len(s) <= n:
        return s
    return s[: n - 3] + "..."


def main():
    active = fetch_memory_items(["active"])
    non_active = fetch_memory_items(["superseded", "archived", "uncertain"])
    selected, considered, avg_score = retrieval_stats()

    print("=== MEMORY MAINTENANCE REPORT ===")
    print()

    print("Active items:", len(active))
    print("Non-active items:", len(non_active))
    print()

    print("=== STALE VOLATILE ITEMS ===")
    found = False
    for item in active:
        days = age_days(item)
        if item.get("freshness_class") == "volatile" and days is not None and days > 30:
            found = True
            print(f"- {item.get('id')} [{days}d old] {preview(item.get('text'))}")
    if not found:
        print("NONE")
    print()

    print("=== WEAKLY SPECIFIED ACTIVE ITEMS ===")
    found = False
    for item in active:
        if not item.get("property") or not item.get("value"):
            found = True
            print(
                f"- {item.get('id')} property={item.get('property')} "
                f"value={item.get('value')} :: {preview(item.get('text'))}"
            )
    if not found:
        print("NONE")
    print()

    print("=== POSSIBLE DUPLICATE ACTIVE ITEMS ===")
    found = False
    for i in range(len(active)):
        for j in range(i + 1, len(active)):
            a = active[i]
            b = active[j]
            if same_slot(a, b) and a.get("value") == b.get("value"):
                found = True
                print(f"- {a.get('id')} <-> {b.get('id')}")
                print(f"  slot=({a.get('entity')}, {a.get('property')}, {a.get('value')})")
    if not found:
        print("NONE")
    print()

    print("=== NON-ACTIVE ITEMS STILL FREQUENTLY CONSIDERED ===")
    found = False
    non_active_ids = {x.get("id") for x in non_active}
    for iid in sorted(non_active_ids):
        if considered[iid] > 0:
            found = True
            print(
                f"- {iid}: considered={considered[iid]} "
                f"selected={selected[iid]} avg_score={avg_score.get(iid, 0):.1f}"
            )
    if not found:
        print("NONE")
    print()

    print("=== ACTIVE ITEMS NEVER SELECTED ===")
    found = False
    for item in active:
        iid = item.get("id")
        if considered[iid] > 0 and selected[iid] == 0:
            found = True
            print(
                f"- {iid}: considered={considered[iid]} "
                f"avg_score={avg_score.get(iid, 0):.1f} :: {preview(item.get('text'))}"
            )
    if not found:
        print("NONE")
    print()

    print("=== TARGET FILE ROUTING CHECKS ===")
    found = False
    for item in active:
        entity = item.get("entity")
        target = item.get("target_file")
        mtype = item.get("memory_type")

        if entity == "browser" and target != "browser.md":
            found = True
            print(f"- {item.get('id')} browser item routed to {target}")

        if entity == "user_preference" and target != "preferences.md":
            found = True
            print(f"- {item.get('id')} preference item routed to {target}")

        if entity == "checkpoint_pipeline" and mtype == "decision" and target != "memory-system.md":
            found = True
            print(f"- {item.get('id')} checkpoint decision routed to {target}")

        if mtype == "learned_fix" and target != "learned-fixes.md":
            found = True
            print(f"- {item.get('id')} learned_fix routed to {target}")

    if not found:
        print("NONE")
    print()

    print("=== LIFECYCLE SANITY CHECKS ===")
    found = False
    for item in non_active:
        if item.get("freshness_class") == "permanent" and item.get("status") == "archived":
            found = True
            print(f"- {item.get('id')} permanent item is archived :: {preview(item.get('text'))}")
    if not found:
        print("NONE")
    print()

    print("=== PROMOTABLE QUEUE ITEMS ===")
    found = False
    inbox = fetch_queue_payloads("memory_inbox")
    pending = fetch_queue_payloads("memory_pending_stable")

    for item in inbox + pending:
        if (
            item.get("scope") == "stable"
            and float(item.get("confidence", 0) or 0) >= 0.85
            and float(item.get("importance", 0) or 0) >= 0.80
            and item.get("memory_type") in {"decision", "preference", "learned_fix", "fact"}
        ):
            found = True
            print(
                f"- {item.get('id')} "
                f"type={item.get('memory_type')} "
                f"entity={item.get('entity')} "
                f"property={item.get('property')} "
                f"confidence={item.get('confidence')} "
                f"importance={item.get('importance')} "
                f"text={preview(item.get('text'))}"
            )
    if not found:
        print("NONE")
    print()

    print("=== FILE-TOPIC MISMATCHES ===")
    found = False
    all_items = active + non_active

    for item in all_items:
        entity = item.get("entity")
        mtype = item.get("memory_type")
        target = item.get("target_file")

        expected = None
        if entity == "browser":
            expected = "browser.md"
        elif entity == "user_preference":
            expected = "preferences.md"
        elif entity == "checkpoint_pipeline" and mtype == "decision":
            expected = "memory-system.md"
        elif mtype == "learned_fix":
            expected = "learned-fixes.md"

        if expected and target != expected:
            found = True
            print(
                f"- {item.get('id')} target_file={target} expected={expected} "
                f"entity={entity} memory_type={mtype}"
            )

    if not found:
        print("NONE")
    print()

    close_pool()


if __name__ == "__main__":
    main()
