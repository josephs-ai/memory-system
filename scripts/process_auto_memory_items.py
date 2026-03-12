import json
import subprocess
import sys
from pathlib import Path
from datetime import datetime, timezone

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from memory_db import close_pool, get_conn

WORKSPACE = Path.home() / ".openclaw" / "workspace"
REVIEW_DIR = WORKSPACE / "memory" / "review"
SCRIPTS_DIR = WORKSPACE / ".memory-index" / "scripts"

DECISIONS_LOG = REVIEW_DIR / "decisions.log"
WRITE_ITEM_SCRIPT = SCRIPTS_DIR / "write_memory_item.py"


def now_iso():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def append_log(line: str):
    with open(DECISIONS_LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def fetch_auto_items():
    # AUTO items are memory_items rows whose suggested_route is AUTO but are not yet finalized.
    # We treat active rows with suggested_route='auto' and no target_file writeback requirement as candidates.
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    id, text, memory_type, scope, entity, property, value, status,
                    confidence, importance, freshness_class,
                    source_agent, source_session, source_chunk,
                    first_seen, last_confirmed, supersedes,
                    tags, notes, candidate_id, candidate_score, candidate_reasons,
                    suggested_route, target_file, target_section,
                    sensitivity, approved_at, approved_by, approval_source,
                    rejected_at, rejected_by, rejection_reason, rejection_source,
                    ranking_bonus, ranking_penalty, feedback_last_applied_at
                FROM memory_items
                WHERE suggested_route = 'auto'
                  AND status = 'active'
                ORDER BY id
                """
            )
            cols = [d[0] for d in cur.description]
            rows = []
            for rec in cur.fetchall():
                row = dict(zip(cols, rec))
                row["tags"] = row.get("tags") or []
                row["candidate_reasons"] = row.get("candidate_reasons") or []
                rows.append(row)
            return rows


def clear_auto_flag(item_id: str):
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE memory_items
                SET suggested_route = NULL, updated_at = now()
                WHERE id = %s
                """,
                (item_id,),
            )
        conn.commit()


def write_temp_item(item: dict, idx: int) -> Path:
    tmp = REVIEW_DIR / f".auto_process_{idx}.json"
    tmp.write_text(json.dumps(item, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return tmp


def main():
    items = fetch_auto_items()
    if not items:
        print("No AUTO items to process.")
        close_pool()
        return

    print(f"AUTO items to process: {len(items)}")

    remaining_count = 0

    for idx, item in enumerate(items, start=1):
        item_id = item.get("id", f"unknown-{idx}")
        tmp = write_temp_item(item, idx)

        try:
            result = subprocess.run(
                [
                    "python3",
                    str(WRITE_ITEM_SCRIPT),
                    "--candidate-item",
                    str(tmp),
                ],
                capture_output=True,
                text=True,
                check=True,
            )

            out = result.stdout.strip()
            print(f"===== ITEM {idx} id={item_id} =====")
            print(out)
            print()

            append_log(f"{now_iso()} process_auto id={item_id}")

            # Successful handling clears AUTO route so it won't be reprocessed.
            if out:
                clear_auto_flag(item_id)
            else:
                remaining_count += 1

        finally:
            try:
                tmp.unlink()
            except Exception:
                pass

    print(f"Remaining AUTO items: {remaining_count}")
    close_pool()


if __name__ == "__main__":
    main()
