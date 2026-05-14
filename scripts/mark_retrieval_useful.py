import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from memory_db import insert_retrieval_feedback_row, close_pool

WORKSPACE = Path.home() / ".openclaw" / "workspace"
REVIEW_DIR = WORKSPACE / "memory" / "review"
DECISIONS_LOG = REVIEW_DIR / "decisions.log"


def now_iso():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def append_log(line: str):
    with open(DECISIONS_LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--selected-id", required=True)
    parser.add_argument("--note", default="")
    parser.add_argument("--target-file", default=None)
    parser.add_argument("--query-text", default=None,
                        help="The query that produced this retrieval result")
    parser.add_argument("--query-type", default=None,
                        help="Query type: code|memory|timeline|general")
    args = parser.parse_args()

    # Auto-classify query type if query_text provided but query_type not
    query_type = args.query_type
    if args.query_text and not query_type:
        try:
            from feedback_score_engine import classify_query_type
            query_type = classify_query_type(args.query_text)
        except Exception:
            pass

    row = {
        "selected_id": args.selected_id,
        "selected_score": None,
        "selected_status": None,
        "operator_feedback": "useful",
        "feedback_note": args.note,
        "feedback_at": now_iso(),
        "candidate_file": None,
        "target_file": args.target_file,
        "include_superseded": None,
        "query_text": args.query_text,
        "query_type": query_type,
    }

    insert_retrieval_feedback_row(row)

    append_log(
        f"{now_iso()} feedback_mark_useful id={args.selected_id}"
        + (f" note={args.note}" if args.note else "")
    )

    print("MARKED_USEFUL")
    close_pool()


if __name__ == "__main__":
    main()
