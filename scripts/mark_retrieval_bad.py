import argparse
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
    args = parser.parse_args()

    row = {
        "selected_id": args.selected_id,
        "selected_score": None,
        "selected_status": None,
        "operator_feedback": "bad",
        "feedback_note": args.note,
        "feedback_at": now_iso(),
        "candidate_file": None,
        "target_file": args.target_file,
        "include_superseded": None,
    }

    insert_retrieval_feedback_row(row)

    append_log(
        f"{now_iso()} feedback_mark_bad id={args.selected_id}"
        + (f" note={args.note}" if args.note else "")
    )

    print("MARKED_BAD")
    close_pool()


if __name__ == "__main__":
    main()
