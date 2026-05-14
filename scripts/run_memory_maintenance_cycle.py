"""
Run periodic memory maintenance — compaction, reconciliation, embedding
backfills, and health checks.
"""
import argparse
import json
import subprocess
from pathlib import Path
from datetime import datetime, timezone

WORKSPACE = Path.home() / ".openclaw" / "workspace"
SCRIPTS_DIR = WORKSPACE / ".memory-index" / "scripts"
LOGS_DIR = WORKSPACE / ".memory-index" / "logs"
REVIEW_DIR = WORKSPACE / "memory" / "review"
DECISIONS_LOG = REVIEW_DIR / "decisions.log"
STATE_FILE = LOGS_DIR / "maintenance-cycle-state.json"

APPLY_FEEDBACK = SCRIPTS_DIR / "apply_feedback_actions.py"
REPORT_MAINT = SCRIPTS_DIR / "report_memory_maintenance.py"
REPORT_FEEDBACK = SCRIPTS_DIR / "report_feedback_actions.py"
SHOW_TOP = SCRIPTS_DIR / "show_top_retrieved.py"
SHOW_LOW = SCRIPTS_DIR / "show_low_utility_retrievals.py"
SHOW_DEMOTION = SCRIPTS_DIR / "show_demotion_candidates.py"
SHOW_PROMOTION = SCRIPTS_DIR / "show_promotion_candidates.py"

AUTO_PROMOTE = SCRIPTS_DIR / "auto_promote_safe_items.py"
SYNC_MARKDOWN = SCRIPTS_DIR / "sync_registry_to_markdown.py"

SHOW_CONFLICTS = SCRIPTS_DIR / "show_conflict_candidates.py"
CREATE_RESTORE_POINT = SCRIPTS_DIR / "create_memory_restore_point.py"

WATCH_FILES = [
    REVIEW_DIR / "retrieval-feedback.jsonl",
    REVIEW_DIR / "canonical-items.jsonl",
    REVIEW_DIR / "canonical-superseded.jsonl",
    REVIEW_DIR / "inbox.jsonl",
    REVIEW_DIR / "auto.jsonl",
    REVIEW_DIR / "discarded.jsonl",
]

LOGS_DIR.mkdir(parents=True, exist_ok=True)


def now_iso():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def run_capture(cmd):
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return result.stdout.strip()


def append_log(line: str):
    with open(DECISIONS_LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def file_signature(path: Path):
    if not path.exists():
        return {"exists": False, "mtime_ns": 0, "size": 0}
    stat = path.stat()
    return {
        "exists": True,
        "mtime_ns": stat.st_mtime_ns,
        "size": stat.st_size,
    }


def current_state():
    return {str(path): file_signature(path) for path in WATCH_FILES}


def load_previous_state():
    if not STATE_FILE.exists():
        return None
    try:
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except Exception:
        return None


def save_state(state: dict):
    STATE_FILE.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    current = current_state()
    previous = load_previous_state()

    if previous == current and not args.force:
        append_log(f"{now_iso()} maintenance_cycle skipped reason=no-meaningful-change")
        print("MAINTENANCE_CYCLE_SKIPPED")
        print("reason=no-meaningful-change")
        return

    ts = now_iso().replace(":", "-")
    cycle_dir = LOGS_DIR / f"maintenance-cycle-{ts}"
    cycle_dir.mkdir(parents=True, exist_ok=True)

    restore_out = run_capture([
        "python3",
        str(CREATE_RESTORE_POINT),
        "--label",
        "pre-maintenance",
    ])
    (cycle_dir / "restore_point_output.txt").write_text(restore_out + "\n", encoding="utf-8")


    apply_out = run_capture(["python3", str(APPLY_FEEDBACK)])
    (cycle_dir / "apply_feedback_actions.txt").write_text(apply_out + "\n", encoding="utf-8")

    promote_out = run_capture(["python3", str(AUTO_PROMOTE)])
    (cycle_dir / "auto_promote_safe_items.txt").write_text(promote_out + "\n", encoding="utf-8")

    sync_out = run_capture(["python3", str(SYNC_MARKDOWN)])
    (cycle_dir / "sync_registry_to_markdown.txt").write_text(sync_out + "\n", encoding="utf-8")

    maint_out = run_capture(["python3", str(REPORT_MAINT)])
    (cycle_dir / "memory_maintenance_report.txt").write_text(maint_out + "\n", encoding="utf-8")

    feedback_out = run_capture(["python3", str(REPORT_FEEDBACK)])
    (cycle_dir / "feedback_actions_report.txt").write_text(feedback_out + "\n", encoding="utf-8")

    top_out = run_capture(["python3", str(SHOW_TOP)])
    (cycle_dir / "top_retrieved.txt").write_text(top_out + "\n", encoding="utf-8")

    low_out = run_capture(["python3", str(SHOW_LOW)])
    (cycle_dir / "low_utility_retrievals.txt").write_text(low_out + "\n", encoding="utf-8")

    demotion_out = run_capture(["python3", str(SHOW_DEMOTION)])
    (cycle_dir / "demotion_candidates.txt").write_text(demotion_out + "\n", encoding="utf-8")

    promotion_out = run_capture(["python3", str(SHOW_PROMOTION)])
    (cycle_dir / "promotion_candidates.txt").write_text(promotion_out + "\n", encoding="utf-8")

    conflict_out = run_capture(["python3", str(SHOW_CONFLICTS)])
    (cycle_dir / "conflict_candidates.txt").write_text(conflict_out + "\n", encoding="utf-8")

    save_state(current_state())

    append_log(f"{now_iso()} maintenance_cycle completed dir={cycle_dir}")

    if args.force:
        print("mode=forced")
    print("MAINTENANCE_CYCLE_COMPLETE")
    print(f"log_dir={cycle_dir}")


if __name__ == "__main__":
    main()
