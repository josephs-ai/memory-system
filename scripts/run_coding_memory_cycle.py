"""
Run the coding memory cycle pipeline.

Key functions: now_iso, run_capture, main
"""
import argparse
import subprocess
import sys
from pathlib import Path
from datetime import datetime, timezone
from project_lock import acquire_project_lock, release_project_lock

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from project_memory_paths import (
    get_project_snapshot_file,
    get_project_summary_file,
)

WORKSPACE = Path.home() / ".openclaw" / "workspace"
SCRIPTS_DIR = WORKSPACE / ".memory-index" / "scripts"
LOGS_DIR = WORKSPACE / ".memory-index" / "logs"

SNAPSHOT = SCRIPTS_DIR / "snapshot_workspace_state.py"
SUMMARIZE = SCRIPTS_DIR / "summarize_workspace_changes.py"
INGEST = SCRIPTS_DIR / "ingest_workspace_state_memory.py"
FULL_PIPELINE = SCRIPTS_DIR / "run_full_memory_pipeline.py"

LOGS_DIR.mkdir(parents=True, exist_ok=True)


def now_iso():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def run_capture(cmd):
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return result.stdout.strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", required=True)
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--project-id", required=True)
    parser.add_argument("--reason", default="coding workflow completion")
    parser.add_argument("--force-maintenance", action="store_true")
    args = parser.parse_args()

    lock_owner = f"{args.agent}:{args.project_id}"
    acquired, current_owner = acquire_project_lock(args.project_id, lock_owner)

    if not acquired:
        print("LOCKED")
        print(f"project_id={args.project_id}")
        print(f"owner={current_owner}")
        return

    try:
        ts = now_iso().replace(":", "-")
        run_dir = LOGS_DIR / f"coding-memory-cycle-{ts}"
        run_dir.mkdir(parents=True, exist_ok=True)

        snapshot_file = get_project_snapshot_file(args.project_id)
        summary_file = get_project_summary_file(args.project_id)

        snapshot_out = run_capture([
            "python3",
            str(SNAPSHOT),
            "--workspace",
            args.workspace,
            "--out",
            str(snapshot_file),
        ])
        (run_dir / "snapshot_output.txt").write_text(snapshot_out + "\n", encoding="utf-8")

        summarize_out = run_capture([
            "python3",
            str(SUMMARIZE),
            "--snapshot",
            str(snapshot_file),
        ])
        summary_file.write_text(summarize_out + "\n", encoding="utf-8")
        (run_dir / "summarize_output.json").write_text(summarize_out + "\n", encoding="utf-8")

        ingest_out = run_capture([
            "python3",
            str(INGEST),
            "--summary",
            str(summary_file),
            "--project-id",
            args.project_id,
        ])
        (run_dir / "ingest_output.txt").write_text(ingest_out + "\n", encoding="utf-8")

        pipeline_cmd = [
            "python3",
            str(FULL_PIPELINE),
            "--agent",
            args.agent,
            "--reason",
            args.reason,
        ]
        if args.force_maintenance:
            pipeline_cmd.append("--force-maintenance")

        pipeline_out = run_capture(pipeline_cmd)
        (run_dir / "full_pipeline_output.txt").write_text(pipeline_out + "\n", encoding="utf-8")

        print("CODING_MEMORY_CYCLE_COMPLETE")
        print(f"project_id={args.project_id}")
        print(f"log_dir={run_dir}")
        print(f"snapshot_file={snapshot_file}")
        print(f"summary_file={summary_file}")

    finally:
        release_project_lock(args.project_id, lock_owner)

    ts = now_iso().replace(":", "-")
    run_dir = LOGS_DIR / f"coding-memory-cycle-{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)

    snapshot_file = get_project_snapshot_file(args.project_id)
    summary_file = get_project_summary_file(args.project_id)

    snapshot_out = run_capture([
        "python3",
        str(SNAPSHOT),
        "--workspace",
        args.workspace,
        "--out",
        str(snapshot_file),
    ])
    (run_dir / "snapshot_output.txt").write_text(snapshot_out + "\n", encoding="utf-8")

    summarize_out = run_capture([
        "python3",
        str(SUMMARIZE),
        "--snapshot",
        str(snapshot_file),
    ])
    summary_file.write_text(summarize_out + "\n", encoding="utf-8")
    (run_dir / "summarize_output.json").write_text(summarize_out + "\n", encoding="utf-8")

    ingest_out = run_capture([
        "python3",
        str(INGEST),
        "--summary",
        str(summary_file),
        "--project-id",
        args.project_id,
    ])
    (run_dir / "ingest_output.txt").write_text(ingest_out + "\n", encoding="utf-8")

    pipeline_cmd = [
        "python3",
        str(FULL_PIPELINE),
        "--agent",
        args.agent,
        "--reason",
        args.reason,
    ]
    if args.force_maintenance:
        pipeline_cmd.append("--force-maintenance")

    pipeline_out = run_capture(pipeline_cmd)
    (run_dir / "full_pipeline_output.txt").write_text(pipeline_out + "\n", encoding="utf-8")



if __name__ == "__main__":
    main()
