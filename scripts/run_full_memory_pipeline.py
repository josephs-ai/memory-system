"""
Run the complete memory pipeline end-to-end — dehydrate, chunk, extract,
judge, route, embed, and reconcile in a single pass.
"""
import argparse
import subprocess
from pathlib import Path
from datetime import datetime, timezone

WORKSPACE = Path.home() / ".openclaw" / "workspace"
SCRIPTS_DIR = WORKSPACE / ".memory-index" / "scripts"
LOGS_DIR = WORKSPACE / ".memory-index" / "logs"

CHECKPOINT = SCRIPTS_DIR / "checkpoint_agent.py"
MAINTENANCE = SCRIPTS_DIR / "run_memory_maintenance_cycle.py"
SYNC_MARKDOWN = SCRIPTS_DIR / "sync_registry_to_markdown.py"

LOGS_DIR.mkdir(parents=True, exist_ok=True)


def now_iso():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def run_capture(cmd):
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return result.stdout.strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", required=True)
    parser.add_argument("--reason", default="full memory pipeline run")
    parser.add_argument("--force-maintenance", action="store_true")
    args = parser.parse_args()

    ts = now_iso().replace(":", "-")
    run_dir = LOGS_DIR / f"full-pipeline-{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_out = run_capture([
        "python3",
        str(CHECKPOINT),
        "--agent",
        args.agent,
        "--reason",
        args.reason,
    ])
    (run_dir / "checkpoint_output.txt").write_text(checkpoint_out + "\n", encoding="utf-8")

    if args.force_maintenance:
        maintenance_out = run_capture(["python3", str(MAINTENANCE), "--force"])
    else:
        maintenance_out = "SKIPPED: checkpoint_agent.py already runs maintenance during checkpoint flow"
    (run_dir / "maintenance_output.txt").write_text(maintenance_out + "\n", encoding="utf-8")

    sync_out = run_capture([
        "python3",
        str(SYNC_MARKDOWN),
    ])
    (run_dir / "sync_output.txt").write_text(sync_out + "\n", encoding="utf-8")

    print("FULL_MEMORY_PIPELINE_COMPLETE")
    print(f"log_dir={run_dir}")


if __name__ == "__main__":
    main()
