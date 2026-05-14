"""
Runtime orchestrator for the memory pipeline — manages process lifecycle,
error handling, and graceful shutdown.
"""
from __future__ import annotations

import subprocess
from datetime import datetime, timezone
from pathlib import Path

WORKSPACE = Path.home() / ".openclaw" / "workspace"
SCRIPTS_DIR = WORKSPACE / ".memory-index" / "scripts"
LOGS_DIR = WORKSPACE / ".memory-index" / "logs"

CHECKPOINT = SCRIPTS_DIR / "checkpoint_agent.py"
MAINTENANCE = SCRIPTS_DIR / "run_memory_maintenance_cycle.py"
SYNC_MARKDOWN = SCRIPTS_DIR / "sync_registry_to_markdown.py"

LOGS_DIR.mkdir(parents=True, exist_ok=True)


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def run_capture(cmd: list[str]) -> str:
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return result.stdout.strip()


def run_memory_pipeline(
    *,
    agent: str,
    reason: str,
    force_maintenance: bool = False,
    mode: str,
    log_prefix: str,
) -> Path:
    ts = now_iso().replace(":", "-")
    run_dir = LOGS_DIR / f"{log_prefix}-{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_out = run_capture(
        [
            "python3",
            str(CHECKPOINT),
            "--agent",
            agent,
            "--reason",
            reason,
        ]
    )
    (run_dir / "checkpoint_output.txt").write_text(checkpoint_out + "\n", encoding="utf-8")

    maintenance_cmd = ["python3", str(MAINTENANCE)]
    if force_maintenance:
        maintenance_cmd.append("--force")
    maintenance_out = run_capture(maintenance_cmd)
    (run_dir / "maintenance_output.txt").write_text(maintenance_out + "\n", encoding="utf-8")

    sync_out = run_capture(["python3", str(SYNC_MARKDOWN)])
    (run_dir / "sync_output.txt").write_text(sync_out + "\n", encoding="utf-8")

    (run_dir / "pipeline_mode.txt").write_text(mode + "\n", encoding="utf-8")
    return run_dir
