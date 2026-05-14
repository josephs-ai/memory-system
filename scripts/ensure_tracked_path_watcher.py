"""
Ensure the tracked path watcher is running, starting it if needed.

Key functions: watcher_enabled, run_json, status_ok, main
"""
from __future__ import annotations

import json
import subprocess
import sys

from heartbeat_manager_common import MEMORY_INDEX_DIR

CONTROL_PANEL_CONFIG = MEMORY_INDEX_DIR / "control_panel" / "control_panel_config.json"
STARTER = MEMORY_INDEX_DIR / "scripts" / "start_tracked_path_watcher.py"
STATUS = MEMORY_INDEX_DIR / "scripts" / "tracked_path_watcher_status.py"


def watcher_enabled() -> bool:
    if not CONTROL_PANEL_CONFIG.exists():
        return True
    try:
        cfg = json.loads(CONTROL_PANEL_CONFIG.read_text(encoding="utf-8"))
        hb = cfg.get("heartbeat", {})
        return bool(hb.get("heartbeat_enabled", True))
    except Exception:
        return True


def run_json(cmd: list[str]) -> dict:
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=20,
        )
    except FileNotFoundError as e:
        return {
            "status": "exec_not_found",
            "error": str(e),
            "cmd": cmd,
        }
    except subprocess.TimeoutExpired as e:
        return {
            "status": "timeout",
            "error": str(e),
            "cmd": cmd,
        }
    except OSError as e:
        return {
            "status": "exec_os_error",
            "error": str(e),
            "cmd": cmd,
        }
    except Exception as e:
        return {
            "status": "exec_exception",
            "error": str(e),
            "cmd": cmd,
        }

    if proc.returncode != 0:
        return {
            "status": "error",
            "returncode": proc.returncode,
            "stderr": proc.stderr,
            "stdout": proc.stdout,
            "cmd": cmd,
        }

    try:
        return json.loads(proc.stdout)
    except Exception:
        return {
            "status": "invalid_json",
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "returncode": proc.returncode,
            "cmd": cmd,
        }


def status_ok(status: dict) -> bool:
    return isinstance(status, dict) and status.get("status") not in {
        "exec_not_found",
        "timeout",
        "exec_os_error",
        "exec_exception",
        "error",
        "invalid_json",
    }


def main():
    if not watcher_enabled():
        print(json.dumps({"status": "disabled_by_config"}, ensure_ascii=False))
        return

    status = run_json([sys.executable, str(STATUS)])

    # Enterprise rule:
    # if status retrieval itself failed, do not blindly try to start;
    # surface the status problem first.
    if not status_ok(status):
        print(json.dumps({
            "status": "status_check_failed",
            "status_result": status,
        }, ensure_ascii=False, indent=2))
        return

    if status.get("running"):
        print(json.dumps({
            "status": "already_running",
            "pid": status.get("pid"),
            "restart_recommended": status.get("restart_recommended"),
        }, ensure_ascii=False))
        return

    result = run_json([sys.executable, str(STARTER)])

    print(json.dumps({
        "status": "start_attempted",
        "prior_status": status,
        "start_result": result,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
