from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

WORKSPACE = Path.home() / ".openclaw" / "workspace"
MEMORY_INDEX_DIR = WORKSPACE / ".memory-index"
CONTROL_PANEL_CONFIG = MEMORY_INDEX_DIR / "control_panel" / "control_panel_config.json"
STARTER = MEMORY_INDEX_DIR / "scripts" / "start_heartbeat_worker.py"
STATUS = MEMORY_INDEX_DIR / "scripts" / "heartbeat_worker_status.py"


def heartbeat_enabled() -> bool:
    if not CONTROL_PANEL_CONFIG.exists():
        return True
    try:
        cfg = json.loads(CONTROL_PANEL_CONFIG.read_text(encoding="utf-8"))
        return bool(cfg.get("heartbeat", {}).get("heartbeat_enabled", True))
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

def main():
    if not heartbeat_enabled():
        print(json.dumps({"status": "disabled_by_config"}, ensure_ascii=False))
        return

    status = run_json([sys.executable, str(STATUS)])
    if status.get("running"):
        print(json.dumps({"status": "already_running", "pid": status.get("pid")}, ensure_ascii=False))
        return

    result = run_json([sys.executable, str(STARTER)])
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
