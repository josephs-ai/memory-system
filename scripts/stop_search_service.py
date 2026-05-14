"""
Stop the search service service/daemon.

Key functions: pid_alive, main
"""
from __future__ import annotations

import json
import os
import signal
import time
from pathlib import Path

WORKSPACE = Path.home() / ".openclaw" / "workspace"
RUNTIME = WORKSPACE / ".memory-index" / "runtime"
PID_FILE = RUNTIME / "search_service.pid"
META_FILE = RUNTIME / "search_service.json"


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def main() -> None:
    if not PID_FILE.exists():
        print(json.dumps({"status": "not_running"}))
        return

    try:
        pid = int(PID_FILE.read_text(encoding="utf-8").strip())
    except Exception:
        PID_FILE.unlink(missing_ok=True)
        META_FILE.unlink(missing_ok=True)
        print(json.dumps({"status": "stale_pid_file"}))
        return

    if not pid_alive(pid):
        PID_FILE.unlink(missing_ok=True)
        META_FILE.unlink(missing_ok=True)
        print(json.dumps({"status": "already_dead", "pid": pid}))
        return

    try:
        os.kill(pid, signal.SIGTERM)
        deadline = time.time() + 8
        while time.time() < deadline:
            if not pid_alive(pid):
                PID_FILE.unlink(missing_ok=True)
                META_FILE.unlink(missing_ok=True)
                print(json.dumps({"status": "stopped", "pid": pid}))
                return
            time.sleep(0.25)

        os.kill(pid, signal.SIGKILL)
        PID_FILE.unlink(missing_ok=True)
        META_FILE.unlink(missing_ok=True)
        print(json.dumps({"status": "killed", "pid": pid}))
    except ProcessLookupError:
        PID_FILE.unlink(missing_ok=True)
        META_FILE.unlink(missing_ok=True)
        print(json.dumps({"status": "already_dead", "pid": pid}))


if __name__ == "__main__":
    main()
