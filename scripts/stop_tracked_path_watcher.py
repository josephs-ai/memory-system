from __future__ import annotations

import json
import signal
import time

from heartbeat_manager_common import pid_alive, terminate_process_group

from pathlib import Path

WORKSPACE = Path.home() / ".openclaw" / "workspace"
RUNTIME_DIR = WORKSPACE / ".memory-index" / "runtime"
PID_FILE = RUNTIME_DIR / "tracked_path_watcher.pid"
META_FILE = RUNTIME_DIR / "tracked_path_watcher.json"


def read_pid() -> int | None:
    if not PID_FILE.exists():
        return None
    try:
        return int(PID_FILE.read_text(encoding="utf-8").strip())
    except Exception:
        return None


def cleanup() -> None:
    PID_FILE.unlink(missing_ok=True)
    META_FILE.unlink(missing_ok=True)


def main():
    pid = read_pid()
    if not pid:
        print(json.dumps({"status": "not_running"}, ensure_ascii=False))
        return

    if not pid_alive(pid):
        cleanup()
        print(json.dumps({"status": "already_dead", "pid": pid}, ensure_ascii=False))
        return

    terminate_process_group(pid, signal.SIGTERM)

    deadline = time.time() + 8
    while time.time() < deadline:
        if not pid_alive(pid):
            cleanup()
            print(json.dumps({"status": "stopped", "pid": pid}, ensure_ascii=False))
            return
        time.sleep(0.25)

    terminate_process_group(pid, signal.SIGKILL)

    deadline = time.time() + 3
    while time.time() < deadline:
        if not pid_alive(pid):
            cleanup()
            print(json.dumps({"status": "killed", "pid": pid}, ensure_ascii=False))
            return
        time.sleep(0.25)

    print(json.dumps({"status": "still_running", "pid": pid}, ensure_ascii=False))


if __name__ == "__main__":
    main()
