from __future__ import annotations

import json
import subprocess
import sys
import time

from heartbeat_manager_common import (
    MEMORY_INDEX_DIR,
    atomic_write_json,
    atomic_write_text,
    build_runtime_env,
    ensure_runtime_dir,
    pid_alive,
    read_json,
)

RUNTIME_DIR = MEMORY_INDEX_DIR / "runtime"
PID_FILE = RUNTIME_DIR / "tracked_path_watcher.pid"
META_FILE = RUNTIME_DIR / "tracked_path_watcher.json"
LOG_FILE = MEMORY_INDEX_DIR / "tracked_path_watcher.log"
WATCHER_SCRIPT = MEMORY_INDEX_DIR / "scripts" / "tracked_path_watcher.py"


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
    ensure_runtime_dir()

    if not WATCHER_SCRIPT.exists():
        print(json.dumps({
            "status": "missing_watcher_script",
            "watcher_script": str(WATCHER_SCRIPT),
        }, ensure_ascii=False))
        return

    existing_pid = read_pid()
    if existing_pid and pid_alive(existing_pid):
        meta = read_json(META_FILE)
        print(json.dumps({
            "status": "already_running",
            "pid": existing_pid,
            "log_file": str(LOG_FILE),
            "started_at": meta.get("started_at"),
        }, ensure_ascii=False))
        return

    if existing_pid and not pid_alive(existing_pid):
        cleanup()

    env = build_runtime_env()

    with open(LOG_FILE, "a", encoding="utf-8") as logf:
        try:
            proc = subprocess.Popen(
                [sys.executable, str(WATCHER_SCRIPT)],
                stdin=subprocess.DEVNULL,
                stdout=logf,
                stderr=logf,
                env=env,
                start_new_session=True,
            )
        except FileNotFoundError as e:
            print(json.dumps({
                "status": "exec_not_found",
                "error": str(e),
                "watcher_script": str(WATCHER_SCRIPT),
            }, ensure_ascii=False))
            return
        except OSError as e:
            print(json.dumps({
                "status": "exec_os_error",
                "error": str(e),
                "watcher_script": str(WATCHER_SCRIPT),
            }, ensure_ascii=False))
            return
        except Exception as e:
            print(json.dumps({
                "status": "exec_exception",
                "error": str(e),
                "watcher_script": str(WATCHER_SCRIPT),
            }, ensure_ascii=False))
            return

    time.sleep(0.75)
    if proc.poll() is not None:
        print(json.dumps({
            "status": "failed_to_start",
            "returncode": proc.returncode,
            "log_file": str(LOG_FILE),
            "watcher_script": str(WATCHER_SCRIPT),
        }, ensure_ascii=False))
        return

    meta = {
        "pid": proc.pid,
        "started_at": int(time.time()),
        "python": sys.executable,
        "watcher_script": str(WATCHER_SCRIPT),
        "log_file": str(LOG_FILE),
    }

    atomic_write_text(PID_FILE, str(proc.pid))
    atomic_write_json(META_FILE, meta)

    print(json.dumps({
        "status": "started",
        "pid": proc.pid,
        "log_file": str(LOG_FILE),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
