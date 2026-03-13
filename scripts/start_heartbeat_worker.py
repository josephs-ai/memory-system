from __future__ import annotations

import json
import subprocess
import sys
import time

from heartbeat_manager_common import (
    LOG_FILE,
    META_FILE,
    PID_FILE,
    WORKER_SCRIPT,
    atomic_write_json,
    atomic_write_text,
    build_runtime_env,
    ensure_runtime_dir,
    pid_alive,
    process_group_alive,
    read_json,
    read_pid,
    remove_runtime_files,
)


def main():
    ensure_runtime_dir()

    existing_pid = read_pid()
    if existing_pid and process_group_alive(existing_pid):
        meta = read_json(META_FILE)
        print(json.dumps({
            "status": "already_running",
            "pid": existing_pid,
            "log_file": str(LOG_FILE),
            "started_at": meta.get("started_at"),
        }, ensure_ascii=False))
        return

    if existing_pid and not pid_alive(existing_pid):
        remove_runtime_files()

    env = build_runtime_env()

    with open(LOG_FILE, "a", encoding="utf-8") as logf:
        proc = subprocess.Popen(
            [sys.executable, str(WORKER_SCRIPT)],
            stdin=subprocess.DEVNULL,
            stdout=logf,
            stderr=logf,
            env=env,
            start_new_session=True,
        )

    time.sleep(0.75)
    if proc.poll() is not None:
        print(json.dumps({
            "status": "failed_to_start",
            "returncode": proc.returncode,
            "log_file": str(LOG_FILE),
        }, ensure_ascii=False))
        return

    meta = {
        "pid": proc.pid,
        "started_at": int(time.time()),
        "python": sys.executable,
        "worker_script": str(WORKER_SCRIPT),
        "log_file": str(LOG_FILE),
        "env_file_used": str(META_FILE.parent.parent / ".env"),
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
