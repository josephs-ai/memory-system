"""
Stop the heartbeat worker service/daemon.

Key functions: main
"""
from __future__ import annotations

import json
import signal
import time

from heartbeat_manager_common import (
    read_pid,
    remove_runtime_files,
    terminate_process_group,
    pid_alive,
)


def main():
    pid = read_pid()
    if not pid:
        print(json.dumps({"status": "not_running"}, ensure_ascii=False))
        return

    if not pid_alive(pid):
        remove_runtime_files()
        print(json.dumps({"status": "already_dead", "pid": pid}, ensure_ascii=False))
        return

    terminate_process_group(pid, signal.SIGTERM)

    deadline = time.time() + 8
    while time.time() < deadline:
        if not pid_alive(pid):
            remove_runtime_files()
            print(json.dumps({"status": "stopped", "pid": pid}, ensure_ascii=False))
            return
        time.sleep(0.25)

    terminate_process_group(pid, signal.SIGKILL)

    deadline = time.time() + 3
    while time.time() < deadline:
        if not pid_alive(pid):
            remove_runtime_files()
            print(json.dumps({"status": "killed", "pid": pid}, ensure_ascii=False))
            return
        time.sleep(0.25)

    print(json.dumps({"status": "still_running", "pid": pid}, ensure_ascii=False))


if __name__ == "__main__":
    main()
