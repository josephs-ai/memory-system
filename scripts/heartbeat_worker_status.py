from __future__ import annotations

import json

from checkpoint_db import close_pool, get_state, latest_committed_checkpoint_id
from heartbeat_manager_common import (
    LOG_FILE,
    META_FILE,
    log_file_stats,
    pid_alive,
    read_json,
    read_pid,
)


def main():
    meta = read_json(META_FILE)
    pid = read_pid()
    running = bool(pid and pid_alive(pid))
    stale_pid_file = bool(pid and not running)

    try:
        latest_checkpoint = latest_committed_checkpoint_id()
        last_checkpoint_result = get_state("last_checkpoint_result", {})
        last_checkpoint_finished_at = get_state("last_checkpoint_finished_at", None)
        last_committed_trigger = get_state("last_committed_trigger", {})
        new_memory_available = get_state("new_memory_available", False)
        worker_last_loop_at = get_state("worker_last_loop_at", None)
        worker_pid_state = get_state("worker_pid", None)
        worker_running_state = get_state("worker_running", None)
        worker_started_at_state = get_state("worker_started_at", None)
    finally:
        close_pool()

    print(json.dumps({
        "running": running,
        "pid": pid,
        "stale_pid_file": stale_pid_file,
        "started_at_meta": meta.get("started_at"),
        "worker_started_at_state": worker_started_at_state,
        "worker_pid_state": worker_pid_state,
        "worker_running_state": worker_running_state,
        "worker_last_loop_at": worker_last_loop_at,
        "log_file": str(LOG_FILE),
        "log": log_file_stats(),
        "latest_committed_checkpoint_id": latest_checkpoint,
        "last_checkpoint_finished_at": last_checkpoint_finished_at,
        "last_checkpoint_result": last_checkpoint_result,
        "last_committed_trigger": last_committed_trigger,
        "new_memory_available": new_memory_available,
        "restart_recommended": (not running),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
