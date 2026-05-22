"""
Tracked Path Watcher Status — utility for the OpenClaw memory system.

Key functions: read_pid, log_stats, ts_to_iso, age_seconds
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

from checkpoint_db import close_pool, get_state
from heartbeat_manager_common import MEMORY_INDEX_DIR, pid_alive, read_json

RUNTIME_DIR = MEMORY_INDEX_DIR / "runtime"
PID_FILE = RUNTIME_DIR / "tracked_path_watcher.pid"
META_FILE = RUNTIME_DIR / "tracked_path_watcher.json"
LOG_FILE = MEMORY_INDEX_DIR / "tracked_path_watcher.log"


def read_pid() -> int | None:
    if not PID_FILE.exists():
        return None
    try:
        return int(PID_FILE.read_text(encoding="utf-8").strip())
    except Exception:
        return None


def log_stats() -> dict:
    try:
        st = LOG_FILE.stat()
        return {
            "exists": True,
            "size_bytes": st.st_size,
            "mtime": st.st_mtime,
            "mtime_iso": ts_to_iso(st.st_mtime),
        }
    except Exception:
        return {
            "exists": False,
            "size_bytes": 0,
            "mtime": None,
            "mtime_iso": None,
        }


def ts_to_iso(ts) -> str | None:
    if ts is None:
        return None
    try:
        return datetime.fromtimestamp(float(ts), tz=timezone.utc).isoformat()
    except Exception:
        return None


def age_seconds(ts) -> float | None:
    if ts is None:
        return None
    try:
        return round(datetime.now(timezone.utc).timestamp() - float(ts), 3)
    except Exception:
        return None


def main():
    meta = read_json(META_FILE)
    pid = read_pid()
    running = bool(pid and pid_alive(pid))
    stale_pid_file = bool(pid and not running)

    # Safe defaults so DB/read failures do not break final JSON rendering
    watcher_running = None
    watcher_pid = None
    watcher_started_at = None
    watcher_watch_count = None
    watcher_tracked_paths = []
    watcher_last_event = {}
    watcher_last_flush_at = None
    watcher_bootstrap = {}
    watcher_trigger = {}
    watcher_limits = {}
    db_error = None

    try:
        watcher_running = get_state("watcher_running", None)
        watcher_pid = get_state("watcher_pid", None)
        watcher_started_at = get_state("watcher_started_at", None)
        watcher_watch_count = get_state("watcher_watch_count", None)
        watcher_tracked_paths = get_state("watcher_tracked_paths", [])
        watcher_last_event = get_state("watcher_last_event", {})
        watcher_last_flush_at = get_state("watcher_last_flush_at", None)
        watcher_bootstrap = get_state("watcher_bootstrap", {})
        watcher_trigger = get_state("watcher_trigger", {})
        watcher_limits = get_state("watcher_limits", {})
    except Exception as e:
        db_error = str(e)
    finally:
        close_pool()

    restart_recommended = (not running) or (
        pid is not None and watcher_pid is not None and pid != watcher_pid
    )

    print(json.dumps({
        "running": running,
        "pid": pid,
        "stale_pid_file": stale_pid_file,

        "started_at_meta": meta.get("started_at"),
        "started_at_meta_iso": ts_to_iso(meta.get("started_at")),
        "started_at_meta_age_seconds": age_seconds(meta.get("started_at")),

        "watcher_running_state": watcher_running,
        "watcher_pid_state": watcher_pid,
        "watcher_started_at_state": watcher_started_at,
        "watcher_started_at_state_iso": ts_to_iso(watcher_started_at),
        "watcher_started_at_state_age_seconds": age_seconds(watcher_started_at),

        "watcher_watch_count": watcher_watch_count,
        "watcher_tracked_paths": watcher_tracked_paths,
        "watcher_last_event": watcher_last_event,
        "watcher_last_flush_at": watcher_last_flush_at,
        "watcher_last_flush_at_iso": ts_to_iso(watcher_last_flush_at),
        "watcher_last_flush_at_age_seconds": age_seconds(watcher_last_flush_at),

        "watcher_bootstrap": watcher_bootstrap,
        "watcher_trigger": watcher_trigger,
        "watcher_limits": watcher_limits,

        "log_file": str(LOG_FILE),
        "log": log_stats(),

        "restart_recommended": restart_recommended,
        "db_error": db_error,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
