from __future__ import annotations

import argparse
import json
import os
import signal
import time
from pathlib import Path
from typing import Any

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from checkpoint_db import close_pool, enqueue_trigger, ensure_checkpoint_tables, set_state
from heartbeat_manager_common import build_runtime_env

WORKSPACE = Path.home() / ".openclaw" / "workspace"
CONTROL_PANEL_DIR = WORKSPACE / ".memory-index" / "control_panel"
CONFIG_PATH = CONTROL_PANEL_DIR / "control_panel_config.json"

IGNORED_DIR_NAMES = {
    "__pycache__",
    ".git",
    ".memory-index",
    "neo4j_data",
    "qdrant_data",
    ".pytest_cache",
    ".mypy_cache",
    "node_modules",
    "dist",
    "build",
    ".next",
    ".venv",
    "venv",
}
IGNORED_SUFFIXES = {
    ".log",
    ".tmp",
    ".temp",
    ".swp",
    ".swo",
    ".pyc",
    ".pyo",
    ".cache",
}
IGNORED_FILE_NAMES = {
    ".DS_Store",
}

STOP_REQUESTED = False
DEFAULT_DEBOUNCE_SECONDS = 10.0
DEFAULT_MAX_WAIT_SECONDS = 45.0
BOOTSTRAP_QUIET_SECONDS = 2.0


def _handle_signal(signum, frame):
    global STOP_REQUESTED
    STOP_REQUESTED = True


def load_config() -> dict[str, Any]:
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Missing config: {CONFIG_PATH}")
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def is_ignored_path(path: Path) -> bool:
    if path.name in IGNORED_FILE_NAMES:
        return True
    if path.suffix.lower() in IGNORED_SUFFIXES:
        return True
    return any(part in IGNORED_DIR_NAMES for part in path.parts)


def current_file_fingerprint(path: Path) -> dict[str, Any] | None:
    try:
        st = path.stat()
        return {
            "exists": True,
            "mtime": st.st_mtime,
            "size": st.st_size,
        }
    except FileNotFoundError:
        return {"exists": False}
    except Exception:
        return None


def read_inotify_limit() -> int | None:
    p = Path("/proc/sys/fs/inotify/max_user_watches")
    if not p.exists():
        return None
    try:
        return int(p.read_text(encoding="utf-8").strip())
    except Exception:
        return None


def estimate_watch_load(paths: list[str]) -> int:
    count = 0
    for raw in paths:
        p = Path(raw).expanduser()
        if not p.exists():
            continue
        if p.is_file():
            count += 1
            continue
        for d, dirs, files in os.walk(p):
            dp = Path(d)
            if is_ignored_path(dp):
                dirs[:] = []
                continue
            dirs[:] = [x for x in dirs if not is_ignored_path(dp / x)]
            count += 1
    return count


class DebouncedTrackedPathHandler(FileSystemEventHandler):
    def __init__(self, tracked_paths: list[str], debounce_seconds: float, max_wait_seconds: float):
        super().__init__()
        self.debounce_seconds = debounce_seconds
        self.max_wait_seconds = max_wait_seconds

        self.first_event_at: float | None = None
        self.last_event_at: float | None = None

        self.changed_paths: set[str] = set()
        self.path_fingerprints: dict[str, dict[str, Any]] = {}

        self.bootstrap_done = False
        self.bootstrap_started_at = time.time()

        self.tracked_files: set[Path] = set()
        self.tracked_dirs: list[Path] = []

        for raw in tracked_paths:
            p = Path(raw).expanduser().resolve()
            if not p.exists():
                continue
            if p.is_file():
                self.tracked_files.add(p)
            else:
                self.tracked_dirs.append(p)

    def finish_bootstrap(self) -> None:
        self.bootstrap_done = True
        set_state(
            "watcher_bootstrap",
            {
                "done": True,
                "finished_at": time.time(),
            },
        )

    def _is_allowed_tracked_path(self, path: Path) -> bool:
        try:
            rp = path.resolve()
        except Exception:
            return False

        if rp in self.tracked_files:
            return True

        for d in self.tracked_dirs:
            try:
                rp.relative_to(d)
                return True
            except ValueError:
                continue

        return False

    def _record_path(self, path: Path, event_type: str) -> None:
        try:
            rp = path.resolve()
        except Exception:
            return

        # hard kill-switch for internal control/runtime files
        if ".memory-index" in rp.parts:
            return

        if is_ignored_path(rp):
            return
        if not self._is_allowed_tracked_path(rp):
            return

        fp = current_file_fingerprint(rp)
        if fp is None:
            return

        key = str(rp)

        old_fp = self.path_fingerprints.get(key)

        if old_fp == fp and event_type == "modified":
            return

        self.path_fingerprints[key] = fp
        self.changed_paths.add(key)

        now = time.time()
        if self.first_event_at is None:
            self.first_event_at = now
        self.last_event_at = now

        set_state(
            "watcher_last_event",
            {
                "path": key,
                "event_type": event_type,
                "at": now,
                "bootstrap_done": self.bootstrap_done,
            },
        )

    def on_created(self, event):
        if event.is_directory:
            return
        self._record_path(Path(event.src_path), "created")

    def on_modified(self, event):
        if event.is_directory:
            return
        self._record_path(Path(event.src_path), "modified")

    def on_deleted(self, event):
        if event.is_directory:
            return
        self._record_path(Path(event.src_path), "deleted")

    def on_moved(self, event):
        if event.is_directory:
            return
        self._record_path(Path(event.src_path), "moved_from")
        self._record_path(Path(event.dest_path), "moved_to")

    def flush_if_ready(self) -> bool:
        if not self.changed_paths or self.last_event_at is None:
            return False

        now = time.time()
        debounce_ready = (now - self.last_event_at) >= self.debounce_seconds
        max_wait_ready = (
            self.first_event_at is not None
            and (now - self.first_event_at) >= self.max_wait_seconds
        )

        if not debounce_ready and not max_wait_ready:
            return False

        changed = sorted(self.changed_paths)
        trigger_reason = (
            "inotify_tracked_path_change_max_wait"
            if max_wait_ready and not debounce_ready
            else "inotify_tracked_path_change"
        )

        payload = {
            "trigger_requested": True,
            "trigger_type": "watcher",
            "trigger_reason": trigger_reason,
            "changed_paths": changed,
            "count": len(changed),
            "requested_at": now,
            "bootstrap_done": self.bootstrap_done,
        }

        dedupe_key = f"watcher:{'|'.join(changed)}"

        trigger_id = enqueue_trigger(
            source_type="watcher",
            reason=trigger_reason,
            payload=payload,
            priority=50,
            dedupe_key=dedupe_key,
        )

        payload["trigger_id"] = trigger_id

        set_state("watcher_trigger", payload)
        set_state("watcher_last_flush_at", now)
        self.changed_paths.clear()
        self.first_event_at = None
        self.last_event_at = None
        return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--debounce-seconds", type=float, default=DEFAULT_DEBOUNCE_SECONDS)
    parser.add_argument("--max-wait-seconds", type=float, default=DEFAULT_MAX_WAIT_SECONDS)
    args = parser.parse_args()

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    ensure_checkpoint_tables()
    build_runtime_env()

    cfg = load_config()
    tracked_paths = [str(Path(p).expanduser()) for p in cfg.get("tracked_paths", [])]

    inotify_limit = read_inotify_limit()
    estimated_watches = estimate_watch_load(tracked_paths)

    set_state(
        "watcher_limits",
        {
            "inotify_max_user_watches": inotify_limit,
            "estimated_watch_load": estimated_watches,
            "warning": bool(inotify_limit is not None and estimated_watches >= int(inotify_limit * 0.8)),
            "checked_at": time.time(),
        },
    )

    handler = DebouncedTrackedPathHandler(
        tracked_paths=tracked_paths,
        debounce_seconds=args.debounce_seconds,
        max_wait_seconds=args.max_wait_seconds,
    )
    observer = Observer()

    # Directories: recursive
    dir_watch_targets: set[str] = set()
    # Files: parent only, non-recursive
    file_watch_targets: set[str] = set()

    for raw in tracked_paths:
        p = Path(raw).expanduser()
        if not p.exists():
            continue
        rp = p.resolve()
        if rp.is_dir():
            dir_watch_targets.add(str(rp))
        else:
            file_watch_targets.add(str(rp.parent))

    for target in sorted(dir_watch_targets):
        observer.schedule(handler, target, recursive=True)

    for target in sorted(file_watch_targets):
        observer.schedule(handler, target, recursive=False)

    set_state("watcher_running", True)
    set_state("watcher_started_at", time.time())
    set_state("watcher_pid", os.getpid())
    set_state("watcher_watch_count", len(dir_watch_targets) + len(file_watch_targets))
    set_state("watcher_tracked_paths", tracked_paths)

    set_state("watcher_last_event", {})
    set_state("watcher_last_flush_at", None)
    set_state(
        "watcher_trigger",
        {
            "trigger_requested": False,
            "changed_paths": [],
            "count": 0,
            "trigger_type": None,
            "trigger_reason": None,
            "requested_at": None,
            "bootstrap_done": False,
        },
    )
    set_state(
        "watcher_bootstrap",
        {
            "done": False,
            "started_at": handler.bootstrap_started_at,
        },
    )

    observer.start()

    try:
        bootstrap_deadline = time.time() + BOOTSTRAP_QUIET_SECONDS

        while not STOP_REQUESTED:
            now = time.time()

            if not handler.bootstrap_done and now >= bootstrap_deadline:
                handler.finish_bootstrap()

            handler.flush_if_ready()

            if args.once:
                break

            time.sleep(1.0)
    finally:
        observer.stop()
        observer.join(timeout=5)
        set_state("watcher_running", False)
        set_state("watcher_pid", None)
        close_pool()


if __name__ == "__main__":
    main()
