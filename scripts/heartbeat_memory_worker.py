"""
Background worker that periodically runs memory pipeline tasks —
checkpointing, embedding, reconciliation, and maintenance cycles.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import signal
import subprocess
import sys
import time
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any
from heartbeat_tuning import compute_effective_interval

SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE = Path.home() / ".openclaw" / "workspace"
CONTROL_PANEL_DIR = WORKSPACE / ".memory-index" / "control_panel"
CONFIG_PATH = CONTROL_PANEL_DIR / "control_panel_config.json"
SCRIPTS_DIR = WORKSPACE / ".memory-index" / "scripts"

from checkpoint_db import (
    advisory_lock,
    begin_checkpoint,
    claim_next_trigger,
    close_pool,
    complete_trigger,
    ensure_checkpoint_tables,
    fail_trigger,
    get_state,
    mark_stale_processing_checkpoints,
    mark_stale_processing_triggers,
    set_state,
    update_checkpoint,
)

from heartbeat_manager_common import LOG_FILE, build_runtime_env

POLL_SECONDS = 5
DEFAULT_DEBOUNCE_SECONDS = 12

IGNORED_DIR_NAMES = {
    "__pycache__",
    ".git",
    "neo4j_data",
    "qdrant_data",
    ".pytest_cache",
    ".mypy_cache",
}
IGNORED_SUFFIXES = {
    ".log",
    ".tmp",
    ".temp",
    ".swp",
    ".swo",
    ".pyc",
}
IGNORED_FILE_NAMES = {
    ".DS_Store",
}

LOGGER = logging.getLogger("heartbeat_worker")
STOP_REQUESTED = False


def setup_logging() -> None:
    if LOGGER.handlers:
        return

    max_bytes = int(os.environ.get("OPENCLAW_HEARTBEAT_LOG_MAX_BYTES", "5242880"))
    backups = int(os.environ.get("OPENCLAW_HEARTBEAT_LOG_BACKUPS", "5"))

    handler = RotatingFileHandler(
        LOG_FILE,
        maxBytes=max_bytes,
        backupCount=backups,
        encoding="utf-8",
    )
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    handler.setFormatter(formatter)

    LOGGER.setLevel(logging.INFO)
    LOGGER.addHandler(handler)
    LOGGER.propagate = False


def _handle_signal(signum, frame):
    global STOP_REQUESTED
    STOP_REQUESTED = True
    LOGGER.info("received_signal signum=%s", signum)


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


def iter_files(path_str: str):
    p = Path(path_str).expanduser()
    if not p.exists():
        return
    if p.is_file():
        if not is_ignored_path(p):
            yield p
        return
    for fp in p.rglob("*"):
        if fp.is_file() and not is_ignored_path(fp):
            yield fp


def sha1_file(path: Path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def snapshot_tracked_paths(paths: list[str]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for path_str in paths:
        for fp in iter_files(path_str):
            try:
                stat = fp.stat()
                out[str(fp)] = {
                    "mtime": stat.st_mtime,
                    "size": stat.st_size,
                    "sha1": sha1_file(fp),
                }
            except Exception:
                continue
    return out


def diff_snapshot(old: dict[str, Any], new: dict[str, Any]) -> dict[str, Any]:
    old_keys = set(old.keys())
    new_keys = set(new.keys())

    added = sorted(new_keys - old_keys)
    removed = sorted(old_keys - new_keys)
    changed = sorted(
        k for k in (old_keys & new_keys)
        if old[k].get("sha1") != new[k].get("sha1")
    )

    return {
        "added": added,
        "removed": removed,
        "changed": changed,
        "count": len(added) + len(removed) + len(changed),
    }


def run_cmd(cmd: list[str], env: dict[str, str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        cwd=str(WORKSPACE),
        capture_output=True,
        text=True,
        env=env,
    )


def run_pipeline_and_sync(env: dict[str, str]) -> dict[str, Any]:
    py = sys.executable

    steps = [
        {
            "name": "maintenance_cycle",
            "cmd": [py, str(SCRIPTS_DIR / "run_memory_maintenance_cycle.py")],
        },
        {
            "name": "embed_active",
            "cmd": [py, str(SCRIPTS_DIR / "embed_memory_items.py"), "--batch-size", "64"],
        },
        {
            "name": "sync_neo4j",
            "cmd": [py, str(SCRIPTS_DIR / "sync_memory_to_neo4j.py")],
        },
    ]

    summary: dict[str, Any] = {"steps": [], "ok": True}

    for step in steps:
        LOGGER.info("pipeline_step_start name=%s", step["name"])
        proc = run_cmd(step["cmd"], env)
        row = {
            "name": step["name"],
            "returncode": proc.returncode,
            "stdout": proc.stdout[-4000:],
            "stderr": proc.stderr[-4000:],
        }
        summary["steps"].append(row)

        if proc.returncode != 0:
            LOGGER.error(
                "pipeline_step_failed name=%s returncode=%s",
                step["name"],
                proc.returncode,
            )
            summary["ok"] = False
            break

        LOGGER.info("pipeline_step_ok name=%s", step["name"])

    return summary


def consume_watcher_trigger() -> dict[str, Any] | None:
    trigger = get_state("watcher_trigger", None)
    if not trigger or not isinstance(trigger, dict):
        return None

    if not trigger.get("trigger_requested"):
        return None

    set_state("watcher_trigger", {
        "trigger_requested": False,
        "consumed_at": time.time(),
        "last_consumed_trigger": trigger,
    })
    set_state("watcher_last_consumed_trigger", trigger)
    return trigger

def claim_durable_trigger() -> dict[str, Any] | None:
    mark_stale_processing_triggers()
    return claim_next_trigger()

def maybe_run_checkpoint(
    trigger_type: str,
    trigger_reason: str,
    delta_meta: dict[str, Any],
    *,
    trigger_row: dict[str, Any] | None = None,
) -> bool:
    env = build_runtime_env()

    with advisory_lock():
        checkpoint_id = begin_checkpoint(trigger_type, trigger_reason, delta_meta)
        LOGGER.info(
            "checkpoint_begin id=%s trigger_type=%s trigger_reason=%s delta_count=%s",
            checkpoint_id,
            trigger_type,
            trigger_reason,
            delta_meta.get("count", 0),
        )

        result = run_pipeline_and_sync(env)

        if result["ok"]:
            counts = {
                "trigger_type": trigger_type,
                "delta_count": delta_meta.get("count", 0),
                "files_changed": len(delta_meta.get("changed", [])),
                "files_added": len(delta_meta.get("added", [])),
                "files_removed": len(delta_meta.get("removed", [])),
            }
            update_checkpoint(
                checkpoint_id,
                status="committed",
                counts=counts,
            )
            if delta_meta.get("new_snapshot"):
                set_state("tracked_snapshot", delta_meta.get("new_snapshot", {}))
            set_state("last_checkpoint_finished_at", time.time())
            set_state("last_checkpoint_result", result)
            set_state("new_memory_available", True)
            set_state("last_committed_trigger", {"type": trigger_type, "reason": trigger_reason})
            LOGGER.info("checkpoint_committed id=%s", checkpoint_id)

            if trigger_row:
                complete_trigger(int(trigger_row["id"]))
                set_state("last_completed_trigger_id", int(trigger_row["id"]))

            return True
        else:
            error_text = json.dumps(result)[-8000:]
            update_checkpoint(
                checkpoint_id,
                status="failed",
                error_text=error_text,
            )
            set_state("last_checkpoint_result", result)
            set_state("last_checkpoint_finished_at", time.time())
            LOGGER.error("checkpoint_failed id=%s", checkpoint_id)

            if trigger_row:
                fail_trigger(int(trigger_row["id"]), error_text)
                set_state("last_failed_trigger_id", int(trigger_row["id"]))

            return False

def worker_loop(once: bool = False) -> None:
    setup_logging()
    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    ensure_checkpoint_tables()
    mark_stale_processing_checkpoints()

    set_state("worker_running", True)
    set_state("worker_pid", os.getpid())
    set_state("worker_started_at", time.time())
    LOGGER.info("worker_started once=%s", once)

    cfg = load_config()

    hb = cfg.get("heartbeat", {})
    heartbeat_enabled = bool(hb.get("heartbeat_enabled", True))
    default_interval = int(hb.get("heartbeat_interval_seconds", 30) or 30)
    tracked_paths = list(cfg.get("tracked_paths", []))

    if not heartbeat_enabled:
        LOGGER.info("heartbeat_disabled_by_config")
        print("heartbeat disabled")
        return

    pending_since: float | None = None

    while not STOP_REQUESTED:
        tuning = compute_effective_interval(default_interval)
        interval = int(tuning["effective_interval_seconds"])
        set_state("heartbeat_tuning", tuning)
        set_state("heartbeat_effective_interval_seconds", interval)

        trigger_row = claim_durable_trigger()
        if trigger_row:
            payload = trigger_row.get("payload") or {}
            LOGGER.info(
                "checkpoint_trigger type=timer reason=interval_%ss tuning_reason=%s delta_count=%s",
                interval,
                tuning.get("reason"),
                delta["count"],
            )
            ok = maybe_run_checkpoint(
                trigger_row.get("source_type", "trigger"),
                trigger_row.get("reason", "queued_trigger"),
                payload,
                trigger_row=trigger_row,
            )

            if ok:
                set_state("tracked_snapshot", snapshot_tracked_paths(tracked_paths))
                pending_since = None

            if once:
                break

            time.sleep(1.0)
            continue

        old_snapshot = get_state("tracked_snapshot", {}) or {}
        new_snapshot = snapshot_tracked_paths(tracked_paths)
        delta = diff_snapshot(old_snapshot, new_snapshot)
        delta["new_snapshot"] = new_snapshot

        now = time.time()
        last_finished = get_state("last_checkpoint_finished_at", 0) or 0
        timer_due = bool(delta["count"] > 0 and (now - float(last_finished)) >= interval)

        if delta["count"] > 0:
            if pending_since is None:
                pending_since = now
            debounce_done = (now - pending_since) >= DEFAULT_DEBOUNCE_SECONDS
        else:
            pending_since = None
            debounce_done = False

        fired = False

        if delta["count"] > 0 and debounce_done:
            LOGGER.info(
                "checkpoint_trigger type=file reason=tracked_file_change delta_count=%s",
                delta["count"],
            )
            maybe_run_checkpoint("file", "tracked_file_change", delta)
            pending_since = None
            fired = True
        elif timer_due:
            LOGGER.info(
                "checkpoint_trigger type=timer reason=interval_%ss delta_count=%s",
                interval,
                delta["count"],
            )
            maybe_run_checkpoint("timer", f"interval_{interval}s", delta)
            pending_since = None
            fired = True

        if once:
            if not fired and delta["count"] > 0:
                LOGGER.info(
                    "checkpoint_trigger type=manual reason=once_mode_unsaved_delta delta_count=%s",
                    delta["count"],
                )
                maybe_run_checkpoint("manual", "once_mode_unsaved_delta", delta)
            break

        time.sleep(POLL_SECONDS)

    LOGGER.info("worker_stopping stop_requested=%s", STOP_REQUESTED)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    try:
        worker_loop(once=args.once)
    finally:
        try:
            set_state("worker_running", False)
        except Exception:
            pass
        close_pool()


if __name__ == "__main__":
    main()
