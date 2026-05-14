"""
CI smoke test runner — quick validation that core imports and basic
operations work correctly.
"""
from __future__ import annotations

import importlib
import json
import os
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from checkpoint_db import close_pool

ROOT = Path.home() / ".openclaw" / "workspace" / ".memory-index"
SCRIPTS = ROOT / "scripts"
CONTROL = ROOT / "control_panel"
RUNTIME = ROOT / "runtime"

# Clean-room signal for tests
os.environ.setdefault("CLAW_ENV", "testing")

# Make sure the expected tree exists even in a fresh CI/container environment
ROOT.mkdir(parents=True, exist_ok=True)
SCRIPTS.mkdir(parents=True, exist_ok=True)
CONTROL.mkdir(parents=True, exist_ok=True)
RUNTIME.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(CONTROL))

TEST_PREFIX = "ci_smoke_testing"

results: list[dict[str, object]] = []


def record(name: str, ok: bool, detail: object = "") -> None:
    results.append({"check": name, "ok": ok, "detail": detail})


def run_check(name: str, fn) -> None:
    try:
        detail = fn()
        record(name, True, detail)
    except Exception as e:
        record(name, False, repr(e))


def check_imports():
    checks = [
        "checkpoint_db",
        "heartbeat_tuning",
        "heartbeat_memory_worker",
        "tracked_path_watcher",
        "start_heartbeat_worker",
        "stop_heartbeat_worker",
        "heartbeat_worker_status",
        "start_tracked_path_watcher",
        "stop_tracked_path_watcher",
        "tracked_path_watcher_status",
        "views",
        "app",
    ]
    loaded = []
    for modname in checks:
        importlib.import_module(modname)
        loaded.append(modname)
    return {"loaded": loaded, "claw_env": os.environ.get("CLAW_ENV")}


def check_db_bootstrap():
    from checkpoint_db import ensure_checkpoint_tables, list_recent_triggers, list_recent_checkpoints

    ensure_checkpoint_tables()
    triggers = list_recent_triggers(1)
    checkpoints = list_recent_checkpoints(1)
    return {
        "trigger_table_ok": isinstance(triggers, list),
        "checkpoint_table_ok": isinstance(checkpoints, list),
    }


def check_tuning_contract():
    from heartbeat_tuning import compute_effective_interval
    import heartbeat_tuning as ht

    original_triggers = ht.list_recent_triggers
    original_checkpoints = ht.list_recent_checkpoints
    try:
        # Emergency brake overrides backlog sprint
        ht.list_recent_triggers = lambda limit=10: [
            {"status": "pending"},
            {"status": "done"},
            {"status": "processing"},
        ]
        ht.list_recent_checkpoints = lambda limit=5: [
            {"status": "failed"},
            {"status": "failed"},
            {"status": "failed"},
        ]
        r1 = compute_effective_interval(30)
        if r1["effective_interval_seconds"] != 300:
            raise AssertionError(f"expected 300, got {r1}")

        now = datetime.now(timezone.utc)

        # Slow pipeline cooling
        ht.list_recent_triggers = lambda limit=10: []
        ht.list_recent_checkpoints = lambda limit=5: [
            {
                "status": "committed",
                "started_at": now - timedelta(seconds=25),
                "finished_at": now,
            }
        ]
        r2 = compute_effective_interval(30)
        if r2["effective_interval_seconds"] != 90:
            raise AssertionError(f"expected 90, got {r2}")

        # High activity settle filter
        ht.list_recent_triggers = lambda limit=10: [
            {"status": "done"},
            {"status": "done"},
            {"status": "processing"},
        ]
        ht.list_recent_checkpoints = lambda limit=5: []
        r3 = compute_effective_interval(30)
        if r3["effective_interval_seconds"] != 60:
            raise AssertionError(f"expected 60, got {r3}")

        # Healthy backlog sprint
        ht.list_recent_triggers = lambda limit=10: [{"status": "pending"}]
        ht.list_recent_checkpoints = lambda limit=5: []
        r4 = compute_effective_interval(30)
        if r4["effective_interval_seconds"] != 15:
            raise AssertionError(f"expected 15, got {r4}")

        # Idle steady pulse uses passed default
        ht.list_recent_triggers = lambda limit=10: []
        ht.list_recent_checkpoints = lambda limit=5: []
        r5 = compute_effective_interval(45)
        if r5["effective_interval_seconds"] != 45:
            raise AssertionError(f"expected 45, got {r5}")

        return {
            "failure_brake": r1,
            "slow_pipeline": r2,
            "high_activity": r3,
            "healthy_backlog": r4,
            "steady_pulse": r5,
        }
    finally:
        ht.list_recent_triggers = original_triggers
        ht.list_recent_checkpoints = original_checkpoints


def check_ignore_list():
    from tracked_path_watcher import is_ignored_path

    cases = {
        ".memory-index runtime": is_ignored_path(Path("/tmp/workspace/.memory-index/runtime/file.json")),
        ".git objects": is_ignored_path(Path("/tmp/workspace/.git/objects/ab")),
        "normal memory file": is_ignored_path(Path("/tmp/workspace/memory/project/current.md")),
    }
    if cases[".memory-index runtime"] is not True:
        raise AssertionError(cases)
    if cases[".git objects"] is not True:
        raise AssertionError(cases)
    if cases["normal memory file"] is not False:
        raise AssertionError(cases)
    return cases


def check_safe_duration():
    from heartbeat_tuning import _safe_duration_seconds

    now = datetime.now(timezone.utc)

    bad_none = _safe_duration_seconds({"started_at": None, "finished_at": None})
    bad_string = _safe_duration_seconds({"started_at": "oops", "finished_at": "bad"})
    negative = _safe_duration_seconds(
        {
            "started_at": now,
            "finished_at": now - timedelta(seconds=5),
        }
    )

    # Current safety contract: malformed values must not crash.
    # Negative durations should not be positive or explode.
    if bad_none is not None:
        raise AssertionError({"bad_none": bad_none})
    if bad_string is not None:
        raise AssertionError({"bad_string": bad_string})
    if negative is not None and negative >= 0:
        raise AssertionError({"negative_duration_result": negative})

    return {
        "bad_none": bad_none,
        "bad_string": bad_string,
        "negative_duration_result": negative,
    }


def check_pid_stale_logic():
    from heartbeat_manager_common import pid_alive

    fake_pid = 999999
    alive = pid_alive(fake_pid)
    if alive:
        raise AssertionError(f"fake pid {fake_pid} unexpectedly alive")
    return {"fake_pid": fake_pid, "alive": alive}


def check_write_permission():
    ROOT.mkdir(parents=True, exist_ok=True)
    RUNTIME.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(dir=RUNTIME, prefix="ci-write-", suffix=".tmp", delete=False) as f:
        f.write(b"ok")
        temp_path = Path(f.name)

    text = temp_path.read_text(encoding="utf-8")
    temp_path.unlink(missing_ok=True)

    return {"runtime_dir": str(RUNTIME), "roundtrip": text}


def check_state_roundtrip():
    from checkpoint_db import set_state, get_state

    key = f"{TEST_PREFIX}_state_roundtrip"
    payload = {"ok": True, "value": 123, "name": TEST_PREFIX}
    set_state(key, payload)
    got = get_state(key, {})
    if got != payload:
        raise AssertionError({"expected": payload, "got": got})
    return {"key": key, "value": got}


def check_trigger_roundtrip():
    from checkpoint_db import (
        enqueue_trigger,
        claim_next_trigger,
        complete_trigger,
        list_recent_triggers,
        mark_stale_processing_triggers,
    )

    mark_stale_processing_triggers(0)

    reason = f"{TEST_PREFIX}_roundtrip"
    dedupe_key = f"{TEST_PREFIX}:trigger_roundtrip"

    trigger_id = enqueue_trigger(
        source_type="ci",
        reason=reason,
        payload={"hello": "world", "test_prefix": TEST_PREFIX},
        priority=999,
        dedupe_key=dedupe_key,
    )
    row = claim_next_trigger()
    if not row or int(row["id"]) != int(trigger_id):
        raise AssertionError({"expected_id": trigger_id, "claimed": row})

    complete_trigger(trigger_id)

    rows = list_recent_triggers(10)
    latest = next((r for r in rows if int(r["id"]) == int(trigger_id)), None)
    if not latest or latest["status"] != "done":
        raise AssertionError({"trigger_id": trigger_id, "latest": latest})

    return {"trigger_id": trigger_id, "status": latest["status"], "reason": reason}


def main():
    run_check("imports", check_imports)
    run_check("db_bootstrap", check_db_bootstrap)
    run_check("tuning_contract", check_tuning_contract)
    run_check("ignore_list", check_ignore_list)
    run_check("safe_duration", check_safe_duration)
    run_check("pid_stale_logic", check_pid_stale_logic)
    run_check("write_permission", check_write_permission)
    run_check("state_roundtrip", check_state_roundtrip)
    run_check("trigger_roundtrip", check_trigger_roundtrip)

    ok = all(r["ok"] for r in results)
    print(json.dumps({"ok": ok, "results": results}, indent=2), flush=True)

    try:
        close_pool()
    except Exception as e:
        print(json.dumps({
            "pool_close_warning": repr(e)
        }), flush=True)

    raise SystemExit(0 if ok else 1)
