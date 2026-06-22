#!/usr/bin/env python3
"""Hermetic tests for the readable-memory refresh orchestrator.

Injects a fake step-runner and a fake health-probe so no heavy generator
subprocess is spawned — fast and deterministic. The single live end-to-end check
is marked @pytest.mark.integration and deselected from the default run.
"""
from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pytest

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_readable_memory_refresh as rr  # noqa: E402


# --- helpers ---------------------------------------------------------------
def ok_runner(record):
    """A runner that records invocation order and returns success + valid JSON."""
    def _run(cmd, timeout):
        record.append(cmd)
        return 0, json.dumps({"ok": True}), "", ""
    return _run


def health_all_ok():
    checks = [{"name": n, "status": "ok", "detail": "fresh", "value": 5}
              for n in rr.GATE_CHECKS]
    checks.append({"name": "rotated_transcripts_visible", "status": "warn",
                   "detail": "82 rotated", "value": 82})
    return {"overall": "warn", "checks": checks}


def _health_with(overrides):
    checks = []
    for n in rr.GATE_CHECKS:
        c = {"name": n, "status": "ok", "detail": "fresh", "value": 5}
        if n in overrides:
            c.update(overrides[n])
        checks.append(c)
    worst = "ok"
    sev = {"ok": 0, "skip": 0, "warn": 1, "fail": 2}
    for c in checks:
        if sev[c["status"]] > sev[worst]:
            worst = c["status"]
    return {"overall": worst, "checks": checks}


# --- order / args ----------------------------------------------------------
def test_steps_run_in_dependency_order(tmp_path):
    calls = []
    s = rr.run_refresh(log_dir=tmp_path, runner=ok_runner(calls),
                       health_probe=health_all_ok, lock=False)
    scripts = [Path(c[1]).name for c in calls]
    assert scripts == [
        "generate_daily_session_card.py", "consolidate_memory_digest.py",
        "generate_agent_boot_pack.py", "card_search_index.py",
        "generate_memory_dashboard.py",
    ]
    assert s["overall"] == "ok" and s["exit_code"] == 0


def test_uses_sys_executable_and_forwards_args(tmp_path):
    calls = []
    rr.run_refresh(log_dir=tmp_path, since_days=3.0, runner=ok_runner(calls),
                   health_probe=health_all_ok, lock=False)
    for c in calls:
        assert c[0] == sys.executable
    # --all-agents on steps 1-3, --since-days on 1&2
    assert "--all-agents" in calls[0] and "--all-agents" in calls[1] and "--all-agents" in calls[2]
    assert "--since-days" in calls[0] and "3.0" in calls[0]
    assert "--since-days" in calls[1]
    assert "--all-agents" not in calls[3] and "--all-agents" not in calls[4]


# --- continue-on-error -----------------------------------------------------
def test_continue_on_error_runs_rest_degraded(tmp_path):
    calls = []

    def runner(cmd, timeout):
        calls.append(cmd)
        if "consolidate_memory_digest.py" in cmd[1]:
            return 1, "", "boom", ""
        return 0, json.dumps({"ok": True}), "", ""

    s = rr.run_refresh(log_dir=tmp_path, runner=runner, health_probe=health_all_ok,
                       lock=False, continue_on_error=True)
    # all 5 steps still invoked
    assert len(calls) == 5
    # artifacts healthy (gate pass) but a step failed -> degraded, exit 2
    assert s["overall"] == "degraded" and s["exit_code"] == 2


def test_no_continue_on_error_aborts_failed(tmp_path):
    calls = []

    def runner(cmd, timeout):
        calls.append(cmd)
        if "consolidate_memory_digest.py" in cmd[1]:
            return 1, "", "boom", ""
        return 0, json.dumps({"ok": True}), "", ""

    s = rr.run_refresh(log_dir=tmp_path, runner=runner, health_probe=health_all_ok,
                       lock=False, continue_on_error=False)
    assert len(calls) == 2  # stopped after step 2
    assert s["overall"] == "failed" and s["exit_code"] == 1


# --- timeout / non-json ----------------------------------------------------
def test_timeout_step_not_fatal(tmp_path):
    def runner(cmd, timeout):
        if "card_search_index.py" in cmd[1]:
            return None, "", "", "timeout"
        return 0, json.dumps({"ok": True}), "", ""

    s = rr.run_refresh(log_dir=tmp_path, runner=runner, health_probe=health_all_ok, lock=False)
    idx = next(st for st in s["steps"] if "card_search_index" in st["name"])
    assert idx["ok"] is False and idx["reason"] == "timeout"
    assert len(s["steps"]) == 5  # chain continued


def test_non_json_stdout_ok_when_rc_zero(tmp_path):
    def runner(cmd, timeout):
        return 0, "not json at all", "", ""

    s = rr.run_refresh(log_dir=tmp_path, runner=runner, health_probe=health_all_ok, lock=False)
    assert all(st["ok"] for st in s["steps"])
    assert all(st["parsed_json"] is None for st in s["steps"])


# --- THE GATE (core) -------------------------------------------------------
def test_gate_missing_after_build_fails(tmp_path):
    """card index reports 0 records AFTER the index step ran -> FAIL, exit 1."""
    probe = lambda: _health_with({"card_search_ready":
                                  {"status": "warn", "detail": "card index empty (0 records)", "value": 0}})
    s = rr.run_refresh(log_dir=tmp_path, runner=ok_runner([]), health_probe=probe, lock=False)
    assert s["gate"]["per_artifact"]["card_search_ready"]["verdict"] == "fail"
    assert s["overall"] == "failed" and s["exit_code"] == 1


def test_gate_stale_is_degraded(tmp_path):
    """dashboard present but stale -> DEGRADED, exit 2 (not fail)."""
    probe = lambda: _health_with({"dashboard_fresh":
                                  {"status": "warn", "detail": "dashboard stale (generated ...)", "value": None}})
    s = rr.run_refresh(log_dir=tmp_path, runner=ok_runner([]), health_probe=probe, lock=False)
    assert s["gate"]["per_artifact"]["dashboard_fresh"]["verdict"] == "degraded"
    assert s["overall"] == "degraded" and s["exit_code"] == 2


def test_gate_digest_stale_not_missing_is_degraded(tmp_path):
    """digest 'none fresh' (stale, value==0) -> DEGRADED, NOT FAIL.

    Pins the load-bearing token: digest missing AND stale both report value==0; only
    'no digest nodes yet' (missing) vs 'none fresh' (stale) text distinguishes them."""
    probe = lambda: _health_with({"digest_fresh":
                                  {"status": "warn",
                                   "detail": "6 digest node(s) but none fresh (newest ...)", "value": 0}})
    s = rr.run_refresh(log_dir=tmp_path, runner=ok_runner([]), health_probe=probe, lock=False)
    assert s["gate"]["per_artifact"]["digest_fresh"]["verdict"] == "degraded"
    assert s["overall"] == "degraded"


def test_gate_digest_missing_is_fail(tmp_path):
    probe = lambda: _health_with({"digest_fresh":
                                  {"status": "warn",
                                   "detail": "no digest nodes yet (auto_dream not run)", "value": 0}})
    s = rr.run_refresh(log_dir=tmp_path, runner=ok_runner([]), health_probe=probe, lock=False)
    assert s["gate"]["per_artifact"]["digest_fresh"]["verdict"] == "fail"
    assert s["overall"] == "failed"


def test_gate_skip_is_pass(tmp_path):
    """All five SKIP (unscaffolded/empty tree) -> pass -> ok, exit 0."""
    probe = lambda: {"overall": "ok",
                     "checks": [{"name": n, "status": "skip",
                                 "detail": "canonical layout not scaffolded", "value": None}
                                for n in rr.GATE_CHECKS]}
    s = rr.run_refresh(log_dir=tmp_path, runner=ok_runner([]), health_probe=probe, lock=False)
    assert all(v["verdict"] == "pass" for v in s["gate"]["per_artifact"].values())
    assert s["overall"] == "ok" and s["exit_code"] == 0


def test_unrelated_warn_not_polluting(tmp_path):
    """Five checks OK but an unrelated rotated_transcripts WARN -> still ok."""
    s = rr.run_refresh(log_dir=tmp_path, runner=ok_runner([]), health_probe=health_all_ok, lock=False)
    assert s["overall"] == "ok"
    assert s["health_overall"] == "warn"  # health overall is warn, but we ignore it


def test_partial_write_then_nonzero_artifact_present(tmp_path):
    """A step exits nonzero but its artifact is present+fresh (check OK) ->
    gate PASS for it; overall degraded (a step did fail) but NOT failed."""
    def runner(cmd, timeout):
        if "card_search_index.py" in cmd[1]:
            return 1, json.dumps({"records": 13}), "late error", ""
        return 0, json.dumps({"ok": True}), "", ""

    s = rr.run_refresh(log_dir=tmp_path, runner=runner, health_probe=health_all_ok, lock=False)
    assert s["gate"]["per_artifact"]["card_search_ready"]["verdict"] == "pass"
    assert s["overall"] == "degraded"  # step failed but artifact fine


# --- skip-health -----------------------------------------------------------
def test_skip_health_no_gate(tmp_path):
    calls = []
    s = rr.run_refresh(log_dir=tmp_path, runner=ok_runner(calls), lock=False, skip_health=True)
    assert s["gate"] is None
    assert s["overall"] == "ok" and s["exit_code"] == 0


# --- lockfile --------------------------------------------------------------
def test_lock_held_by_live_holder(tmp_path, monkeypatch):
    lock = tmp_path / ".lock"
    monkeypatch.setattr(rr, "LOCKFILE", lock)
    lock.write_text(json.dumps({"pid": __import__("os").getpid(), "started": "x"}), encoding="utf-8")
    calls = []
    s = rr.run_refresh(log_dir=tmp_path / "logs", runner=ok_runner(calls),
                       health_probe=health_all_ok, lock=True)
    assert s["overall"] == "locked" and s["exit_code"] == 2
    assert calls == []  # no steps run


def test_lock_stale_is_stolen(tmp_path, monkeypatch):
    lock = tmp_path / ".lock"
    monkeypatch.setattr(rr, "LOCKFILE", lock)
    # dead pid -> stale -> stolen
    lock.write_text(json.dumps({"pid": 999999999, "started": "x"}), encoding="utf-8")
    calls = []
    s = rr.run_refresh(log_dir=tmp_path / "logs", runner=ok_runner(calls),
                       health_probe=health_all_ok, lock=True)
    assert s["overall"] == "ok"
    assert len(calls) == 5  # proceeded


def test_force_steals_lock(tmp_path, monkeypatch):
    lock = tmp_path / ".lock"
    monkeypatch.setattr(rr, "LOCKFILE", lock)
    lock.write_text(json.dumps({"pid": __import__("os").getpid(), "started": "x"}), encoding="utf-8")
    calls = []
    s = rr.run_refresh(log_dir=tmp_path / "logs", runner=ok_runner(calls),
                       health_probe=health_all_ok, lock=True, force=True)
    assert s["overall"] == "ok" and len(calls) == 5


# --- log dir / outputs -----------------------------------------------------
def test_log_dir_uncreatable_fails_fast(tmp_path, monkeypatch):
    # point log_dir at a path under an existing FILE -> mkdir raises
    f = tmp_path / "afile"
    f.write_text("x", encoding="utf-8")
    calls = []
    s = rr.run_refresh(log_dir=f / "sub", runner=ok_runner(calls),
                       health_probe=health_all_ok, lock=False)
    assert s["overall"] == "failed" and s["exit_code"] == 1
    assert calls == []  # no steps run


def test_summary_and_per_step_logs_written(tmp_path):
    rr.run_refresh(log_dir=tmp_path, runner=ok_runner([]), health_probe=health_all_ok, lock=False)
    assert (tmp_path / "summary.json").is_file()
    assert (tmp_path / "00-generate_daily_session_card.py.out").is_file()
    assert (tmp_path / "00-generate_daily_session_card.py.err").is_file()
    data = json.loads((tmp_path / "summary.json").read_text())
    assert data["overall"] == "ok"


def test_idempotent_two_runs(tmp_path):
    a = rr.run_refresh(log_dir=tmp_path / "a", runner=ok_runner([]), health_probe=health_all_ok, lock=False)
    b = rr.run_refresh(log_dir=tmp_path / "b", runner=ok_runner([]), health_probe=health_all_ok, lock=False)
    assert a["overall"] == b["overall"] == "ok"


# --- live integration (deselected by default) ------------------------------
@pytest.mark.integration
def test_live_end_to_end():
    importlib.reload(rr)
    s = rr.run_refresh()  # real generators + real health, real lock
    assert s["exit_code"] in (0, 2)
    assert s["overall"] in ("ok", "degraded")
