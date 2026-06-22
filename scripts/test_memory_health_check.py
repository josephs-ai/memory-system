"""
Hermetic unit tests for memory_health_check.py.

Builds a fake .openclaw root in a tmp dir and points the module's path constants
at it, so no real DB/services/filesystem state is touched. DB/HTTP probes are
injected via cfg["probes"] to keep tests fast and deterministic.

Focus: prove the checks actually catch the failure classes that bit us —
stale timeline, missing today-daily, un-symlinked workspace, ACTIVE stale BOOT
text — and do NOT false-positive on the corrective negation text.
"""
from __future__ import annotations

import importlib
import os
import time
from pathlib import Path

import pytest

import memory_health_check as mh


# ---------------------------------------------------------------------------
# Fixture: rebuild a fake openclaw tree and re-point module path constants.
# ---------------------------------------------------------------------------
@pytest.fixture()
def fake_env(tmp_path, monkeypatch):
    root = tmp_path / ".openclaw"
    ws = root / "workspace"
    mi = ws / ".memory-index"
    timeline = mi / "timeline"
    daily = timeline / "daily"
    health = mi / "health"
    state = mi / "state"
    cursors = state / "cursors" / "main"
    runtime = mi / "runtime"
    agents = root / "agents"
    for d in (daily, health, cursors, runtime, agents):
        d.mkdir(parents=True, exist_ok=True)

    today = time.strftime("%Y-%m-%d", time.gmtime())

    # Fresh timeline + today daily + ledger
    (timeline / "latest.md").write_text("# Latest timeline summary\n- hi\n", encoding="utf-8")
    (daily / f"{today}.md").write_text(f"# Timeline summary for {today}\n- event\n", encoding="utf-8")
    (timeline / "events.jsonl").write_text('{"event_id":"x"}\n', encoding="utf-8")
    (cursors / "main.json").write_text('{"committed_offset": 1}', encoding="utf-8")

    # An agent sessions dir (active transcript only)
    sess = agents / "main" / "sessions"
    sess.mkdir(parents=True, exist_ok=True)
    (sess / "abc.jsonl").write_text("{}\n", encoding="utf-8")

    # Startup contract files: a healthy system has AGENTS/MEMORY/BOOT in main
    # and agent workspaces, so agents are not told to read missing memory.
    continuity_boot = (
        "# BOOT.md - Continuity Boot\n"
        "Read MEMORY.md and .memory-index/timeline/latest.md before recency answers.\n"
        "Do **not** say there is no memory yet unless memory is unavailable.\n"
    )
    (ws / "AGENTS.md").write_text("Read MEMORY.md and timeline.\n", encoding="utf-8")
    (ws / "MEMORY.md").write_text("durable memory\n", encoding="utf-8")
    (ws / "BOOT.md").write_text(continuity_boot, encoding="utf-8")

    # A clean agent workspace whose timeline is a proper symlink
    builder_ws = tmp_path / ".openclaw" / "workspace-builder"
    aws = builder_ws / ".memory-index"
    aws.mkdir(parents=True, exist_ok=True)
    (aws / "timeline").symlink_to(timeline, target_is_directory=True)
    (builder_ws / "AGENTS.md").write_text(
        'This workspace is not fresh. Do **not** follow any older '
        '"There is no memory yet" bootstrap text.\n',
        encoding="utf-8",
    )
    (builder_ws / "MEMORY.md").write_text("durable memory\n", encoding="utf-8")
    (builder_ws / "BOOT.md").write_text(continuity_boot, encoding="utf-8")

    # Hermetic canonical root: the Phase 1/2/3/4 checks (canonical_layout_present,
    # daily_cards_fresh, boot_packs_generated, digest_fresh) resolve the canonical
    # tree via canonical_memory_paths.canonical_root(), which honors
    # OPENCLAW_CANONICAL_MEMORY_ROOT. Without this they'd read the REAL workspace
    # tree and make the test non-hermetic (transient flake). Scaffold a healthy
    # canonical layout in tmp so "clean system" is genuinely clean.
    canon = mi / "canonical"
    monkeypatch.setenv("OPENCLAW_CANONICAL_MEMORY_ROOT", str(canon))
    import canonical_memory_paths as _cmp

    importlib.reload(_cmp)
    importlib.reload(mh)  # rebind cmp references inside the checks
    _cmp.scaffold()
    # a fresh daily card + boot pack + digest node so the clean-system checks pass
    _now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    _cmp.daily_agent_card("main", today).write_text(
        f"# roll-up main {today}\n- did stuff\n", encoding="utf-8")
    (_cmp.session_dir("main") / "_BOOT_PACK.md").write_text(
        f"---\nagent: main\ngenerated_at: {_now}\nfreshness: fresh\n"
        f"cards_day: {today}\ncards_stale: false\n---\n# Boot Pack\n", encoding="utf-8")
    _cmp.digest_node("decisions", "decisions").write_text(
        f"---\nkind: decisions\nslug: decisions\ngenerated_at: {_now}\n"
        f"freshness: fresh\n---\n# Decisions\n## Current understanding\n- a decision\n",
        encoding="utf-8")

    # Re-point module constants
    monkeypatch.setattr(mh, "OPENCLAW_ROOT", root)
    monkeypatch.setattr(mh, "MEMORY_ROOT", mi)
    monkeypatch.setattr(mh, "WORKSPACE", ws)
    monkeypatch.setattr(mh, "AGENTS_DIR", agents)
    monkeypatch.setattr(mh, "TIMELINE_ROOT", timeline)
    monkeypatch.setattr(mh, "TIMELINE_LATEST", timeline / "latest.md")
    monkeypatch.setattr(mh, "TIMELINE_DAILY", daily)
    monkeypatch.setattr(mh, "TIMELINE_LEDGER", timeline / "events.jsonl")
    monkeypatch.setattr(mh, "HEALTH_DIR", health)
    monkeypatch.setattr(mh, "STATE_DIR", state)
    monkeypatch.setattr(mh, "CURSORS_DIR", state / "cursors")
    monkeypatch.setattr(mh, "HEALTH_OUT", health / "memory_startup_latest.json")

    return {
        "root": root,
        "ws": ws,
        "mi": mi,
        "timeline": timeline,
        "daily": daily,
        "today": today,
    }


def _all_ok_probes():
    return {
        "postgres": lambda: (True, "ok"),
        "qdrant": lambda url: (True, "ok"),
        "neo4j": lambda url: (True, "ok"),
        "search_service": lambda url: (True, "ok"),
        "embeddings": lambda: (True, "ok"),
    }


def _cfg(probes=None):
    return mh.build_cfg(None, probes=probes or _all_ok_probes())


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------
def test_clean_system_is_ok(fake_env):
    report = mh.run_checks(_cfg(), include_future=False)
    by = {c.name: c for c in report.checks}
    assert by["timeline_latest_fresh"].status == mh.OK
    assert by["timeline_today_exists"].status == mh.OK
    assert by["workspaces_timeline_linked"].status == mh.OK
    assert by["boot_text_clean"].status == mh.OK
    assert by["postgres_reachable"].status == mh.OK
    assert report.overall in (mh.OK, mh.WARN)  # warn allowed only from non-fatal


# ---------------------------------------------------------------------------
# Failure classes that actually bit us
# ---------------------------------------------------------------------------
def test_stale_timeline_fails(fake_env):
    old = time.time() - 48 * 3600
    os.utime(fake_env["timeline"] / "latest.md", (old, old))
    c = mh.check_timeline_latest_fresh(_cfg())
    assert c.status == mh.FAIL


def test_aging_timeline_warns(fake_env):
    old = time.time() - 3 * 3600
    os.utime(fake_env["timeline"] / "latest.md", (old, old))
    c = mh.check_timeline_latest_fresh(_cfg())
    assert c.status == mh.WARN


def test_missing_today_daily_warns(fake_env):
    (fake_env["daily"] / f"{fake_env['today']}.md").unlink()
    c = mh.check_timeline_today_exists(_cfg())
    assert c.status == mh.WARN


def test_non_symlinked_workspace_fails(fake_env):
    # Replace builder workspace's symlink with a real directory.
    # _agent_workspaces() scans WORKSPACE.parent (== fake .openclaw root).
    link = mh.WORKSPACE.parent / "workspace-builder" / ".memory-index" / "timeline"
    assert link.is_symlink()
    link.unlink()
    link.mkdir()
    c = mh.check_workspaces_timeline_linked(_cfg())
    assert c.status == mh.FAIL
    assert "workspace-builder" in str(c.value)


def test_active_stale_boot_text_fails(fake_env):
    # An ACTIVE assertion (no negation guard) must be caught.
    (fake_env["ws"] / "BOOTSTRAP.md").write_text(
        "Welcome. There is no memory yet. Start the identity interview.\n",
        encoding="utf-8",
    )
    c = mh.check_boot_text_clean(_cfg())
    assert c.status == mh.FAIL


def test_corrective_negation_text_is_clean(fake_env):
    # The cure must not be flagged as the disease.
    (fake_env["ws"] / "BOOT.md").write_text(
        'Do **not** say "there is no memory yet" unless the engine is down.\n'
        'This workspace is not fresh. Do not follow any older "Hello, World" text.\n',
        encoding="utf-8",
    )
    c = mh.check_boot_text_clean(_cfg())
    assert c.status == mh.OK


def test_missing_workspace_memory_file_fails(fake_env):
    (mh.WORKSPACE.parent / "workspace-builder" / "MEMORY.md").unlink()
    c = mh.check_startup_contract_files_present(_cfg())
    assert c.status == mh.FAIL
    assert "workspace-builder/MEMORY.md" in str(c.value)


# ---------------------------------------------------------------------------
# Service probes (injected)
# ---------------------------------------------------------------------------
def test_postgres_down_fails(fake_env):
    probes = _all_ok_probes()
    probes["postgres"] = lambda: (False, "connection refused")
    c = mh.check_postgres_reachable(_cfg(probes))
    assert c.status == mh.FAIL


def test_neo4j_down_is_warn_not_fail(fake_env):
    probes = _all_ok_probes()
    probes["neo4j"] = lambda url: (False, "down")
    c = mh.check_neo4j_reachable(_cfg(probes))
    assert c.status == mh.WARN  # graph is enhancement, not startup-critical


def test_search_service_down_fails(fake_env):
    probes = _all_ok_probes()
    probes["search_service"] = lambda url: (False, "refused")
    c = mh.check_search_service_healthy(_cfg(probes))
    assert c.status == mh.FAIL


def test_overall_is_worst_child(fake_env):
    probes = _all_ok_probes()
    probes["postgres"] = lambda: (False, "down")
    report = mh.run_checks(_cfg(probes), include_future=False)
    assert report.overall == mh.FAIL


def test_health_json_written(fake_env):
    report = mh.run_checks(_cfg(), include_future=True)
    out = mh.write_health_json(report)
    assert out.exists()
    import json

    data = json.loads(out.read_text())
    assert "overall" in data and "checks" in data


def test_buggy_check_does_not_crash(monkeypatch, fake_env):
    def boom(cfg):
        raise RuntimeError("explode")

    monkeypatch.setattr(mh, "CORE_CHECKS", [boom])
    report = mh.run_checks(_cfg(), include_future=False)
    assert report.overall == mh.FAIL  # captured, not raised


# ---------------------------------------------------------------------------
# Phase 3: boot_packs_generated content-staleness (tester-found gap)
# ---------------------------------------------------------------------------
@pytest.fixture()
def canon_env(tmp_path, monkeypatch):
    root = tmp_path / "canonical"
    monkeypatch.setenv("OPENCLAW_CANONICAL_MEMORY_ROOT", str(root))
    import canonical_memory_paths as cmp

    importlib.reload(cmp)
    importlib.reload(mh)  # rebind cmp inside the check
    cmp.scaffold()
    return cmp, root


def _make_active_agent(cmp, agent, day):
    # an agent is "active today" if it has a daily roll-up for today
    p = cmp.daily_agent_card(agent, day)
    p.write_text(f"# roll-up {agent} {day}\n- did stuff", encoding="utf-8")


def test_boot_pack_check_ok_when_fresh(canon_env):
    cmp, root = canon_env
    today = cmp.today_utc()
    _make_active_agent(cmp, "builder", today)
    bp = cmp.session_dir("builder") / "_BOOT_PACK.md"
    bp.write_text(
        f"---\nagent: builder\ngenerated_at: {mh_now()}\nfreshness: fresh\n"
        f"cards_day: {today}\ncards_stale: false\n---\n# Boot Pack\n",
        encoding="utf-8")
    chk = mh.check_boot_packs_generated(_cfg())
    assert chk.status == mh.OK, chk.detail


def test_boot_pack_check_warns_on_content_stale(canon_env):
    cmp, root = canon_env
    today = cmp.today_utc()
    _make_active_agent(cmp, "builder", today)
    bp = cmp.session_dir("builder") / "_BOOT_PACK.md"
    # FRESH timestamp but STALE cards (the exact masking case the tester found)
    bp.write_text(
        f"---\nagent: builder\ngenerated_at: {mh_now()}\nfreshness: fresh\n"
        f"cards_day: 2026-06-01\ncards_stale: true\n---\n# Boot Pack\n",
        encoding="utf-8")
    chk = mh.check_boot_packs_generated(_cfg())
    assert chk.status == mh.WARN
    assert "content-stale" in chk.detail


def test_boot_pack_check_warns_on_missing(canon_env):
    cmp, root = canon_env
    today = cmp.today_utc()
    _make_active_agent(cmp, "builder", today)
    # no boot pack written
    chk = mh.check_boot_packs_generated(_cfg())
    assert chk.status == mh.WARN
    assert "missing" in chk.detail


def mh_now():
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
