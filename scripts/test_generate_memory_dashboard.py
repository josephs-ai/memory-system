#!/usr/bin/env python3
"""Hermetic tests for the Phase 6 memory control-panel dashboard."""
from __future__ import annotations

import importlib
import json
import sys
import time
from pathlib import Path

import pytest

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))


@pytest.fixture()
def env(tmp_path, monkeypatch):
    canon = tmp_path / "canonical"
    monkeypatch.setenv("OPENCLAW_CANONICAL_MEMORY_ROOT", str(canon))
    import canonical_memory_paths as cmp
    importlib.reload(cmp)
    import generate_memory_dashboard as gmd
    importlib.reload(gmd)
    cmp.scaffold()
    return cmp, gmd, canon


_NOW = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _session(cmp, agent, sid, day, *, boot=False):
    (cmp.session_dir(agent) / f"{sid}.md").write_text(
        f"---\nagent: {agent}\nsession_id: {sid}\nday: {day}\n"
        f"freshness: fresh\ngenerated_at: {_NOW}\n---\n# Session Card\n## Summary\n- did x\n",
        encoding="utf-8")
    if boot:
        (cmp.session_dir(agent) / "_BOOT_PACK.md").write_text(
            f"---\nagent: {agent}\ngenerated_at: {_NOW}\nfreshness: fresh\n---\n# Boot Pack\n",
            encoding="utf-8")


def _digest(cmp, kind, slug):
    cmp.digest_node(kind, slug).write_text(
        f"---\nkind: {kind}\nslug: {slug}\nfreshness: fresh\ngenerated_at: {_NOW}\n"
        f"source_through: 2026-06-22\n---\n# {kind}: {slug}\n## Current understanding\n- a thing\n",
        encoding="utf-8")


def _daily(cmp, agent, day):
    cmp.daily_agent_card(agent, day).write_text(
        f"# Daily roll-up: {agent} — {day}\n_Generated {_NOW}_\n## Summary\n- y\n", encoding="utf-8")


def _full_tree(cmp):
    _session(cmp, "builder", "s1", "2026-06-22", boot=True)
    _daily(cmp, "builder", "2026-06-22")
    _digest(cmp, "decisions", "decisions")
    _digest(cmp, "bugs", "blockers")
    # card index
    idx = cmp.metadata_dir("indexes") / "card_index.jsonl"
    meta = {"_index_meta": {"records": 4, "by_kind": {"session": 1, "daily": 1, "digest": 2},
                            "generated_at": _NOW}}
    idx.write_text(json.dumps(meta) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
def test_dashboard_truthful_counts(env):
    cmp, gmd, canon = env
    _full_tree(cmp)
    data = gmd.build_dashboard(cached=False)
    inv = data["inventory"]
    assert inv["layout_present"] is True
    agents = {a["agent"]: a for a in inv["agents"]}
    assert agents["builder"]["session_cards"] == 1
    assert agents["builder"]["boot_pack"] is True
    stages = {s["name"]: s for s in inv["stages"]}
    assert stages["session cards"]["count"] == 1
    assert stages["digest nodes"]["count"] == 2
    assert stages["card index"]["count"] == 4
    # digests surfaced as link targets
    links = {d["wikilink"] for d in inv["digests"]}
    assert "decisions/decisions" in links and "bugs/blockers" in links


def test_markdown_links_and_sections(env):
    cmp, gmd, canon = env
    _full_tree(cmp)
    md = gmd.render_markdown(gmd.build_dashboard(cached=False))
    assert "# 🧠 Memory Control Panel" in md
    assert "[[decisions/decisions]]" in md
    assert "## 🔄 Pipeline freshness" in md
    assert "## 👤 Agents" in md


def test_missing_stage_surfaced(env):
    """A missing pipeline stage must show as missing, not be hidden."""
    cmp, gmd, canon = env
    _session(cmp, "builder", "s1", "2026-06-22", boot=True)
    # no digests, no daily, no index
    md = gmd.render_markdown(gmd.build_dashboard(cached=False))
    assert "🔴 missing — **digest nodes**" in md
    assert "🔴 missing — **card index**" in md
    assert "no digest nodes yet" in md


def test_stale_stage_flagged(env):
    cmp, gmd, canon = env
    old = "2026-01-01T00:00:00Z"
    (cmp.session_dir("builder") / "s1.md").write_text(
        f"---\nagent: builder\nday: 2026-01-01\nfreshness: stale\ngenerated_at: {old}\n---\n# c\n- x\n",
        encoding="utf-8")
    data = gmd.build_dashboard(cached=False)
    stages = {s["name"]: s for s in data["inventory"]["stages"]}
    assert stages["session cards"]["stale"] is True


def test_problems_surfaced_at_top(env):
    """WARN/FAIL checks get a dedicated 'Needs attention' section."""
    cmp, gmd, canon = env
    _full_tree(cmp)
    # craft a health dict with a WARN by NOT generating a dashboard (dashboard_fresh WARN)
    data = gmd.build_dashboard(cached=False)
    # inject a synthetic failing check to exercise rendering deterministically
    data["health"]["checks"].append({"name": "synthetic_warn", "status": "warn",
                                      "detail": "synthetic", "value": None, "source": ""})
    md = gmd.render_markdown(data)
    assert "## ⚠️ Needs attention" in md
    assert "synthetic_warn" in md.split("## 🔄")[0]  # appears before pipeline section


def test_cached_fallback_flagged(env, monkeypatch):
    """When in-process health fails, fall back to cached JSON and FLAG it."""
    cmp, gmd, canon = env
    _full_tree(cmp)
    import memory_health_check as mh
    # write a cached health report
    cached = {"generated_at": "2026-06-22T00:00:00Z", "overall": "ok", "checks": []}
    Path(mh.HEALTH_OUT).parent.mkdir(parents=True, exist_ok=True)
    Path(mh.HEALTH_OUT).write_text(json.dumps(cached), encoding="utf-8")
    # force in-process run to fail
    monkeypatch.setattr(mh, "run_checks", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    data = gmd.build_dashboard(cached=False)
    assert any("CACHED" in n or "cached" in n for n in data["notes"])
    assert data["health"]["overall"] == "ok"


def test_no_layout_graceful(env, monkeypatch):
    cmp, gmd, canon = env
    # don't scaffold extra; remove the scaffolded marker by pointing at empty dir
    import canonical_memory_paths as cmp2
    monkeypatch.setattr(cmp2, "layout_present", lambda: False)
    importlib.reload(gmd)
    # gmd imported cmp at module load; patch its reference too
    monkeypatch.setattr(gmd.cmp, "layout_present", lambda: False)
    data = gmd.build_dashboard(cached=False)
    assert data["inventory"]["layout_present"] is False
    md = gmd.render_markdown(data)
    assert "not scaffolded" in md


def test_stray_empty_agent_dir_skipped(env):
    """An empty session/<x> dir (no cards, no boot) is not a real agent."""
    cmp, gmd, canon = env
    _session(cmp, "builder", "s1", "2026-06-22", boot=True)
    (cmp.session_dir("etc"))  # creates empty dir via ensure
    data = gmd.build_dashboard(cached=False)
    agents = {a["agent"] for a in data["inventory"]["agents"]}
    assert "builder" in agents and "etc" not in agents


def test_write_and_idempotent(env):
    cmp, gmd, canon = env
    _full_tree(cmp)
    data = gmd.build_dashboard(cached=False)
    md1, js1 = gmd.write_dashboard(data)
    assert md1.is_file() and js1.is_file()
    # re-render the SAME data -> identical markdown (generated_at fixed in data)
    assert gmd.render_markdown(data) == gmd.render_markdown(data)


def test_dashboard_health_check(env):
    cmp, gmd, canon = env
    _full_tree(cmp)
    import memory_health_check as mh
    importlib.reload(mh)
    cfg = mh.build_cfg(None, probes={})
    # before generating: WARN missing
    c = mh.check_dashboard_fresh(cfg)
    assert c.status == mh.WARN
    # after generating: OK
    gmd.write_dashboard(gmd.build_dashboard(cached=False))
    c2 = mh.check_dashboard_fresh(cfg)
    assert c2.status == mh.OK
