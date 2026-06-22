"""
Hermetic tests for generate_agent_boot_pack.py (Phase 3).

Points the canonical root and timeline dirs at tmp locations, synthesizes
timeline daily files + session-card roll-ups, and asserts the boot pack assembles
correctly, degrades loudly, bounds size, and is idempotent. No real workspace
state touched.
"""
from __future__ import annotations

import importlib
from pathlib import Path

import pytest


@pytest.fixture()
def env(tmp_path, monkeypatch):
    root = tmp_path / "canonical"
    monkeypatch.setenv("OPENCLAW_CANONICAL_MEMORY_ROOT", str(root))
    import canonical_memory_paths as cmp

    importlib.reload(cmp)
    import generate_agent_boot_pack as g

    importlib.reload(g)
    # redirect timeline dirs into tmp
    tl = tmp_path / "timeline"
    (tl / "daily").mkdir(parents=True)
    g.TIMELINE_DIR = tl
    g.TIMELINE_DAILY = tl / "daily"
    g.cmp.scaffold()
    return g, cmp, root, tmp_path, tl


def _write_timeline(tl: Path, day: str, text: str):
    (tl / "daily" / f"{day}.md").write_text(text, encoding="utf-8")


def _write_agent_rollup(cmp, agent: str, day: str, text: str):
    p = cmp.daily_agent_card(agent, day)
    p.write_text(text, encoding="utf-8")
    return p


def test_fresh_boot_pack_assembles_all_sections(env):
    g, cmp, root, tmp, tl = env
    day = cmp.today_utc()
    _write_timeline(tl, day, "- [09:00] USER: did we ship the PR to Engram?\n- [09:01] ASSISTANT: yes")
    _write_agent_rollup(cmp, "builder", day,
                        "# Daily roll-up: builder\n## Session abc\n## Files changed\n- /x/y.py")
    pack = g.build_boot_pack("builder")
    assert pack.freshness == g.FRESH
    rendered = g.render_boot_pack(pack)
    assert "## 1. System Timeline" in rendered
    assert "## 2. Your Recent Activity" in rendered
    assert "PR to Engram" in rendered
    assert "/x/y.py" in rendered
    assert "## Provenance" in rendered
    # both sources ok
    assert all(s.ok for s in pack.sources)


def test_degraded_when_no_sources(env):
    g, cmp, root, tmp, tl = env
    # no timeline files, no agent rollup
    pack = g.build_boot_pack("ghost")
    assert pack.freshness == g.DEGRADED
    rendered = g.render_boot_pack(pack)
    assert "DEGRADED MEMORY" in rendered
    assert "DO NOT GUESS" in rendered
    assert pack.degraded_reasons


def test_partial_timeline_only_is_fresh_but_notes_gap(env):
    g, cmp, root, tmp, tl = env
    day = cmp.today_utc()
    _write_timeline(tl, day, "- [10:00] ASSISTANT: working on memory upgrade")
    pack = g.build_boot_pack("builder")  # no card rollup
    assert pack.freshness == g.FRESH  # timeline alone is enough to orient
    rendered = g.render_boot_pack(pack)
    assert "No session-card roll-up" in rendered


def test_stale_card_fallback_is_flagged_not_silent(env):
    """Tester-found honesty gap: today's timeline + only an OLD card must NOT be
    presented as 'Recent Activity' freshly. It must be loudly marked stale."""
    g, cmp, root, tmp, tl = env
    today = cmp.today_utc()
    _write_timeline(tl, today, "- [09:00] ASSISTANT: working today")
    # only an old roll-up exists (e.g. 3 weeks ago)
    _write_agent_rollup(cmp, "builder", "2026-06-01",
                        "# old roll-up\n## Session zzz\n- did old thing")
    pack = g.build_boot_pack("builder")
    assert pack.cards_stale is True
    assert pack.cards_day == "2026-06-01"
    rendered = g.render_boot_pack(pack)
    assert "STALE" in rendered
    assert "2026-06-01" in rendered
    # the stale card source must not count as a healthy/current source
    cards_src = next(s for s in pack.sources if s.name == "session_cards")
    assert cards_src.ok is False
    assert "STALE" in cards_src.note
    # there IS a current timeline, so overall still fresh — but the gap is recorded
    assert pack.freshness == g.FRESH
    assert any("stale" in r.lower() for r in pack.degraded_reasons)


def test_only_stale_card_no_timeline_is_degraded(env):
    """If the ONLY source is a stale card (no timeline), the pack is degraded —
    a stale card alone must not mask total lack of current orientation."""
    g, cmp, root, tmp, tl = env
    _write_agent_rollup(cmp, "builder", "2026-06-01", "# old\n- ancient")
    pack = g.build_boot_pack("builder")
    assert pack.cards_stale is True
    assert pack.freshness == g.DEGRADED
    assert "DEGRADED MEMORY" in g.render_boot_pack(pack)


def test_current_day_card_not_marked_stale(env):
    g, cmp, root, tmp, tl = env
    today = cmp.today_utc()
    _write_timeline(tl, today, "- x")
    _write_agent_rollup(cmp, "builder", today, "# today\n- fresh work")
    pack = g.build_boot_pack("builder")
    assert pack.cards_stale is False
    assert pack.cards_day == today
    assert "STALE" not in g.render_boot_pack(pack)


def test_falls_back_to_latest_md_when_no_daily(env):
    g, cmp, root, tmp, tl = env
    (tl / "latest.md").write_text("- fallback timeline content here", encoding="utf-8")
    pack = g.build_boot_pack("builder")
    assert any(s.name == "system_timeline" and s.ok for s in pack.sources)
    assert "fallback timeline content" in g.render_boot_pack(pack)


def test_timeline_is_clipped_to_recent_tail(env):
    g, cmp, root, tmp, tl = env
    day = cmp.today_utc()
    big = "\n".join(f"- line {i}" for i in range(5000))  # > MAX_TIMELINE_CHARS
    _write_timeline(tl, day, big + "\n- LAST_RECENT_LINE")
    pack = g.build_boot_pack("builder")
    src = next(s for s in pack.sources if s.name == "system_timeline")
    assert src.clipped
    assert src.chars <= g.MAX_TIMELINE_CHARS
    # tail kept: the most recent line survives, an early one is dropped
    assert "LAST_RECENT_LINE" in pack.timeline_text
    assert "- line 0" not in pack.timeline_text


def test_write_and_idempotent(env):
    g, cmp, root, tmp, tl = env
    day = cmp.today_utc()
    _write_timeline(tl, day, "- something happened")
    p1 = g.write_boot_pack(g.build_boot_pack("builder"))
    p2 = g.write_boot_pack(g.build_boot_pack("builder"))
    assert p1 == p2 == g.boot_pack_path("builder")
    # only one boot pack header; no duplicate accretion
    assert p2.read_text().count("# Boot Pack:") == 1


def test_path_traversal_safe(env):
    g, cmp, root, tmp, tl = env
    p = g.boot_pack_path("../../etc")
    resolved = p.resolve()
    assert str(root.resolve()) in str(resolved)
    assert resolved.name == "_BOOT_PACK.md"


def test_frontmatter_well_formed(env):
    g, cmp, root, tmp, tl = env
    day = cmp.today_utc()
    _write_timeline(tl, day, "- x")
    rendered = g.render_boot_pack(g.build_boot_pack("builder"))
    lines = rendered.splitlines()
    assert lines[0] == "---"
    fm_end = lines.index("---", 1)
    fm = "\n".join(lines[1:fm_end])
    for key in ("agent:", "generated_at:", "freshness:", "sources_ok:"):
        assert key in fm


def test_project_section_included_when_requested(env, monkeypatch):
    g, cmp, root, tmp, tl = env
    day = cmp.today_utc()
    _write_timeline(tl, day, "- x")

    # stub project_memory_paths so we don't depend on real project state
    import types

    fake = types.SimpleNamespace(
        normalize_project_id=lambda x: x,
        get_project_current_file=lambda pid: _mk(tmp / "proj_current.md",
                                                 "PROJECT STATE: phase 3 in progress"),
        get_latest_project_daily_file=lambda pid: None,
    )
    monkeypatch.setitem(__import__("sys").modules, "project_memory_paths", fake)
    pack = g.build_boot_pack("builder", project_id="memory")
    rendered = g.render_boot_pack(pack)
    assert "## 3. Project Context" in rendered
    assert "PROJECT STATE: phase 3" in rendered


def _mk(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path
