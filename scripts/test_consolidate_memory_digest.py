"""
Hermetic tests for consolidate_memory_digest.py (Phase 4 auto_dream).

Synthesizes canonical daily agent roll-ups across multiple days under a tmp
canonical root, runs auto_dream, and asserts: correct kind routing, bullet dedup
across days, idempotent upsert (no duplicated understanding), wikilink rendering,
provenance/freshness frontmatter, the digest index, and that unroutable items go
to systems/misc (never dropped). No real workspace state touched.
"""
from __future__ import annotations

import importlib

import pytest


@pytest.fixture()
def env(tmp_path, monkeypatch):
    root = tmp_path / "canonical"
    monkeypatch.setenv("OPENCLAW_CANONICAL_MEMORY_ROOT", str(root))
    import canonical_memory_paths as cmp

    importlib.reload(cmp)
    import consolidate_memory_digest as d

    importlib.reload(d)
    d.cmp.scaffold()
    return d, cmp, root


def _rollup(cmp, agent, day, body):
    p = cmp.daily_agent_card(agent, day)
    p.write_text(body, encoding="utf-8")
    return p


_DAY1 = """# Daily roll-up: builder — 2026-06-20
## Decisions
- (user) ship the PR to Engram
- (did) Fixed the discard-ban bug
## Blockers
- push failed: no API token
## Tests run
- pytest -q test_thing.py
## Files changed
- /x/checkpoint.py
"""

_DAY2 = """# Daily roll-up: builder — 2026-06-21
## Decisions
- (user) ship the PR to Engram
- (did) Added the boot pack generator
## Blockers
- timeline stale
"""


def test_routing_into_kinds(env):
    d, cmp, root = env
    _rollup(cmp, "builder", "2026-06-20", _DAY1)
    res = d.run_auto_dream(since_days=3650, agent="builder")
    kinds = {(n["kind"], n["slug"]) for n in res["nodes"]}
    assert ("decisions", "decisions") in kinds
    assert ("bugs", "blockers") in kinds
    assert ("systems", "tests-and-tooling") in kinds
    assert ("systems", "code-changes") in kinds


def test_dedup_across_days(env):
    d, cmp, root = env
    _rollup(cmp, "builder", "2026-06-20", _DAY1)
    _rollup(cmp, "builder", "2026-06-21", _DAY2)
    d.run_auto_dream(since_days=3650, agent="builder")
    node = cmp.digest_node("decisions", "decisions", ensure=False)
    text = node.read_text()
    # the duplicate "ship the PR to Engram" appears once, both unique (did)s present
    assert text.count("ship the PR to Engram") == 1
    assert "Fixed the discard-ban bug" in text
    assert "Added the boot pack generator" in text
    # both days recorded in provenance
    assert "2026-06-20" in text and "2026-06-21" in text


def test_idempotent_rerun(env):
    d, cmp, root = env
    _rollup(cmp, "builder", "2026-06-20", _DAY1)
    d.run_auto_dream(since_days=3650, agent="builder")
    node = cmp.digest_node("decisions", "decisions", ensure=False)
    first = node.read_text()
    n1 = first.count("\n- ")
    d.run_auto_dream(since_days=3650, agent="builder")
    second = node.read_text()
    n2 = second.count("\n- ")
    assert n1 == n2, "re-run must not duplicate understanding"


def test_wikilinks_rendered(env):
    d, cmp, root = env
    _rollup(cmp, "builder", "2026-06-20", _DAY1)
    d.run_auto_dream(since_days=3650, agent="builder")
    # decisions/decisions and decisions/open-threads or other decisions nodes link
    node = cmp.digest_node("systems", "code-changes", ensure=False)
    text = node.read_text()
    # systems/code-changes should link to systems/tests-and-tooling (same kind)
    assert "[[systems/tests-and-tooling]]" in text
    # frontmatter links line is not double-bracketed into [[[ ]]]
    assert "[[[" not in text


def test_unroutable_goes_to_misc_not_dropped(env):
    d, cmp, root = env
    body = "# roll-up\n## SomethingWeird\n- a totally generic note with no keywords xyzzy\n"
    _rollup(cmp, "builder", "2026-06-20", body)
    d.run_auto_dream(since_days=3650, agent="builder")
    node = cmp.digest_node("systems", "misc", ensure=False)
    assert node.exists()
    assert "xyzzy" in node.read_text()


def test_provenance_sections_do_not_pollute_digests(env):
    """Tester-found bug: daily roll-ups inline each session card's ## Provenance
    (transcript path, event count, span). Those machine-metadata bullets must NOT
    flow into systems/misc or any digest node."""
    d, cmp, root = env
    body = (
        "# Daily roll-up: builder — 2026-06-20\n"
        "## Session abc123  (freshness: fresh)\n"
        "# Session Card: builder — 2026-06-20\n"
        "## Decisions\n"
        "- (did) shipped phase 4\n"
        "## Provenance\n"
        "- transcript: /home/x/.openclaw/agents/builder/sessions/abc123.jsonl\n"
        "- events: 352\n"
        "- span: 2026-06-20T08:00Z .. 2026-06-20T09:00Z\n"
    )
    _rollup(cmp, "builder", "2026-06-20", body)
    d.run_auto_dream(since_days=3650, agent="builder")
    # the real decision survives
    dec = cmp.digest_node("decisions", "decisions", ensure=False)
    assert "shipped phase 4" in dec.read_text()
    # provenance noise must NOT appear anywhere in any digest node
    for kind in cmp.DIGEST_KINDS:
        ddir = cmp.digest_dir(kind, ensure=False)
        if not ddir.is_dir():
            continue
        for node in ddir.glob("*.md"):
            text = node.read_text()
            # strip the node's OWN provenance section before asserting (that's
            # legitimate, structural — not inlined card metadata)
            understanding = text.split("## Provenance")[0]
            assert "abc123.jsonl" not in understanding
            assert "events: 352" not in understanding
            assert ".. 2026-06-20T09:00Z" not in understanding


def test_ignored_section_variants(env):
    """Defense-in-depth: Provenance heading variants must also be ignored."""
    d, cmp, root = env
    for variant in ("Provenance", "Provenance:", "Provenance notes", "## PROVENANCE"):
        assert d._is_ignored_section(variant) is True, variant
    for live in ("Decisions", "Blockers", "Summary"):
        assert d._is_ignored_section(live) is False, live


def test_frontmatter_and_freshness(env):
    d, cmp, root = env
    _rollup(cmp, "builder", "2026-06-20", _DAY1)
    d.run_auto_dream(since_days=3650, agent="builder")
    node = cmp.digest_node("decisions", "decisions", ensure=False)
    lines = node.read_text().splitlines()
    assert lines[0] == "---"
    fm = "\n".join(lines[1:lines.index("---", 1)])
    for key in ("kind:", "slug:", "generated_at:", "freshness:", "source_through:",
                "source_days:", "source_cards:", "links:"):
        assert key in fm
    assert "freshness: fresh" in fm


def test_digest_index_written(env):
    d, cmp, root = env
    _rollup(cmp, "builder", "2026-06-20", _DAY1)
    res = d.run_auto_dream(since_days=3650, agent="builder")
    idx = cmp.metadata_dir("catalogs", ensure=False) / "digest_index.md"
    assert idx.exists()
    text = idx.read_text()
    assert "[[decisions/decisions]]" in text
    assert "# Digest Index" in text


def test_dry_run_writes_nothing(env):
    d, cmp, root = env
    _rollup(cmp, "builder", "2026-06-20", _DAY1)
    res = d.run_auto_dream(since_days=3650, agent="builder", dry_run=True)
    assert res["dry_run"] is True
    assert not cmp.digest_node("decisions", "decisions", ensure=False).exists()


def test_window_excludes_old_rollups(env):
    d, cmp, root = env
    # one ancient rollup, one recent
    _rollup(cmp, "builder", "2000-01-01", "# old\n## Decisions\n- ancient decision\n")
    today = cmp.today_utc()
    _rollup(cmp, "builder", today, "# new\n## Decisions\n- recent decision\n")
    d.run_auto_dream(since_days=7, agent="builder")
    text = cmp.digest_node("decisions", "decisions", ensure=False).read_text()
    assert "recent decision" in text
    assert "ancient decision" not in text
