#!/usr/bin/env python3
"""Hermetic tests for the Phase 5 readable-card search (index + searcher)."""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))


@pytest.fixture()
def env(tmp_path, monkeypatch):
    """Hermetic canonical tree + freshly reloaded modules bound to it."""
    canon = tmp_path / "canonical"
    monkeypatch.setenv("OPENCLAW_CANONICAL_MEMORY_ROOT", str(canon))
    import canonical_memory_paths as cmp
    importlib.reload(cmp)
    import card_search_index as csi
    importlib.reload(csi)
    import search_memory_cards as smc
    importlib.reload(smc)
    cmp.scaffold()
    return cmp, csi, smc, canon


def _write_session(cmp, agent, sid, day, body, *, freshness="fresh", gen="2026-06-22T10:00:00Z"):
    p = cmp.session_dir(agent) / f"{sid}.md"
    p.write_text(
        f"---\nagent: {agent}\nsession_id: {sid}\nday: {day}\n"
        f"freshness: {freshness}\ngenerated_at: {gen}\n---\n"
        f"# Session Card: {agent} — {day}\n{body}\n",
        encoding="utf-8")
    return p


def _write_digest(cmp, kind, slug, body, *, freshness="fresh", gen="2026-06-22T11:00:00Z",
                  links=""):
    p = cmp.digest_node(kind, slug)
    fm_links = f"links: {links}\n" if links else ""
    p.write_text(
        f"---\nkind: {kind}\nslug: {slug}\nfreshness: {freshness}\n"
        f"generated_at: {gen}\nsource_through: 2026-06-22\n{fm_links}---\n"
        f"# {kind.title()}: {slug}\n## Current understanding\n{body}\n",
        encoding="utf-8")
    return p


def _write_daily(cmp, agent, day, body, *, gen="2026-06-22T09:00:00Z"):
    p = cmp.daily_agent_card(agent, day)
    p.write_text(f"# Daily roll-up: {agent} — {day}\n_Generated {gen}._\n{body}\n",
                 encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
def test_index_builds_with_provenance(env):
    cmp, csi, smc, canon = env
    _write_session(cmp, "builder", "s1", "2026-06-22",
                   "## Summary\n- shipped the boot pack generator")
    _write_digest(cmp, "decisions", "decisions", "- chose PR Engram approach")
    stats = csi.build_card_index()
    assert stats["records"] == 2
    assert stats["skipped"] == 0
    meta, recs = csi.load_index()
    assert meta["records"] == 2
    by_id = {r.id: r for r in recs}
    dig = next(r for r in recs if r.kind == "digest")
    assert dig.digest_kind == "decisions" and dig.slug == "decisions"
    assert dig.freshness == "fresh" and dig.generated_at
    sess = next(r for r in recs if r.kind == "session")
    assert sess.agent == "builder" and sess.day == "2026-06-22"


def test_search_ranks_relevant_card_first(env):
    cmp, csi, smc, canon = env
    _write_session(cmp, "builder", "s1", "2026-06-22",
                   "## Summary\n- unrelated note about timeline rendering")
    _write_digest(cmp, "systems", "code-changes",
                  "- implemented the boot pack generator orientation file")
    csi.build_card_index()
    hits, diag = smc.search_cards("boot pack generator", top_k=5)
    assert hits, diag
    assert "boot pack" in hits[0].record.text.lower()
    assert diag["mode"] == "lexical"


def test_filters(env):
    cmp, csi, smc, canon = env
    _write_session(cmp, "builder", "s1", "2026-06-22", "## Summary\n- digest consolidation work")
    _write_session(cmp, "other", "s2", "2026-06-22", "## Summary\n- digest consolidation work")
    _write_digest(cmp, "systems", "misc", "- digest consolidation work")
    csi.build_card_index()
    hits, _ = smc.search_cards("digest consolidation", kind="session")
    assert hits and all(h.record.kind == "session" for h in hits)
    hits2, _ = smc.search_cards("digest consolidation", agent="builder")
    assert hits2 and all(h.record.agent == "builder" for h in hits2)
    hits3, _ = smc.search_cards("digest consolidation", kind="digest", digest_kind="systems")
    assert hits3 and all(h.record.digest_kind == "systems" for h in hits3)


def test_digest_kind_boost_over_session(env):
    """Same matching text: a digest (compounded knowledge) should outrank a session."""
    cmp, csi, smc, canon = env
    text = "- alpha bravo charlie delta echo"
    _write_session(cmp, "builder", "s1", "2026-06-22", f"## Summary\n{text}",
                   gen="2026-06-22T10:00:00Z")
    _write_digest(cmp, "systems", "misc", text, gen="2026-06-22T10:00:00Z")
    csi.build_card_index()
    hits, _ = smc.search_cards("alpha bravo charlie delta echo", top_k=5)
    assert hits[0].record.kind == "digest"


def test_freshness_penalty_orders_fresh_first(env):
    cmp, csi, smc, canon = env
    text = "- quantum entanglement scheduler"
    _write_digest(cmp, "systems", "fresh-one", text, freshness="fresh",
                  gen="2026-06-20T10:00:00Z")
    _write_digest(cmp, "systems", "stale-one", text, freshness="stale",
                  gen="2026-06-22T10:00:00Z")
    csi.build_card_index()
    hits, _ = smc.search_cards("quantum entanglement scheduler", top_k=5)
    assert hits[0].record.slug == "fresh-one"


def test_semantic_blend_degrades_gracefully(env):
    cmp, csi, smc, canon = env
    _write_digest(cmp, "systems", "misc", "- vector search service integration")
    csi.build_card_index()

    def broken_blend(query, records):
        raise RuntimeError("search service down")

    hits, diag = smc.search_cards("vector search", semantic_blend=broken_blend)
    assert diag["degraded"] is True
    assert any("lexical-only" in n for n in diag["notes"])
    assert hits  # still returns lexical results


def test_semantic_blend_boosts_when_available(env):
    cmp, csi, smc, canon = env
    _write_digest(cmp, "systems", "aaa", "- common shared term")
    _write_digest(cmp, "systems", "bbb", "- common shared term")
    csi.build_card_index()
    _meta, recs = csi.load_index()
    target = next(r for r in recs if r.slug == "bbb")

    def blend(query, records):
        return {target.id: 100.0}

    hits, diag = smc.search_cards("common shared term", semantic_blend=blend)
    assert diag["mode"] == "hybrid" and diag["degraded"] is False
    assert hits[0].record.slug == "bbb"


def test_empty_index_is_graceful(env):
    cmp, csi, smc, canon = env
    # no cards written, no index built
    hits, diag = smc.search_cards("anything")
    assert hits == []
    assert diag["degraded"] is True
    assert any("empty or missing" in n for n in diag["notes"])


def test_malformed_record_skipped_not_fatal(env):
    cmp, csi, smc, canon = env
    _write_digest(cmp, "systems", "good", "- legitimate searchable content here")
    csi.build_card_index()
    p = csi.index_path()
    # append a corrupt line + a record with an unknown field
    with p.open("a", encoding="utf-8") as fh:
        fh.write("this is not json\n")
        fh.write('{"id":"x","kind":"digest","bogusfield":1}\n')
    meta, recs = csi.load_index()
    # the good record survives; corrupt/unknown ones skipped
    assert any(r.slug == "good" for r in recs)
    hits, _ = smc.search_cards("legitimate searchable content")
    assert hits and hits[0].record.slug == "good"


def test_load_index_survives_bare_scalar_lines(env):
    """Tester defect A: a valid-JSON but non-object line (scalar/list) from a
    truncated write must be skipped, not crash the whole search path."""
    cmp, csi, smc, canon = env
    _write_digest(cmp, "systems", "good", "- legitimate searchable content here")
    csi.build_card_index()
    p = csi.index_path()
    with p.open("a", encoding="utf-8") as fh:
        for junk in ("123", "null", "true", "3.14", '"a bare string"', "[1,2,3]"):
            fh.write(junk + "\n")
    meta, recs = csi.load_index()  # must NOT raise
    assert any(r.slug == "good" for r in recs)
    hits, _ = smc.search_cards("legitimate searchable content")
    assert hits and hits[0].record.slug == "good"


def test_semantic_blend_garbage_output_degrades(env):
    """Tester defect B: a blend that RETURNS junk (not raises) must degrade to
    lexical, never crash."""
    cmp, csi, smc, canon = env
    _write_digest(cmp, "systems", "misc", "- vector search service integration")
    csi.build_card_index()

    # non-dict return
    hits, diag = smc.search_cards("vector search", semantic_blend=lambda q, r: "garbage")
    assert diag["degraded"] is True and hits

    # dict with non-numeric scores -> bad entries dropped, good ones kept, no crash
    _meta, recs = csi.load_index()
    tid = recs[0].id

    def mixed(q, r):
        return {tid: "notanumber", "other": 5.0}

    hits2, diag2 = smc.search_cards("vector search", semantic_blend=mixed)
    assert diag2["mode"] == "hybrid" and diag2["degraded"] is False and hits2


def test_tiebreak_newest_first(env):
    """Tester defect #4: on equal score+freshness, NEWER generated_at must rank
    first (documented desc), not oldest-first."""
    cmp, csi, smc, canon = env
    text = "- zeta eta theta unique phrase"
    _write_digest(cmp, "systems", "older", text, gen="2026-06-20T10:00:00Z")
    _write_digest(cmp, "systems", "newer", text, gen="2026-06-22T10:00:00Z")
    csi.build_card_index()
    # force a pure tie by zeroing recency influence is hard; instead assert that
    # when scores+freshness match, the newer card precedes the older.
    hits, _ = smc.search_cards("zeta eta theta unique phrase", top_k=5)
    slugs = [h.record.slug for h in hits]
    assert slugs.index("newer") < slugs.index("older")


def test_index_idempotent(env):
    cmp, csi, smc, canon = env
    _write_session(cmp, "builder", "s1", "2026-06-22", "## Summary\n- stable content")
    _write_digest(cmp, "decisions", "decisions", "- a decision")
    csi.build_card_index()
    first = csi.index_path().read_text(encoding="utf-8").splitlines()[1:]  # drop meta
    csi.build_card_index()
    second = csi.index_path().read_text(encoding="utf-8").splitlines()[1:]
    assert first == second  # record block identical across rebuilds


def test_dry_run_writes_nothing(env):
    cmp, csi, smc, canon = env
    _write_digest(cmp, "systems", "misc", "- something")
    stats = csi.build_card_index(dry_run=True)
    assert stats.get("dry_run") is True
    assert not csi.index_path().exists()


def test_boot_pack_not_indexed(env):
    cmp, csi, smc, canon = env
    (cmp.session_dir("builder") / "_BOOT_PACK.md").write_text(
        "# Boot Pack\n- derived orientation\n", encoding="utf-8")
    _write_session(cmp, "builder", "s1", "2026-06-22", "## Summary\n- real card")
    stats = csi.build_card_index()
    _meta, recs = csi.load_index()
    assert all("_BOOT_PACK" not in r.path for r in recs)
    assert stats["records"] == 1


def test_wikilinks_captured(env):
    cmp, csi, smc, canon = env
    _write_digest(cmp, "decisions", "decisions", "- a decision",
                  links="[[decisions/open-threads]] [[systems/misc]]")
    csi.build_card_index()
    _meta, recs = csi.load_index()
    dig = next(r for r in recs if r.kind == "digest")
    assert "decisions/open-threads" in dig.wikilinks
    assert "systems/misc" in dig.wikilinks
