"""
Hermetic tests for generate_daily_session_card.py (Phase 2 auto_memory).

Synthesizes tiny fake transcripts (matching the real event shape: message events
with role user/assistant/toolResult and assistant content blocks of type
text/toolCall) and points the canonical root at a tmp dir. No real workspace
state touched. Includes the PR/Engram regression: the exact context that was lost
when an agent couldn't reconstruct why "PR" mattered.
"""
from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest


@pytest.fixture()
def env(tmp_path, monkeypatch):
    root = tmp_path / "canonical"
    monkeypatch.setenv("OPENCLAW_CANONICAL_MEMORY_ROOT", str(root))
    import canonical_memory_paths as cmp

    importlib.reload(cmp)
    import generate_daily_session_card as g

    importlib.reload(g)
    g.cmp.scaffold()
    return g, cmp, root, tmp_path


def _msg(role, blocks, ts="2026-06-22T10:00:00.000Z"):
    return {"type": "message", "timestamp": ts, "message": {"role": role, "content": blocks}}


def _write_transcript(path: Path, events):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for e in events:
            f.write(json.dumps(e) + "\n")


def _basic_events():
    return [
        _msg("user", [{"type": "text", "text": "feature branch + PR (pull request): do that"}],
             "2026-06-22T09:00:00.000Z"),
        _msg("assistant", [
            {"type": "text", "text": "I'll create the branch and push to Engram."},
            {"type": "toolCall", "name": "edit",
             "arguments": {"file_path": "/x/checkpoint_agent.py"}},
        ], "2026-06-22T09:01:00.000Z"),
        _msg("assistant", [
            {"type": "toolCall", "name": "write",
             "arguments": {"file_path": "/x/backfill.py"}},
            {"type": "toolCall", "name": "exec",
             "arguments": {"command": "python3 -m pytest -q test_thing.py"}},
        ], "2026-06-22T09:02:00.000Z"),
        _msg("toolResult", [{"type": "text", "text": "11 passed in 0.2s"}],
             "2026-06-22T09:02:30.000Z"),
        _msg("assistant", [
            {"type": "text", "text": "Fixed the bug. The push failed: no API token. "
                                     "Want me to draft a PR description?"}],
             "2026-06-22T09:03:00.000Z"),
    ]


def test_build_card_extracts_all_sections(env):
    g, cmp, root, tmp = env
    tp = tmp / "agents" / "builder" / "sessions" / "abc.jsonl"
    _write_transcript(tp, _basic_events())

    m = g.build_card("builder", tp)
    assert m.freshness == g.FRESH
    assert m.day == "2026-06-22"
    assert "/x/checkpoint_agent.py" in m.files_changed
    assert "/x/backfill.py" in m.files_changed
    assert any("pytest" in t for t in m.tests_run)
    assert m.blockers, "should detect 'failed: no API token'"
    assert m.next_step and "draft a pr description" in m.next_step.lower()
    # decision captured the PR directive
    assert any("pr (pull request)" in d.lower() for d in m.decisions)


def test_pr_engram_context_preserved(env):
    """The exact regression: a fresh agent reading this card must be able to connect
    'PR' to the pull-request/Engram flow."""
    g, cmp, root, tmp = env
    tp = tmp / "agents" / "builder" / "sessions" / "pr.jsonl"
    _write_transcript(tp, _basic_events())
    m = g.build_card("builder", tp)
    rendered = g.render_card(m).lower()
    assert "pull request" in rendered
    assert "engram" in rendered
    assert "branch" in rendered


def test_card_written_and_idempotent(env):
    g, cmp, root, tmp = env
    tp = tmp / "agents" / "builder" / "sessions" / "abc.jsonl"
    _write_transcript(tp, _basic_events())
    p1 = g.write_card(g.build_card("builder", tp))
    first = p1.read_text()
    p2 = g.write_card(g.build_card("builder", tp))
    assert p1 == p2
    # Re-render differs only by generated_at timestamp line, not by duplication
    assert p2.read_text().count("# Session Card:") == 1
    assert first.count("## Files changed") == 1


def test_missing_transcript_is_degraded(env):
    g, cmp, root, tmp = env
    m = g.build_card("builder", tmp / "nope.jsonl")
    assert m.freshness == g.DEGRADED
    assert "not found" in (m.note or "")
    rendered = g.render_card(m)
    assert "degraded" in rendered.lower()


def test_empty_transcript_is_degraded(env):
    g, cmp, root, tmp = env
    tp = tmp / "agents" / "x" / "sessions" / "empty.jsonl"
    tp.parent.mkdir(parents=True, exist_ok=True)
    tp.write_text("", encoding="utf-8")
    m = g.build_card("x", tp)
    assert m.freshness == g.DEGRADED


def test_user_directive_dedup(env):
    g, cmp, root, tmp = env
    events = [
        _msg("user", [{"type": "text", "text": "do it?"}], "2026-06-22T09:00:00.000Z"),
        _msg("user", [{"type": "text", "text": "do it?"}], "2026-06-22T09:00:01.000Z"),
    ]
    tp = tmp / "agents" / "b" / "sessions" / "d.jsonl"
    _write_transcript(tp, events)
    m = g.build_card("b", tp)
    user_decisions = [d for d in m.decisions if d.startswith("(user)")]
    assert len(user_decisions) == 1


def test_tests_run_ignores_non_runner_commands(env):
    g, cmp, root, tmp = env
    events = [
        _msg("assistant", [
            {"type": "toolCall", "name": "exec",
             "arguments": {"command": "git log main..fix/test_branch --oneline"}},
            {"type": "toolCall", "name": "exec",
             "arguments": {"command": "pytest -q test_real.py"}},
        ]),
    ]
    tp = tmp / "agents" / "b" / "sessions" / "t.jsonl"
    _write_transcript(tp, events)
    m = g.build_card("b", tp)
    assert len(m.tests_run) == 1
    assert "pytest" in m.tests_run[0]


def test_daily_rollup_and_index(env):
    g, cmp, root, tmp = env
    # two agents, same day
    for agent in ("builder", "memory-coder"):
        tp = tmp / "agents" / agent / "sessions" / f"{agent}.jsonl"
        _write_transcript(tp, _basic_events())
        g.generate_one(agent, tp)
    idx = cmp.daily_index("2026-06-22", ensure=False)
    assert idx.exists()
    text = idx.read_text()
    assert "builder" in text and "memory-coder" in text
    # per-agent rollups exist
    assert cmp.daily_agent_card("builder", "2026-06-22", ensure=False).exists()
    assert cmp.daily_agent_card("memory-coder", "2026-06-22", ensure=False).exists()


def test_frontmatter_well_formed(env):
    g, cmp, root, tmp = env
    tp = tmp / "agents" / "builder" / "sessions" / "abc.jsonl"
    _write_transcript(tp, _basic_events())
    rendered = g.render_card(g.build_card("builder", tp))
    lines = rendered.splitlines()
    assert lines[0] == "---"
    fm_end = lines.index("---", 1)
    fm = "\n".join(lines[1:fm_end])
    for key in ("agent:", "session_id:", "generated_at:", "freshness:", "day:"):
        assert key in fm
