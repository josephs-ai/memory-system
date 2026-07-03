"""
Regression tests for reset/rotation transcript capture.

These protect against the data-loss class where checkpoint_all_agents only
processed the latest active *.jsonl and skipped *.jsonl.reset.* files.
"""
import json

import checkpoint_agent as ca
import checkpoint_all_agents as all_agents


def write_jsonl(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text + "\n", encoding="utf-8")


def test_discover_session_files_includes_reset_transcripts(tmp_path, monkeypatch):
    agents_dir = tmp_path / "agents"
    sessions = agents_dir / "main" / "sessions"
    active = sessions / "active.jsonl"
    reset = sessions / "old.jsonl.reset.2026-06-21T00-00-00.000Z"
    lock = sessions / "active.jsonl.lock"

    write_jsonl(active, '{"type":"message","message":{"role":"user","content":[]}}')
    write_jsonl(reset, '{"type":"message","message":{"role":"user","content":[]}}')
    lock.write_text("ignore me", encoding="utf-8")

    (sessions / "sessions.json").write_text(
        json.dumps({"s1": {"sessionFile": str(active), "updatedAt": 2}}),
        encoding="utf-8",
    )

    monkeypatch.setattr(all_agents, "AGENTS_DIR", agents_dir)

    found = {p.name for p in all_agents.discover_session_files_for_agent("main")}
    assert "active.jsonl" in found
    assert "old.jsonl.reset.2026-06-21T00-00-00.000Z" in found
    assert "active.jsonl.lock" not in found


def test_list_agents_with_sessions_returns_all_transcripts(tmp_path, monkeypatch):
    agents_dir = tmp_path / "agents"
    write_jsonl(agents_dir / "main" / "sessions" / "a.jsonl", "{}")
    write_jsonl(agents_dir / "main" / "sessions" / "b.jsonl.reset.1", "{}")
    write_jsonl(agents_dir / "memory-coder" / "sessions" / "c.jsonl.reset.1", "{}")

    monkeypatch.setattr(all_agents, "AGENTS_DIR", agents_dir)

    rows = all_agents.list_agents_with_sessions()
    assert ("main", agents_dir / "main" / "sessions" / "a.jsonl") in rows
    assert ("main", agents_dir / "main" / "sessions" / "b.jsonl.reset.1") in rows
    assert ("memory-coder", agents_dir / "memory-coder" / "sessions" / "c.jsonl.reset.1") in rows


def test_checkpoint_all_state_key_is_per_transcript(tmp_path):
    a = tmp_path / "same-agent" / "one.jsonl"
    b = tmp_path / "same-agent" / "two.jsonl.reset.1"
    write_jsonl(a, "{}")
    write_jsonl(b, "{}")

    assert all_agents.state_key("main", a) != all_agents.state_key("main", b)


def test_checkpoint_agent_cursor_path_is_per_transcript(tmp_path, monkeypatch):
    monkeypatch.setattr(ca, "STATE_DIR", tmp_path / "state")
    active = tmp_path / "active.jsonl"
    reset = tmp_path / "active.jsonl.reset.1"
    write_jsonl(active, "{}")
    write_jsonl(reset, "{}")

    ca.save_cursor("main", ca.build_cursor(active), active)
    ca.save_cursor("main", ca.build_cursor(reset), reset)

    active_cursor = ca.cursor_path("main", active)
    reset_cursor = ca.cursor_path("main", reset)

    assert active_cursor != reset_cursor
    assert active_cursor.exists()
    assert reset_cursor.exists()
    assert ca.load_cursor("main", active)["path"] == str(active)
    assert ca.load_cursor("main", reset)["path"] == str(reset)


def test_legacy_cursor_path_still_supported(tmp_path, monkeypatch):
    monkeypatch.setattr(ca, "STATE_DIR", tmp_path / "state")
    transcript = tmp_path / "session.jsonl"
    write_jsonl(transcript, "{}")
    cursor = ca.build_cursor(transcript)

    ca.save_cursor("main", cursor)

    assert ca.load_cursor("main")["committed_offset"] == cursor["committed_offset"]
    assert ca.cursor_path("main") == tmp_path / "state" / "main.cursor.json"
