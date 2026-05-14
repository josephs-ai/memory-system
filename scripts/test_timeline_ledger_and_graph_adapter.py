from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import Mock, patch

from graph_search_adapter import search_graph_text
from timeline_event_ledger import append_timeline_events, load_session_events, visible_event_records



def test_timeline_ledger_appends_visible_events_idempotently(tmp_path: Path):
    session_path = tmp_path / "session.jsonl"
    session_path.write_text(
        "\n".join(
            [
                json.dumps({
                    "type": "message",
                    "id": "u1",
                    "timestamp": "2026-04-10T13:00:00Z",
                    "message": {"role": "user", "content": [{"type": "text", "text": "Pipeline Alpha must stay fully automated."}]},
                }),
                json.dumps({
                    "type": "message",
                    "id": "a1",
                    "timestamp": "2026-04-10T13:00:05Z",
                    "message": {"role": "assistant", "content": [{"type": "text", "text": "Understood."}]},
                }),
                json.dumps({
                    "type": "message",
                    "id": "t1",
                    "timestamp": "2026-04-10T13:00:06Z",
                    "message": {"role": "toolResult", "content": [{"type": "text", "text": "ignored"}]},
                }),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    events = load_session_events(session_path)
    records = visible_event_records(events, agent_name="main", session_path=session_path)
    ledger_path = tmp_path / "events.jsonl"

    first = append_timeline_events(records, ledger_path)
    second = append_timeline_events(records, ledger_path)

    assert len(records) == 2
    assert first == {"appended": 2, "skipped": 0}
    assert second == {"appended": 0, "skipped": 2}
    persisted = [json.loads(line) for line in ledger_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(persisted) == 2
    assert {row["role"] for row in persisted} == {"user", "assistant"}



def test_graph_search_adapter_passes_scope_filters_and_parses_json():
    completed = Mock(returncode=0, stdout='{"text":"Scoped fact","scope_id":"scope-1"}\n', stderr='')
    with patch("graph_search_adapter.subprocess.run", return_value=completed) as run:
        rows = search_graph_text(
            "automation",
            limit=5,
            project_id="proj-a",
            subproject_id="enemy_ai",
            workflow_id="wf-1",
            pipeline_id="alpha",
        )

    assert rows == [{"text": "Scoped fact", "scope_id": "scope-1"}]
    cmd = run.call_args.args[0]
    assert "--query" in cmd
    assert "automation" in cmd
    assert "--project-id" in cmd and "proj-a" in cmd
    assert "--subproject-id" in cmd and "enemy_ai" in cmd
    assert "--workflow-id" in cmd and "wf-1" in cmd
    assert "--pipeline-id" in cmd and "alpha" in cmd
