from __future__ import annotations

from orchestrator_role_targets.writeback_service import persist_writeback


def test_writeback_persists_system_and_technical_design(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

    packet = {
        "work_id": "wd-1",
        "project_id": "test_project",
        "role": "writeback",
        "title": "Boss framework",
        "kind": "feature",
        "risk_level": "medium",
        "memory_mode_effective": "game_dev",
        "summaries": {
            "work_summary": "work summary",
            "project_summary": "project summary",
            "subproject_summary": "subproject summary",
        },
        "instructions": {
            "focus": "persist summary",
            "avoid": "do not persist junk",
        },
        "system_design": {
            "system_goal": "Combat framework boundaries",
            "scope": "bounded system slice",
        },
        "technical_design": {
            "design_goal": "Boss mechanic flow",
            "mechanic_scope": "bounded combat slice",
        },
    }

    persisted = persist_writeback(packet)
    payload = persisted["payload"]

    assert payload["system_design"]["system_goal"] == "Combat framework boundaries"
    assert payload["technical_design"]["design_goal"] == "Boss mechanic flow"
