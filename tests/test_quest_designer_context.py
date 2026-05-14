from __future__ import annotations

from orchestrator_role_targets.context_builder import build_quest_designer_context


def test_quest_designer_context_shape() -> None:
    packet = {
        "work_id": "qd-1",
        "title": "Quest objective flow",
        "kind": "feature",
        "risk_level": "medium",
        "memory_mode_effective": "game_dev",
        "tags": ["quest", "mission", "objective", "reward"],
        "summaries": {
            "work_summary": "work",
            "project_summary": "project",
            "subproject_summary": "subproject",
        },
        "instructions": {
            "focus": "quest design focus",
            "avoid": "avoid drift",
        },
        "memory_refs": ["mem://project/overview"],
        "changed_files": [],
    }
    ctx = build_quest_designer_context(packet)
    assert ctx["work_id"] == "qd-1"
    assert ctx["title"] == "Quest objective flow"
    assert ctx["focus"] == "quest design focus"
