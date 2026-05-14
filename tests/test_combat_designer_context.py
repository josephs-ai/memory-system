from __future__ import annotations

from orchestrator_role_targets.context_builder import build_combat_designer_context


def test_combat_designer_context_shape() -> None:
    packet = {
        "work_id": "cd-1",
        "title": "Boss combat loop",
        "kind": "feature",
        "risk_level": "medium",
        "memory_mode_effective": "game_dev",
        "tags": ["combat", "boss"],
        "summaries": {
            "work_summary": "work",
            "project_summary": "project",
            "subproject_summary": "subproject",
        },
        "instructions": {
            "focus": "combat design focus",
            "avoid": "avoid drift",
        },
        "memory_refs": ["mem://project/overview"],
        "changed_files": [],
    }
    ctx = build_combat_designer_context(packet)
    assert ctx["work_id"] == "cd-1"
    assert ctx["title"] == "Boss combat loop"
    assert ctx["focus"] == "combat design focus"
