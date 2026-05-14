from __future__ import annotations

from orchestrator_role_targets.context_builder import build_enemy_ai_designer_context


def test_enemy_ai_designer_context_shape() -> None:
    packet = {
        "work_id": "aid-1",
        "title": "Enemy targeting and coordination",
        "kind": "feature",
        "risk_level": "medium",
        "memory_mode_effective": "game_dev",
        "tags": ["enemy_ai", "ai", "targeting", "coordination"],
        "summaries": {
            "work_summary": "work",
            "project_summary": "project",
            "subproject_summary": "subproject",
        },
        "instructions": {
            "focus": "enemy ai design focus",
            "avoid": "avoid drift",
        },
        "memory_refs": ["mem://project/overview"],
        "changed_files": [],
    }
    ctx = build_enemy_ai_designer_context(packet)
    assert ctx["work_id"] == "aid-1"
    assert ctx["title"] == "Enemy targeting and coordination"
    assert ctx["focus"] == "enemy ai design focus"
