from __future__ import annotations

from orchestrator_role_targets.context_builder import build_economy_designer_context


def test_economy_designer_context_shape() -> None:
    packet = {
        "work_id": "ed-1",
        "title": "Reward and shop economy",
        "kind": "feature",
        "risk_level": "medium",
        "memory_mode_effective": "game_dev",
        "tags": ["economy", "shop", "reward", "currency"],
        "summaries": {
            "work_summary": "work",
            "project_summary": "project",
            "subproject_summary": "subproject",
        },
        "instructions": {
            "focus": "economy design focus",
            "avoid": "avoid drift",
        },
        "memory_refs": ["mem://project/overview"],
        "changed_files": [],
    }
    ctx = build_economy_designer_context(packet)
    assert ctx["work_id"] == "ed-1"
    assert ctx["title"] == "Reward and shop economy"
    assert ctx["focus"] == "economy design focus"
