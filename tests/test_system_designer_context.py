from __future__ import annotations

from orchestrator_role_targets.context_builder import build_system_designer_context


def test_system_designer_context_shape() -> None:
    packet = {
        "work_id": "sd-1",
        "title": "Combat framework",
        "kind": "feature",
        "risk_level": "medium",
        "memory_mode_effective": "game_dev",
        "tags": ["cross_system"],
        "summaries": {
            "work_summary": "work",
            "project_summary": "project",
            "subproject_summary": "subproject",
        },
        "instructions": {
            "focus": "system design focus",
            "avoid": "avoid drift",
        },
        "memory_refs": ["mem://project/overview"],
        "changed_files": [],
    }
    ctx = build_system_designer_context(packet)
    assert ctx["work_id"] == "sd-1"
    assert ctx["title"] == "Combat framework"
    assert ctx["focus"] == "system design focus"
