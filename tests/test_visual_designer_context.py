from __future__ import annotations

from orchestrator_role_targets.context_builder import build_visual_designer_context


def test_visual_designer_context_shape() -> None:
    packet = {
        "work_id": "vd-1",
        "title": "Combat visual clarity",
        "kind": "feature",
        "risk_level": "medium",
        "memory_mode_effective": "game_dev",
        "tags": ["visual", "readability", "vfx"],
        "summaries": {
            "work_summary": "work",
            "project_summary": "project",
            "subproject_summary": "subproject",
        },
        "instructions": {
            "focus": "visual design focus",
            "avoid": "avoid drift",
        },
        "memory_refs": ["mem://project/overview"],
        "changed_files": [],
    }
    ctx = build_visual_designer_context(packet)
    assert ctx["work_id"] == "vd-1"
    assert ctx["title"] == "Combat visual clarity"
    assert ctx["focus"] == "visual design focus"
