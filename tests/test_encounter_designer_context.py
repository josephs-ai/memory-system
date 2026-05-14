from __future__ import annotations

from orchestrator_role_targets.context_builder import build_encounter_designer_context


def test_encounter_designer_context_shape() -> None:
    packet = {
        "work_id": "ed-1",
        "title": "Boss arena encounter",
        "kind": "feature",
        "risk_level": "medium",
        "memory_mode_effective": "game_dev",
        "tags": ["encounter", "boss", "arena"],
        "summaries": {
            "work_summary": "work",
            "project_summary": "project",
            "subproject_summary": "subproject",
        },
        "instructions": {
            "focus": "encounter design focus",
            "avoid": "avoid drift",
        },
        "memory_refs": ["mem://project/overview"],
        "changed_files": [],
    }
    ctx = build_encounter_designer_context(packet)
    assert ctx["work_id"] == "ed-1"
    assert ctx["title"] == "Boss arena encounter"
    assert ctx["focus"] == "encounter design focus"
