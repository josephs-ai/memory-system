from __future__ import annotations

from orchestrator_role_targets.context_builder import build_narrative_designer_context


def test_narrative_designer_context_shape() -> None:
    packet = {
        "work_id": "nd-1",
        "title": "Quest dialogue flow",
        "kind": "feature",
        "risk_level": "medium",
        "memory_mode_effective": "game_dev",
        "tags": ["narrative", "story", "dialogue", "quest"],
        "summaries": {
            "work_summary": "work",
            "project_summary": "project",
            "subproject_summary": "subproject",
        },
        "instructions": {
            "focus": "narrative design focus",
            "avoid": "avoid drift",
        },
        "memory_refs": ["mem://project/overview"],
        "changed_files": [],
    }
    ctx = build_narrative_designer_context(packet)
    assert ctx["work_id"] == "nd-1"
    assert ctx["title"] == "Quest dialogue flow"
    assert ctx["focus"] == "narrative design focus"
