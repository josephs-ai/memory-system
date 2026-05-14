from __future__ import annotations

from orchestrator_role_targets.prompt_compiler import build_level_designer_prompt


def test_level_designer_prompt_contains_core_sections() -> None:
    ctx = {
        "work_id": "ld-1",
        "title": "Boss level layout",
        "kind": "feature",
        "risk_level": "medium",
        "work_summary_short": "none",
        "project_summary_short": "project",
        "subproject_summary_short": "subproject",
        "focus": "Produce a bounded level design.",
        "avoid": "Do not redesign the whole game.",
        "changed_files_text": "(none)",
        "memory_refs": [],
        "acceptance_criteria": ["Level design is bounded and actionable."],
    }
    prompt = build_level_designer_prompt(ctx)
    assert "ROLE" in prompt
    assert "Level Designer" in prompt
    assert "LEVEL DESIGN ARTIFACT FIELDS" in prompt
    assert "layout_structure" in prompt
    assert "room_flow" in prompt
