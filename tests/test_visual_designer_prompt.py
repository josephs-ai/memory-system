from __future__ import annotations

from orchestrator_role_targets.prompt_compiler import build_visual_designer_prompt


def test_visual_designer_prompt_contains_core_sections() -> None:
    ctx = {
        "work_id": "vd-1",
        "title": "Combat visual clarity",
        "kind": "feature",
        "risk_level": "medium",
        "work_summary_short": "none",
        "project_summary_short": "project",
        "subproject_summary_short": "subproject",
        "focus": "Produce a bounded visual design.",
        "avoid": "Do not redesign the whole game.",
        "changed_files_text": "(none)",
        "memory_refs": [],
        "acceptance_criteria": ["Visual design is bounded and actionable."],
    }
    prompt = build_visual_designer_prompt(ctx)
    assert "ROLE" in prompt
    assert "Visual Designer" in prompt
    assert "VISUAL DESIGN ARTIFACT FIELDS" in prompt
    assert "visual_language" in prompt
    assert "contrast_rules" in prompt
