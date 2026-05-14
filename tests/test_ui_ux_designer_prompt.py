from __future__ import annotations

from orchestrator_role_targets.prompt_compiler import build_ui_ux_designer_prompt


def test_ui_ux_designer_prompt_contains_core_sections() -> None:
    ctx = {
        "work_id": "ux-1",
        "title": "HUD and menu flow",
        "kind": "feature",
        "risk_level": "medium",
        "work_summary_short": "none",
        "project_summary_short": "project",
        "subproject_summary_short": "subproject",
        "focus": "Produce a bounded UI/UX design.",
        "avoid": "Do not redesign the whole game.",
        "changed_files_text": "(none)",
        "memory_refs": [],
        "acceptance_criteria": ["UI/UX design is bounded and actionable."],
    }
    prompt = build_ui_ux_designer_prompt(ctx)
    assert "ROLE" in prompt
    assert "UI UX Designer" in prompt
    assert "UI UX DESIGN ARTIFACT FIELDS" in prompt
    assert "screen_flow" in prompt
    assert "hud_rules" in prompt
