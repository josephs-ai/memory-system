from __future__ import annotations

from orchestrator_role_targets.prompt_compiler import build_gameplay_designer_prompt


def test_gameplay_designer_prompt_contains_core_sections() -> None:
    ctx = {
        "work_id": "gd-1",
        "title": "Core gameplay loop",
        "kind": "feature",
        "risk_level": "medium",
        "work_summary_short": "none",
        "project_summary_short": "project",
        "subproject_summary_short": "subproject",
        "focus": "Produce a bounded gameplay design.",
        "avoid": "Do not redesign the whole game.",
        "changed_files_text": "(none)",
        "memory_refs": [],
        "acceptance_criteria": ["Gameplay design is bounded and actionable."],
    }
    prompt = build_gameplay_designer_prompt(ctx)
    assert "ROLE" in prompt
    assert "Gameplay Designer" in prompt
    assert "GAMEPLAY DESIGN ARTIFACT FIELDS" in prompt
    assert "core_loop" in prompt
    assert "player_verbs" in prompt
