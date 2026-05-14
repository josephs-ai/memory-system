from __future__ import annotations

from orchestrator_role_targets.prompt_compiler import build_combat_designer_prompt


def test_combat_designer_prompt_contains_core_sections() -> None:
    ctx = {
        "work_id": "cd-1",
        "title": "Boss combat loop",
        "kind": "feature",
        "risk_level": "medium",
        "work_summary_short": "none",
        "project_summary_short": "project",
        "subproject_summary_short": "subproject",
        "focus": "Produce bounded combat design.",
        "avoid": "Do not redesign the whole game.",
        "changed_files_text": "(none)",
        "memory_refs": [],
        "acceptance_criteria": ["Combat spec is bounded and actionable."],
    }
    prompt = build_combat_designer_prompt(ctx)
    assert "ROLE" in prompt
    assert "Combat Designer" in prompt
    assert "COMBAT DESIGN ARTIFACT FIELDS" in prompt
    assert "attack_loop" in prompt
    assert "telegraph_rules" in prompt
