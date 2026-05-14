from __future__ import annotations

from orchestrator_role_targets.prompt_compiler import build_enemy_ai_designer_prompt


def test_enemy_ai_designer_prompt_contains_core_sections() -> None:
    ctx = {
        "work_id": "aid-1",
        "title": "Enemy targeting and coordination",
        "kind": "feature",
        "risk_level": "medium",
        "work_summary_short": "none",
        "project_summary_short": "project",
        "subproject_summary_short": "subproject",
        "focus": "Produce a bounded enemy AI design.",
        "avoid": "Do not redesign the whole game.",
        "changed_files_text": "(none)",
        "memory_refs": [],
        "acceptance_criteria": ["Enemy AI design is bounded and actionable."],
    }
    prompt = build_enemy_ai_designer_prompt(ctx)
    assert "ROLE" in prompt
    assert "Enemy AI Designer" in prompt
    assert "ENEMY AI DESIGN ARTIFACT FIELDS" in prompt
    assert "state_logic_rules" in prompt
    assert "target_selection_rules" in prompt
