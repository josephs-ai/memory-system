from __future__ import annotations

from orchestrator_role_targets.prompt_compiler import build_encounter_designer_prompt


def test_encounter_designer_prompt_contains_core_sections() -> None:
    ctx = {
        "work_id": "ed-1",
        "title": "Boss arena encounter",
        "kind": "feature",
        "risk_level": "medium",
        "work_summary_short": "none",
        "project_summary_short": "project",
        "subproject_summary_short": "subproject",
        "focus": "Produce a bounded encounter design.",
        "avoid": "Do not redesign the whole game.",
        "changed_files_text": "(none)",
        "memory_refs": [],
        "acceptance_criteria": ["Encounter design is bounded and actionable."],
    }
    prompt = build_encounter_designer_prompt(ctx)
    assert "ROLE" in prompt
    assert "Encounter Designer" in prompt
    assert "ENCOUNTER DESIGN ARTIFACT FIELDS" in prompt
    assert "encounter_structure" in prompt
    assert "enemy_roles" in prompt
