from __future__ import annotations

from orchestrator_role_targets.prompt_compiler import build_narrative_designer_prompt


def test_narrative_designer_prompt_contains_core_sections() -> None:
    ctx = {
        "work_id": "nd-1",
        "title": "Quest dialogue flow",
        "kind": "feature",
        "risk_level": "medium",
        "work_summary_short": "none",
        "project_summary_short": "project",
        "subproject_summary_short": "subproject",
        "focus": "Produce a bounded narrative design.",
        "avoid": "Do not redesign the whole game.",
        "changed_files_text": "(none)",
        "memory_refs": [],
        "acceptance_criteria": ["Narrative design is bounded and actionable."],
    }
    prompt = build_narrative_designer_prompt(ctx)
    assert "ROLE" in prompt
    assert "Narrative Designer" in prompt
    assert "NARRATIVE DESIGN ARTIFACT FIELDS" in prompt
    assert "story_intent" in prompt
    assert "dialogue_rules" in prompt
