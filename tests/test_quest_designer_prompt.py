from __future__ import annotations

from orchestrator_role_targets.prompt_compiler import build_quest_designer_prompt


def test_quest_designer_prompt_contains_core_sections() -> None:
    ctx = {
        "work_id": "qd-1",
        "title": "Quest objective flow",
        "kind": "feature",
        "risk_level": "medium",
        "work_summary_short": "none",
        "project_summary_short": "project",
        "subproject_summary_short": "subproject",
        "focus": "Produce a bounded quest design.",
        "avoid": "Do not redesign the whole game.",
        "changed_files_text": "(none)",
        "memory_refs": [],
        "acceptance_criteria": ["Quest design is bounded and actionable."],
    }
    prompt = build_quest_designer_prompt(ctx)
    assert "ROLE" in prompt
    assert "Quest Designer" in prompt
    assert "QUEST DESIGN ARTIFACT FIELDS" in prompt
    assert "objective_flow" in prompt
    assert "reward_rules" in prompt
