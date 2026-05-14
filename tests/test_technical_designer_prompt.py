from orchestrator_role_targets.prompt_compiler import build_prompt_for_role


def test_technical_designer_prompt_builds():
    context = {
        "work_id": "w1",
        "title": "Combat mechanic spec",
        "kind": "feature",
        "risk_level": "medium",
        "work_summary_short": "Need clearer mechanic rules.",
        "project_summary_short": "Action roguelike combat system.",
        "subproject_summary_short": "Boss combat subsystem.",
        "focus": "Produce bounded spec.",
        "avoid": "Do not over-redesign.",
        "memory_refs": ["mem://a"],
        "acceptance_criteria": ["clarify mechanic behavior"],
        "changed_files_text": "(none)",
    }

    prompt = build_prompt_for_role("technical_designer", context)
    assert "Technical Designer" in prompt
    assert "TECHNICAL DESIGN ARTIFACT FIELDS" in prompt
    assert "behavior_rules" in prompt
