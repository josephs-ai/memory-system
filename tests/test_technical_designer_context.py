from orchestrator_role_targets.context_builder import build_context_for_role


def test_technical_designer_context_builds():
    packet = {
        "work_id": "w1",
        "title": "Combat mechanic spec",
        "kind": "feature",
        "risk_level": "medium",
        "memory_mode_effective": "game_dev",
        "tags": ["game_dev", "combat"],
        "changed_files": [],
        "memory_refs": ["mem://a"],
        "summaries": {
            "work_summary": "Need clearer mechanic rules.",
            "project_summary": "Action roguelike combat system.",
            "subproject_summary": "Boss combat subsystem.",
        },
        "instructions": {
            "focus": "Produce a bounded technical design spec.",
            "avoid": "Do not implement the whole system.",
        },
        "requires_review": True,
        "requires_test": True,
    }

    context = build_context_for_role("technical_designer", packet)
    assert context["work_id"] == "w1"
    assert context["kind"] == "feature"
    assert "focus" in context
    assert "memory_refs" in context
