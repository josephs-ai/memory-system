from orchestrator_role_targets.adapter_planner import _classify_planner_result


def test_direct_execute_for_small_feature():
    packet = {
        "title": "Small movement fix",
        "description": "Tight scoped fix.",
        "kind": "feature",
        "risk_level": "medium",
        "priority": 50,
        "tags": ["game_dev"],
        "memory_refs": [],
    }
    context = {
        "planning_scope": {"mode": "game_dev"},
        "guardrails": [],
    }

    result = _classify_planner_result(packet, context)
    assert result["classification"] == "direct_execute"
    assert result["closure_policy"] == "none"


def test_decompose_for_broad_feature():
    packet = {
        "title": "Implement cross-system combat and movement coordination feature",
        "description": "This feature spans multiple gameplay systems and requires a broader implementation plan across movement, combat, and integration behavior.",
        "kind": "feature",
        "risk_level": "medium",
        "priority": 50,
        "tags": ["game_dev", "cross_system"],
        "memory_refs": [],
    }
    context = {
        "planning_scope": {"mode": "game_dev"},
        "guardrails": [],
    }

    result = _classify_planner_result(packet, context)
    assert result["classification"] == "decompose"
    assert result["closure_policy"] == "all_children_complete"
    assert len(result["proposed_child_items"]) >= 1


def test_needs_human_for_ambiguous_direction():
    packet = {
        "title": "Decide combat direction and architecture tradeoff for boss fight system",
        "description": "This task is ambiguous and involves a product choice plus an architecture decision. The approach is unclear, a decision is needed.",
        "kind": "feature",
        "risk_level": "medium",
        "priority": 50,
        "tags": ["game_dev"],
        "memory_refs": [],
    }
    context = {
        "planning_scope": {"mode": "game_dev"},
        "guardrails": [],
    }

    result = _classify_planner_result(packet, context)
    assert result["classification"] == "needs_human"
    assert result["requires_human_approval"] is True
    assert result["approval_reason"]
