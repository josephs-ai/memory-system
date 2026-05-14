from __future__ import annotations

from orchestrator_role_targets.adapter_planner import _classify_planner_result


def test_planner_requests_system_designer_for_cross_system_feature() -> None:
    packet = {
        "title": "Build combat encounter progression framework",
        "description": "Cross-system feature affecting combat, encounters, progression, and integration.",
        "kind": "feature",
        "risk_level": "medium",
        "tags": ["cross_system"],
        "priority": 50,
        "memory_refs": [],
    }
    context = {"planning_scope": "cross system feature", "guardrails": []}

    result = _classify_planner_result(packet, context)
    assert result["classification"] == "decompose"
    assert "system_designer" in result["specialist_roles_requested"]
    assert "technical_designer" in result["specialist_roles_requested"]


def test_planner_does_not_request_system_designer_for_small_feature() -> None:
    packet = {
        "title": "Small dash timing fix",
        "description": "Tight scoped mechanic tweak.",
        "kind": "feature",
        "risk_level": "low",
        "tags": ["local_fix"],
        "priority": 50,
        "memory_refs": [],
    }
    context = {"planning_scope": "small local change", "guardrails": []}

    result = _classify_planner_result(packet, context)
    if result["classification"] == "decompose":
        assert "system_designer" not in result["specialist_roles_requested"]


def test_planner_requests_system_designer_for_critical_ambiguous_feature() -> None:
    packet = {
        "title": "Architecture decision for progression and room generation",
        "description": "Needs approval for a cross-system architecture decision.",
        "kind": "feature",
        "risk_level": "critical",
        "tags": ["cross_system"],
        "priority": 50,
        "memory_refs": [],
    }
    context = {"planning_scope": "architecture decision", "guardrails": []}

    result = _classify_planner_result(packet, context)
    assert result["classification"] == "needs_human"
    assert "system_designer" in result["specialist_roles_requested"]
    assert "technical_designer" in result["specialist_roles_requested"]
