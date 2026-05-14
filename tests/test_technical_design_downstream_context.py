from __future__ import annotations

from orchestrator_role_targets.context_builder import (
    build_reviewer_context,
    build_tester_context,
)


def _packet() -> dict:
    return {
        "work_id": "wd-1",
        "title": "Boss mechanic",
        "kind": "feature",
        "risk_level": "medium",
        "memory_mode_effective": "game_dev",
        "tags": ["game_dev", "combat"],
        "summaries": {
            "work_summary": "builder summary",
            "project_summary": "project summary",
            "subproject_summary": "subproject summary",
        },
        "instructions": {
            "focus": "check it",
            "avoid": "do not drift",
        },
        "memory_refs": ["mem://project/overview"],
        "changed_files": ["orchestrator/repository.py"],
        "requires_test": True,
        "technical_design": {
            "design_goal": "Boss fight combat flow",
            "mechanic_scope": "bounded combat slice",
            "behavior_rules": ["Rule A", "Rule B"],
            "constraints": ["Constraint A"],
            "edge_cases": ["Edge A"],
            "state_or_flow_notes": ["Flow A"],
            "implementation_notes": ["Impl A"],
            "acceptance_notes": ["Accept A"],
        },
    }


def test_reviewer_context_includes_technical_design():
    ctx = build_reviewer_context(_packet())
    assert ctx["technical_design_goal"] == "Boss fight combat flow"
    assert ctx["technical_design_scope"] == "bounded combat slice"
    assert "Rule A" in ctx["technical_design_behavior_rules"]
    assert "Constraint A" in ctx["technical_design_constraints"]
    assert "Edge A" in ctx["technical_design_edge_cases"]
    assert "Flow A" in ctx["technical_design_flow_notes"]
    assert "Impl A" in ctx["technical_design_implementation_notes"]
    assert "Accept A" in ctx["technical_design_acceptance_notes"]


def test_tester_context_includes_technical_design():
    ctx = build_tester_context(_packet())
    assert ctx["technical_design_goal"] == "Boss fight combat flow"
    assert ctx["technical_design_scope"] == "bounded combat slice"
    assert "Rule B" in ctx["technical_design_behavior_rules"]
    assert "Constraint A" in ctx["technical_design_constraints"]
    assert "Edge A" in ctx["technical_design_edge_cases"]
    assert "Flow A" in ctx["technical_design_flow_notes"]
    assert "Impl A" in ctx["technical_design_implementation_notes"]
    assert "Accept A" in ctx["technical_design_acceptance_notes"]
