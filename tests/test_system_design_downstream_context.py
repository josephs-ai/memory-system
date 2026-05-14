from __future__ import annotations

from orchestrator_role_targets.context_builder import (
    build_technical_designer_context,
    build_builder_context,
    build_reviewer_context,
    build_tester_context,
    build_writeback_context,
)


def _packet() -> dict:
    return {
        "work_id": "wd-1",
        "title": "Boss framework",
        "kind": "feature",
        "risk_level": "medium",
        "memory_mode_effective": "game_dev",
        "tags": ["game_dev", "cross_system"],
        "summaries": {
            "work_summary": "work summary",
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
        "system_design": {
            "system_goal": "Combat framework boundaries",
            "scope": "bounded system slice",
            "subsystems_touched": ["combat", "encounters"],
            "ownership_map": ["Combat owns attack resolution"],
            "interface_contracts": ["Encounter -> combat event contract"],
            "data_contracts": ["Attack profile data contract"],
            "event_flow": ["Encounter starts combat flow"],
            "integration_order": ["System first"],
            "risks": ["coupling risk"],
            "guardrails": ["bounded only"],
            "reviewer_focus": ["validate subsystem boundaries"],
            "tester_focus": ["validate integration behavior"],
            "technical_designer_followups": ["refine mechanic states"],
            "builder_constraints": ["do not bypass interfaces"],
            "acceptance_notes": ["system design actionable"],
        },
        "technical_design": {
            "design_goal": "Boss mechanic flow",
            "mechanic_scope": "bounded combat slice",
            "behavior_rules": ["Rule A"],
            "constraints": ["Constraint A"],
            "edge_cases": ["Edge A"],
            "state_or_flow_notes": ["Flow A"],
            "implementation_notes": ["Impl A"],
            "acceptance_notes": ["Accept A"],
        },
    }


def test_downstream_contexts_include_system_design() -> None:
    packet = _packet()

    td = build_technical_designer_context(packet)
    builder = build_builder_context(packet)
    reviewer = build_reviewer_context(packet)
    tester = build_tester_context(packet)
    writeback = build_writeback_context(packet)

    assert td["system_design_goal"] == "Combat framework boundaries"
    assert builder["system_design_goal"] == "Combat framework boundaries"
    assert reviewer["system_design_goal"] == "Combat framework boundaries"
    assert tester["system_design_goal"] == "Combat framework boundaries"
    assert writeback["system_design_goal"] == "Combat framework boundaries"
