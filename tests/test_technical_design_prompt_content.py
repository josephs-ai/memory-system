from __future__ import annotations

from orchestrator_role_targets.prompt_compiler import build_reviewer_prompt, build_tester_prompt


def _ctx() -> dict:
    return {
        "work_id": "wd-1",
        "title": "Boss mechanic",
        "kind": "feature",
        "risk_level": "medium",
        "builder_summary_short": "builder summary",
        "review_or_work_summary_short": "review summary",
        "project_summary_short": "project summary",
        "focus": "check it",
        "avoid": "do not drift",
        "changed_files": ["orchestrator/repository.py"],
        "changed_files_text": "orchestrator/repository.py",
        "memory_refs": ["mem://project/overview"],
        "acceptance_criteria": ["criterion A"],
        "requires_test": True,
        "technical_design_goal": "Boss fight combat flow",
        "technical_design_scope": "bounded combat slice",
        "technical_design_system_intent": ["Intent A"],
        "technical_design_behavior_rules": ["Rule A", "Rule B"],
        "technical_design_constraints": ["Constraint A"],
        "technical_design_invariants": ["Invariant A"],
        "technical_design_failure_modes": ["Failure A"],
        "technical_design_edge_cases": ["Edge A"],
        "technical_design_flow_notes": ["Flow A"],
        "technical_design_interfaces_touched": ["Interface A"],
        "technical_design_observability_notes": ["Observe A"],
        "technical_design_implementation_notes": ["Impl A"],
        "technical_design_reviewer_focus": ["Review A"],
        "technical_design_tester_focus": ["Test A"],
        "technical_design_acceptance_notes": ["Accept A"],
    }


def test_reviewer_prompt_contains_technical_design_section():
    prompt = build_reviewer_prompt(_ctx())
    assert "TECHNICAL DESIGN" in prompt
    assert "design_goal: Boss fight combat flow" in prompt
    assert "mechanic_scope: bounded combat slice" in prompt
    assert "BEHAVIOR RULES" in prompt
    assert "- Rule A" in prompt
    assert "CONSTRAINTS" in prompt
    assert "- Constraint A" in prompt
    assert "EDGE CASES" in prompt
    assert "- Edge A" in prompt
    assert "FLOW NOTES" in prompt
    assert "- Flow A" in prompt
    assert "IMPLEMENTATION NOTES" in prompt
    assert "- Impl A" in prompt
    assert "ACCEPTANCE NOTES" in prompt
    assert "- Accept A" in prompt


def test_tester_prompt_contains_technical_design_section():
    prompt = build_tester_prompt(_ctx())
    assert "TECHNICAL DESIGN" in prompt
    assert "design_goal: Boss fight combat flow" in prompt
    assert "mechanic_scope: bounded combat slice" in prompt
    assert "BEHAVIOR RULES" in prompt
    assert "- Rule B" in prompt
    assert "CONSTRAINTS" in prompt
    assert "- Constraint A" in prompt
    assert "EDGE CASES" in prompt
    assert "- Edge A" in prompt
    assert "FLOW NOTES" in prompt
    assert "- Flow A" in prompt
    assert "IMPLEMENTATION NOTES" in prompt
    assert "- Impl A" in prompt
    assert "ACCEPTANCE NOTES" in prompt
    assert "- Accept A" in prompt
