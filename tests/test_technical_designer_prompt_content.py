from __future__ import annotations

from orchestrator_role_targets.prompt_compiler import (
    build_technical_designer_prompt,
    build_reviewer_prompt,
    build_tester_prompt,
)

def test_technical_designer_prompt_contains_required_sections() -> None:
    ctx = {
        "work_id": "td-1",
        "title": "Boss flow spec",
        "kind": "feature",
        "risk_level": "medium",
        "work_summary_short": "none",
        "project_summary_short": "project",
        "subproject_summary_short": "subproject",
        "focus": "Produce a bounded technical design spec.",
        "avoid": "Do not own full implementation.",
        "changed_files_text": "(none)",
        "memory_refs": [],
        "acceptance_criteria": ["Spec is bounded and actionable."],
    }
    prompt = build_technical_designer_prompt(ctx)
    assert "ROLE" in prompt
    assert "Technical Designer" in prompt
    assert "TECHNICAL DESIGN ARTIFACT FIELDS" in prompt
    assert "design_version" in prompt
    assert "reviewer_focus" in prompt
    assert "tester_focus" in prompt

def test_reviewer_prompt_contains_technical_design_section() -> None:
    ctx = {
        "work_id": "rv-1",
        "title": "Boss flow impl",
        "kind": "feature",
        "risk_level": "medium",
        "builder_summary_short": "builder summary",
        "project_summary_short": "project summary",
        "focus": "review",
        "avoid": "no drift",
        "changed_files": ["orchestrator/repository.py"],
        "changed_files_text": "orchestrator/repository.py",
        "memory_refs": ["mem://project/overview"],
        "acceptance_criteria": ["criterion A"],
        "requires_test": True,
        "technical_design_goal": "Boss fight combat flow",
        "technical_design_scope": "bounded combat slice",
        "technical_design_behavior_rules": ["Rule A"],
        "technical_design_constraints": ["Constraint A"],
        "technical_design_edge_cases": ["Edge A"],
        "technical_design_flow_notes": ["Flow A"],
        "technical_design_implementation_notes": ["Impl A"],
        "technical_design_acceptance_notes": ["Accept A"],
    }
    prompt = build_reviewer_prompt(ctx)
    assert "TECHNICAL DESIGN" in prompt
    assert "Boss fight combat flow" in prompt
    assert "Rule A" in prompt
    assert "Accept A" in prompt

def test_tester_prompt_contains_technical_design_section() -> None:
    ctx = {
        "work_id": "ts-1",
        "title": "Boss flow impl",
        "kind": "feature",
        "risk_level": "medium",
        "review_or_work_summary_short": "review summary",
        "project_summary_short": "project summary",
        "focus": "test",
        "avoid": "no drift",
        "changed_files": ["orchestrator/repository.py"],
        "changed_files_text": "orchestrator/repository.py",
        "memory_refs": ["mem://project/overview"],
        "acceptance_criteria": ["criterion A"],
        "technical_design_goal": "Boss fight combat flow",
        "technical_design_scope": "bounded combat slice",
        "technical_design_behavior_rules": ["Rule A"],
        "technical_design_constraints": ["Constraint A"],
        "technical_design_edge_cases": ["Edge A"],
        "technical_design_flow_notes": ["Flow A"],
        "technical_design_implementation_notes": ["Impl A"],
        "technical_design_acceptance_notes": ["Accept A"],
    }
    prompt = build_tester_prompt(ctx)
    assert "TECHNICAL DESIGN" in prompt
    assert "Boss fight combat flow" in prompt
    assert "Rule A" in prompt
    assert "Accept A" in prompt
