from __future__ import annotations

from orchestrator_role_targets.prompt_compiler import (
    build_technical_designer_prompt,
    build_builder_prompt,
    build_reviewer_prompt,
    build_tester_prompt,
)


def _ctx() -> dict:
    return {
        "work_id": "wd-1",
        "title": "Boss framework",
        "kind": "feature",
        "risk_level": "medium",
        "work_summary_short": "work summary",
        "project_summary_short": "project summary",
        "subproject_summary_short": "subproject summary",
        "builder_summary_short": "builder summary",
        "review_or_work_summary_short": "review summary",
        "focus": "check it",
        "avoid": "avoid drift",
        "changed_files": ["orchestrator/repository.py"],
        "changed_files_text": "orchestrator/repository.py",
        "memory_refs": ["mem://project/overview"],
        "acceptance_criteria": ["criterion A"],
        "requires_test": True,
        "system_design_goal": "Combat framework boundaries",
        "system_design_scope": "bounded system slice",
        "technical_design_goal": "Boss mechanic flow",
        "technical_design_scope": "bounded combat slice",
        "technical_design_behavior_rules": ["Rule A"],
        "technical_design_constraints": ["Constraint A"],
        "technical_design_edge_cases": ["Edge A"],
        "technical_design_flow_notes": ["Flow A"],
        "technical_design_implementation_notes": ["Impl A"],
        "technical_design_acceptance_notes": ["Accept A"],
    }


def test_prompts_render_system_design_sections() -> None:
    ctx = _ctx()
    assert "SYSTEM DESIGN" in build_builder_prompt(ctx)
    assert "SYSTEM DESIGN" in build_reviewer_prompt(ctx)
    assert "SYSTEM DESIGN" in build_tester_prompt(ctx)
    assert "UPSTREAM SYSTEM DESIGN" in build_technical_designer_prompt(ctx)
