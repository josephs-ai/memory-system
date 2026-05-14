from __future__ import annotations

from orchestrator_role_targets.prompt_compiler import build_system_designer_prompt


def test_system_designer_prompt_contains_core_sections() -> None:
    ctx = {
        "work_id": "sd-1",
        "title": "Combat framework",
        "kind": "feature",
        "risk_level": "medium",
        "work_summary_short": "none",
        "project_summary_short": "project",
        "subproject_summary_short": "subproject",
        "focus": "Produce a bounded system design.",
        "avoid": "Do not redesign the whole project.",
        "changed_files_text": "(none)",
        "memory_refs": [],
        "acceptance_criteria": ["System design is bounded and actionable."],
    }
    prompt = build_system_designer_prompt(ctx)
    assert "ROLE" in prompt
    assert "System Designer" in prompt
    assert "SYSTEM DESIGN ARTIFACT FIELDS" in prompt
    assert "ownership_map" in prompt
    assert "interface_contracts" in prompt
