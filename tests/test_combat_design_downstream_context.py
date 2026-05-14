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
        "combat_design": {
            "combat_goal": "Combat framework boundaries",
            "player_experience_targets": ["Readable pressure"],
            "core_mechanics": ["Dash", "Melee"],
            "attack_loop": ["Attack cadence"],
            "defense_loop": ["Dodge timing"],
            "movement_pressure_rules": ["Keep moving"],
            "enemy_pressure_rules": ["Force reactions"],
            "telegraph_rules": ["Readable telegraphs"],
            "hit_reaction_rules": ["Clear hit confirm"],
            "invulnerability_rules": ["Short i-frames"],
            "resource_tension": ["Commitment windows"],
            "tuning_targets": ["Fast but fair"],
            "failure_cases": ["Unreadable overlaps"],
            "reviewer_focus": ["Compare implementation to combat pacing goals"],
            "tester_focus": ["Validate dodge windows and telegraph readability"],
            "acceptance_notes": ["Combat loop is bounded and actionable"],
        },
    }


def test_downstream_contexts_include_combat_design() -> None:
    packet = _packet()

    td = build_technical_designer_context(packet)
    builder = build_builder_context(packet)
    reviewer = build_reviewer_context(packet)
    tester = build_tester_context(packet)
    writeback = build_writeback_context(packet)

    assert td["combat_design_goal"] == "Combat framework boundaries"
    assert builder["combat_design_goal"] == "Combat framework boundaries"
    assert reviewer["combat_design_goal"] == "Combat framework boundaries"
    assert tester["combat_design_goal"] == "Combat framework boundaries"
    assert writeback["combat_design_goal"] == "Combat framework boundaries"
