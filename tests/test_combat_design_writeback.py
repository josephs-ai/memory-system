from __future__ import annotations

from orchestrator_role_targets.writeback_service import _safe_summary


def test_writeback_safe_summary_includes_combat_design() -> None:
    packet = {
        "work_id": "w1",
        "project_id": "test_project",
        "title": "Boss combat loop",
        "kind": "feature",
        "risk_level": "medium",
        "memory_mode_effective": "game_dev",
        "role": "writeback",
        "summaries": {},
        "instructions": {},
        "combat_design": {
            "combat_design_version": "combat_design_v1",
            "combat_goal": "Boss pressure loop",
        },
    }
    summary = _safe_summary(packet)
    assert summary["combat_design"] is not None
    assert summary["combat_design"]["combat_goal"] == "Boss pressure loop"
