from __future__ import annotations

import json
from pathlib import Path

from orchestrator_role_targets.adapter_reviewer import run


def test_reviewer_fails_when_required_combat_design_missing(tmp_path: Path) -> None:
    packet = {
        "work_id": "rev-1",
        "project_id": "test_project",
        "role": "reviewer",
        "title": "Review combat change",
        "kind": "feature",
        "risk_level": "medium",
        "memory_mode_effective": "game_dev",
        "requires_test": True,
        "depends_on_combat_design": True,
        "combat_design": {},
        "depends_on_technical_design": False,
        "technical_design": {},
        "summaries": {"work_summary": "builder completed"},
        "instructions": {"focus": "review", "avoid": "none"},
        "changed_files": [],
        "tags": ["combat"],
        "memory_refs": [],
    }
    packet_path = tmp_path / "packet.json"
    packet_path.write_text(json.dumps(packet), encoding="utf-8")

    result = run(str(packet_path))
    assert result["ok"] is False
    checks = {c["name"]: c["ok"] for c in result["checks"]}
    assert checks["combat_design_present_when_required"] is False
    assert any("combat design spec missing" in issue["message"].lower() for issue in result["issues"])


def test_reviewer_passes_when_required_combat_design_present(tmp_path: Path) -> None:
    packet = {
        "work_id": "rev-2",
        "project_id": "test_project",
        "role": "reviewer",
        "title": "Review combat change",
        "kind": "feature",
        "risk_level": "medium",
        "memory_mode_effective": "game_dev",
        "requires_test": True,
        "depends_on_combat_design": True,
        "combat_design": {"combat_goal": "Boss pressure loop"},
        "depends_on_technical_design": False,
        "technical_design": {},
        "summaries": {"work_summary": "builder completed"},
        "instructions": {"focus": "review", "avoid": "none"},
        "changed_files": [],
        "tags": ["combat"],
        "memory_refs": [],
    }
    packet_path = tmp_path / "packet.json"
    packet_path.write_text(json.dumps(packet), encoding="utf-8")

    result = run(str(packet_path))
    assert result["ok"] is True
    checks = {c["name"]: c["ok"] for c in result["checks"]}
    assert checks["combat_design_present_when_required"] is True
    assert checks["combat_design_compared_against_change"] is True
