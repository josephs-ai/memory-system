from __future__ import annotations

import json
from pathlib import Path

from orchestrator_role_targets.adapter_tester import run


def test_tester_fails_when_required_combat_design_missing(tmp_path: Path) -> None:
    packet = {
        "work_id": "test-1",
        "project_id": "test_project",
        "role": "tester",
        "title": "Test combat change",
        "kind": "feature",
        "risk_level": "medium",
        "memory_mode_effective": "game_dev",
        "depends_on_combat_design": True,
        "combat_design": {},
        "depends_on_technical_design": False,
        "technical_design": {},
        "summaries": {"work_summary": "review passed"},
        "instructions": {"focus": "test", "avoid": "none"},
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


def test_tester_passes_when_required_combat_design_present(tmp_path: Path) -> None:
    packet = {
        "work_id": "test-2",
        "project_id": "test_project",
        "role": "tester",
        "title": "Test combat change",
        "kind": "feature",
        "risk_level": "medium",
        "memory_mode_effective": "game_dev",
        "depends_on_combat_design": True,
        "combat_design": {"combat_goal": "Boss pressure loop"},
        "depends_on_technical_design": False,
        "technical_design": {},
        "summaries": {"work_summary": "review passed"},
        "instructions": {"focus": "test", "avoid": "none"},
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
    assert checks["combat_design_compared_against_behavior"] is True

