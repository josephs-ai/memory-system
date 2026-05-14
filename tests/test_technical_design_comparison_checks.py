from __future__ import annotations

import json
from pathlib import Path

from orchestrator_role_targets.adapter_reviewer import run as run_reviewer
from orchestrator_role_targets.adapter_tester import run as run_tester


def _packet(tmp_path: Path, *, with_design: bool) -> str:
    packet = {
        "work_id": "w-1",
        "project_id": "test_project",
        "role": "reviewer",
        "title": "Boss mechanic",
        "kind": "feature",
        "risk_level": "medium",
        "memory_mode_effective": "game_dev",
        "requires_test": True,
        "depends_on_technical_design": True,
        "technical_design": (
            {
                "design_version": "technical_design_v2",
                "design_goal": "Boss flow",
                "mechanic_scope": "bounded slice",
                "behavior_rules": ["Rule A"],
            }
            if with_design else None
        ),
        "changed_files": ["orchestrator/repository.py"],
        "summaries": {
            "work_summary": "summary",
            "project_summary": None,
            "subproject_summary": None,
        },
        "instructions": {
            "focus": "check",
            "avoid": "no drift",
        },
        "tags": ["game_dev"],
        "memory_refs": ["mem://project/overview"],
    }
    p = tmp_path / "packet.json"
    p.write_text(json.dumps(packet), encoding="utf-8")
    return str(p)


def test_reviewer_reports_design_compared_when_present(tmp_path: Path) -> None:
    packet_path = _packet(tmp_path, with_design=True)
    result = run_reviewer(packet_path)
    checks = {c["name"]: c["ok"] for c in result["checks"]}
    assert checks["design_spec_present_when_required"] is True
    assert checks["design_spec_compared_against_change"] is True
    assert result["ok"] is True


def test_reviewer_fails_when_required_design_missing(tmp_path: Path) -> None:
    packet_path = _packet(tmp_path, with_design=False)
    result = run_reviewer(packet_path)
    checks = {c["name"]: c["ok"] for c in result["checks"]}
    assert checks["design_spec_present_when_required"] is False
    assert checks["design_spec_compared_against_change"] is False
    assert result["ok"] is False


def test_tester_reports_design_compared_when_present(tmp_path: Path) -> None:
    packet_path = _packet(tmp_path, with_design=True)
    packet = json.loads(Path(packet_path).read_text(encoding="utf-8"))
    packet["role"] = "tester"
    Path(packet_path).write_text(json.dumps(packet), encoding="utf-8")

    result = run_tester(packet_path)
    checks = {c["name"]: c["ok"] for c in result["checks"]}
    assert checks["design_spec_present_when_required"] is True
    assert checks["design_spec_compared_against_behavior"] is True
    assert result["ok"] is True


def test_tester_fails_when_required_design_missing(tmp_path: Path) -> None:
    packet_path = _packet(tmp_path, with_design=False)
    packet = json.loads(Path(packet_path).read_text(encoding="utf-8"))
    packet["role"] = "tester"
    Path(packet_path).write_text(json.dumps(packet), encoding="utf-8")

    result = run_tester(packet_path)
    checks = {c["name"]: c["ok"] for c in result["checks"]}
    assert checks["design_spec_present_when_required"] is False
    assert checks["design_spec_compared_against_behavior"] is False
    assert result["ok"] is False
