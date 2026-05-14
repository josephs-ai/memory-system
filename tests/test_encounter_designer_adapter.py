from __future__ import annotations

import json
from pathlib import Path

from orchestrator_role_targets.adapter_encounter_designer import run


def test_encounter_designer_adapter_emits_artifact(tmp_path: Path) -> None:
    packet = {
        "work_id": "ed-1",
        "project_id": "test_project",
        "role": "encounter_designer",
        "title": "Boss arena encounter",
        "kind": "feature",
        "risk_level": "medium",
        "memory_mode_effective": "game_dev",
        "instructions": {
            "focus": "Produce bounded encounter design",
            "avoid": "Do not redesign whole game",
        },
        "summaries": {},
        "tags": ["encounter", "boss", "arena"],
        "memory_refs": [],
        "changed_files": [],
    }
    packet_path = tmp_path / "packet.json"
    packet_path.write_text(json.dumps(packet), encoding="utf-8")

    result = run(str(packet_path))
    assert result["ok"] is True
    checks = {c["name"]: c["ok"] for c in result["checks"]}
    assert checks["encounter_design_artifact_persisted"] is True
    assert any(a["kind"] == "encounter_design_spec" for a in result["artifacts"])
