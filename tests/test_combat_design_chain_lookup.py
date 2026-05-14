from __future__ import annotations

import json
import sys
import types
from pathlib import Path

# Stub psycopg before importing dispatch_runner
sys.modules.setdefault("psycopg", types.ModuleType("psycopg"))

import orchestrator.dispatch_runner as dr


def test_combat_chain_lookup_loads_direct_file_artifact(monkeypatch) -> None:
    runtime = Path.home() / ".openclaw" / "workspace" / "orchestrator_runtime"
    runtime.mkdir(parents=True, exist_ok=True)

    artifact = {
        "combat_design_version": "combat_design_v1",
        "combat_goal": "Boss flow",
    }
    (runtime / "combat_design_cd-1.json").write_text(json.dumps(artifact), encoding="utf-8")

    monkeypatch.setattr(dr, "fetch_latest_completed_run_for_work_item", lambda work_id, role: None)
    monkeypatch.setattr(
        dr,
        "fetch_work_item_context",
        lambda work_id: {"depends_on_work_id": None} if work_id == "cd-1" else None,
    )

    result = dr._load_combat_design_from_dependency_chain("cd-1")
    assert result is not None
    assert result["combat_goal"] == "Boss flow"


def test_combat_chain_lookup_walks_upstream(monkeypatch) -> None:
    runtime = Path.home() / ".openclaw" / "workspace" / "orchestrator_runtime"
    runtime.mkdir(parents=True, exist_ok=True)

    artifact = {
        "combat_design_version": "combat_design_v1",
        "combat_goal": "Boss flow",
    }
    (runtime / "combat_design_cd-root.json").write_text(json.dumps(artifact), encoding="utf-8")

    monkeypatch.setattr(dr, "fetch_latest_completed_run_for_work_item", lambda work_id, role: None)

    chain = {
        "child-2": {"depends_on_work_id": "child-1"},
        "child-1": {"depends_on_work_id": "cd-root"},
        "cd-root": {"depends_on_work_id": None},
    }
    monkeypatch.setattr(dr, "fetch_work_item_context", lambda work_id: chain.get(work_id))

    result = dr._load_combat_design_from_dependency_chain("child-2")
    assert result is not None
    assert result["combat_goal"] == "Boss flow"
