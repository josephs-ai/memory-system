from __future__ import annotations

import json
from pathlib import Path

import orchestrator.dispatch_runner as dr


def test_chain_lookup_loads_direct_file_artifact(monkeypatch) -> None:
    runtime = Path.home() / ".openclaw" / "workspace" / "orchestrator_runtime"
    runtime.mkdir(parents=True, exist_ok=True)

    artifact = {
        "design_version": "technical_design_v2",
        "design_goal": "Boss flow",
        "mechanic_scope": "bounded slice",
    }
    (runtime / "technical_design_td-1.json").write_text(json.dumps(artifact), encoding="utf-8")

    monkeypatch.setattr(dr, "fetch_latest_completed_run_for_work_item", lambda work_id, role: None)
    monkeypatch.setattr(
        dr,
        "fetch_work_item_context",
        lambda work_id: {"depends_on_work_id": None} if work_id == "td-1" else None,
    )

    result = dr._load_technical_design_from_dependency_chain("td-1")
    assert result is not None
    assert result["design_goal"] == "Boss flow"


def test_chain_lookup_walks_upstream(monkeypatch) -> None:
    runtime = Path.home() / ".openclaw" / "workspace" / "orchestrator_runtime"
    runtime.mkdir(parents=True, exist_ok=True)

    artifact = {
        "design_version": "technical_design_v2",
        "design_goal": "Boss flow",
        "mechanic_scope": "bounded slice",
    }
    (runtime / "technical_design_td-root.json").write_text(json.dumps(artifact), encoding="utf-8")

    monkeypatch.setattr(dr, "fetch_latest_completed_run_for_work_item", lambda work_id, role: None)

    chain = {
        "child-2": {"depends_on_work_id": "child-1"},
        "child-1": {"depends_on_work_id": "td-root"},
        "td-root": {"depends_on_work_id": None},
    }
    monkeypatch.setattr(dr, "fetch_work_item_context", lambda work_id: chain.get(work_id))

    result = dr._load_technical_design_from_dependency_chain("child-2")
    assert result is not None
    assert result["design_goal"] == "Boss flow"
