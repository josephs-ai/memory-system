from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from context_scope import build_context_scope
from graph_store_neo4j import prepare_graph_item

SCRIPT_DIR = Path(__file__).resolve().parent
EXTRACT_UPDATES = SCRIPT_DIR / "extract_updates.py"


def test_build_context_scope_isolates_same_pipeline_name_by_project():
    a = build_context_scope(
        {
            "text": "In pipeline Alpha, everything must be 100% automated.",
            "project_id": "proj-a",
            "scope": "stable",
        }
    )
    b = build_context_scope(
        {
            "text": "In pipeline Alpha, everything must be 100% automated.",
            "project_id": "proj-b",
            "scope": "stable",
        }
    )

    assert a["scope_type"] == "pipeline"
    assert b["scope_type"] == "pipeline"
    assert a["inheritance_policy"] == "local_only"
    assert b["inheritance_policy"] == "local_only"
    assert a["scope_id"] != b["scope_id"]



def test_prepare_graph_item_anchors_scope_fields_for_graph_ingestion():
    prepared = prepare_graph_item(
        {
            "id": "mem-1",
            "text": "In pipeline Alpha, everything must be 100% automated.",
            "entity": "pipeline_alpha",
            "property": "automation_mode",
            "value": "fully_automated",
            "memory_type": "guardrail",
            "project_id": "proj-a",
            "confidence": 0.95,
            "importance": 0.95,
        }
    )

    assert prepared is not None
    assert prepared["scope_type"] == "pipeline"
    assert prepared["pipeline_id"] == "alpha"
    assert prepared["project_id"] == "proj-a"
    assert prepared["inheritance_policy"] == "local_only"
    assert prepared["scope_id"].startswith("scope-")



def test_extract_updates_emits_context_scope_for_pipeline_rule(tmp_path: Path):
    input_path = tmp_path / "chunk.txt"
    input_path.write_text("In pipeline Alpha, everything must be 100% automated.\n", encoding="utf-8")

    proc = subprocess.run(
        [
            sys.executable,
            str(EXTRACT_UPDATES),
            str(input_path),
            "--source-agent",
            "tester",
            "--source-session",
            "session-1",
        ],
        capture_output=True,
        text=True,
        cwd=str(SCRIPT_DIR),
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    rows = [json.loads(line) for line in proc.stdout.splitlines() if line.strip() and line.strip() != "NONE"]
    assert rows
    item = rows[0]
    assert item["context_scope_type"] == "pipeline"
    assert item["inheritance_policy"] == "local_only"
    assert item["context_scope_id"].startswith("scope-")
    assert item["context_scope_payload"]["pipeline_id"] == "alpha"
