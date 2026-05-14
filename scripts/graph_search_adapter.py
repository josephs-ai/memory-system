"""
Adapter bridging Neo4j graph queries into the memory retrieval pipeline.
Provides code context expansion via graph traversal.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

SCRIPTS_DIR = Path(__file__).resolve().parent
PY = Path.home() / ".openclaw" / "venvs" / "memory-db" / "bin" / "python"
QUERY_NEO4J = SCRIPTS_DIR / "query_neo4j.py"


def search_graph_text(
    query: str,
    limit: int = 10,
    *,
    project_id: str | None = None,
    subproject_id: str | None = None,
    workflow_id: str | None = None,
    pipeline_id: str | None = None,
) -> list[dict[str, Any]]:
    cmd = [
        str(PY),
        str(QUERY_NEO4J),
        "--query",
        query,
        "--limit",
        str(limit),
    ]
    if project_id:
        cmd += ["--project-id", project_id]
    if subproject_id:
        cmd += ["--subproject-id", subproject_id]
    if workflow_id:
        cmd += ["--workflow-id", workflow_id]
    if pipeline_id:
        cmd += ["--pipeline-id", pipeline_id]

    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=str(SCRIPTS_DIR),
        check=False,
    )

    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or f"query_neo4j.py failed with code {proc.returncode}")

    rows: list[dict[str, Any]] = []
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line or line == "NONE":
            continue
        try:
            obj = json.loads(line)
            if isinstance(obj, dict):
                rows.append(obj)
        except Exception:
            continue

    return rows
