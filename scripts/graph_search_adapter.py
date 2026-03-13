from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

SCRIPTS_DIR = Path(__file__).resolve().parent
PY = Path.home() / ".openclaw" / "venvs" / "memory-db" / "bin" / "python"
QUERY_NEO4J = SCRIPTS_DIR / "query_neo4j.py"


def search_graph_text(query: str, limit: int = 10) -> list[dict[str, Any]]:
    """
    Thin adapter around the existing CLI-based query_neo4j.py.
    This preserves the core Neo4j component instead of deleting it.
    """
    proc = subprocess.run(
        [
            str(PY),
            str(QUERY_NEO4J),
            "--query",
            query,
            "--limit",
            str(limit),
        ],
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
        if not line:
            continue
        try:
            obj = json.loads(line)
            if isinstance(obj, dict):
                rows.append(obj)
        except Exception:
            continue

    return rows
