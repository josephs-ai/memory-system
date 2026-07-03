"""Adapter bridging Qdrant vector search into the memory retrieval pipeline."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

SCRIPTS_DIR = Path(__file__).resolve().parent
PY = Path.home() / ".openclaw" / "venvs" / "memory-db" / "bin" / "python"
SEARCH_QDRANT = SCRIPTS_DIR / "search_qdrant.py"


def search_qdrant_text(
    query: str,
    limit: int = 10,
    since_epoch: int | None = None,
    until_epoch: int | None = None,
) -> list[dict[str, Any]]:
    cmd = [
        str(PY),
        str(SEARCH_QDRANT),
        "--query",
        query,
        "--limit",
        str(limit),
    ]
    if since_epoch is not None:
        cmd += ["--since-epoch", str(int(since_epoch))]
    if until_epoch is not None:
        cmd += ["--until-epoch", str(int(until_epoch))]

    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=str(SCRIPTS_DIR),
        check=False,
    )

    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or f"search_qdrant.py failed with code {proc.returncode}")

    rows: list[dict[str, Any]] = []
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            if isinstance(obj, dict):
                rows.append(obj)
        except Exception as e:
            import logging
            logging.getLogger("qdrant_adapter").warning(f"Failed to parse qdrant output line: {e} | line: {line[:100]}")
            continue

    return rows
