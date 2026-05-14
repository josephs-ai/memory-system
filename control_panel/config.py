"""Control panel configuration — paths, constants, defaults."""
from __future__ import annotations

import json
import os
from pathlib import Path

WORKSPACE = Path.home() / ".openclaw" / "workspace"
MEMORY_INDEX = WORKSPACE / ".memory-index"
CONTROL_PANEL_DIR = MEMORY_INDEX / "control_panel"
SCRIPTS_DIR = MEMORY_INDEX / "scripts"
OPENCLAW_CONFIG = Path.home() / ".openclaw" / "openclaw.json"

DEFAULT_PORT = int(os.environ.get("OPENCLAW_CONTROL_PANEL_PORT", "8788"))
PY = os.environ.get(
    "OPENCLAW_MEMORY_PY",
    str(Path.home() / ".openclaw" / "venvs" / "memory-db" / "bin" / "python"),
)
DB_DSN = os.environ.get("OPENCLAW_MEMORY_DB_DSN", "dbname=openclaw_memory")
NEO4J_URI = os.environ.get("OPENCLAW_NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.environ.get("OPENCLAW_NEO4J_USER", "neo4j")
NEO4J_PASS = os.environ.get("OPENCLAW_NEO4J_PASSWORD", "neo4jpassword")
QDRANT_URL = os.environ.get("OPENCLAW_QDRANT_URL", "http://localhost:6333")


def load_openclaw_config() -> dict:
    """Load the real OpenClaw config from openclaw.json."""
    try:
        # openclaw.json uses JSON5 (trailing commas, comments) — try json5 first
        text = OPENCLAW_CONFIG.read_text(encoding="utf-8")
        try:
            import json5
            return json5.loads(text)
        except ImportError:
            pass
        # Fallback: strip trailing commas crudely and parse
        import re
        cleaned = re.sub(r',\s*([}\]])', r'\1', text)
        return json.loads(cleaned)
    except Exception:
        return {}
