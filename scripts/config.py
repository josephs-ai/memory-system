"""
config.py — Centralized configuration for the OpenClaw Memory System.

All path and connection defaults are resolved here. Scripts should import
from this module instead of hardcoding paths.

Environment variables:
    OPENCLAW_MEMORY_ROOT   — Root of the memory index (default: parent of scripts/)
    OPENCLAW_MEMORY_DSN    — PostgreSQL connection string
    OPENCLAW_MEMORY_DB_DSN — Alias for OPENCLAW_MEMORY_DSN (legacy compat)
    QDRANT_HOST            — Qdrant server host (default: localhost)
    QDRANT_PORT            — Qdrant server port (default: 6333)
    NEO4J_URI              — Neo4j bolt URI (default: bolt://localhost:7687)
    NEO4J_USER             — Neo4j username (default: neo4j)
    NEO4J_PASSWORD         — Neo4j password (default: neo4jpassword)
    OPENCLAW_RERANK_MODEL  — Cross-encoder model name
    OPENCLAW_RERANK_CAP    — Max items for cross-encoder reranking (default: 20)
"""

from __future__ import annotations

import json
import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------

# Default: memory index root is the parent of the scripts/ directory
_SCRIPT_DIR = Path(__file__).resolve().parent
_DEFAULT_ROOT = _SCRIPT_DIR.parent

MEMORY_ROOT = Path(
    os.environ.get("OPENCLAW_MEMORY_ROOT", str(_DEFAULT_ROOT))
).resolve()

SCRIPTS_DIR = MEMORY_ROOT / "scripts"
HEALTH_DIR = MEMORY_ROOT / "health"
CONFIG_JSON = MEMORY_ROOT / "config.json"

# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------

DB_DSN = (
    os.environ.get("OPENCLAW_MEMORY_DSN")
    or os.environ.get("OPENCLAW_MEMORY_DB_DSN")
    or "dbname=openclaw_memory"
)

# ---------------------------------------------------------------------------
# Qdrant
# ---------------------------------------------------------------------------

QDRANT_HOST = os.environ.get("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.environ.get("QDRANT_PORT", "6333"))

# ---------------------------------------------------------------------------
# Neo4j
# ---------------------------------------------------------------------------

NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "neo4jpassword")

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

EMBEDDING_MODEL = os.environ.get(
    "OPENCLAW_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)
RERANK_MODEL = os.environ.get(
    "OPENCLAW_RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"
)
RERANK_CAP = int(os.environ.get("OPENCLAW_RERANK_CAP", "20"))

# ---------------------------------------------------------------------------
# Config file loader (for legacy scripts that read config.json)
# ---------------------------------------------------------------------------

_config_cache: dict | None = None


def load_config() -> dict:
    """Load config.json from memory root. Returns empty dict if not found."""
    global _config_cache
    if _config_cache is not None:
        return _config_cache
    if CONFIG_JSON.exists():
        _config_cache = json.loads(CONFIG_JSON.read_text(encoding="utf-8"))
    else:
        _config_cache = {}
    return _config_cache
