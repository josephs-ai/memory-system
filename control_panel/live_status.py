from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import psycopg
from datetime import datetime

WORKSPACE = Path.home() / ".openclaw" / "workspace"
HEARTBEAT_FILE = WORKSPACE / "HEARTBEAT.md"
HEARTBEAT_AGENT_FILE = WORKSPACE / "HEARTBEAT_AGENTS.md"
REVIEW_DIR = WORKSPACE / "memory" / "review"
DECISIONS_LOG = REVIEW_DIR / "decisions.log"

DEFAULT_DSN = os.environ.get("OPENCLAW_MEMORY_DB_DSN", "dbname=openclaw_memory user=postgres")


def db_query(sql: str, params: tuple = ()) -> list[dict[str, Any]]:
    try:
        with psycopg.connect(DEFAULT_DSN) as conn:
            with conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
                cur.execute(sql, params)
                return list(cur.fetchall())
    except Exception:
        return []


def db_scalar(sql: str, params: tuple = ()) -> int:
    try:
        with psycopg.connect(DEFAULT_DSN) as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                row = cur.fetchone()
                return int(row[0]) if row and row[0] is not None else 0
    except Exception:
        return 0


def safe_mtime(path: Path) -> float | None:
    try:
        return path.stat().st_mtime
    except Exception:
        return None


def recent_queue_items(limit: int = 8) -> dict[str, list[dict[str, Any]]]:
    return {
        "inbox": db_query(
            """
            SELECT id, text, memory_type, scope, entity, property, value, confidence, importance
            FROM memory_inbox
            ORDER BY first_seen DESC NULLS LAST
            LIMIT %s
            """,
            (limit,),
        ),
        "pending_stable": db_query(
            """
            SELECT id, text, memory_type, scope, entity, property, value, confidence, importance
            FROM memory_pending_stable
            ORDER BY first_seen DESC NULLS LAST
            LIMIT %s
            """,
            (limit,),
        ),
        "canonical": db_query(
            """
            SELECT id, text, memory_type, scope, entity, property, value, confidence, importance
            FROM memory_items
            WHERE status = 'active'
            ORDER BY last_confirmed DESC NULLS LAST
            LIMIT %s
            """,
            (limit,),
        ),
        "discarded": db_query(
            """
            SELECT id, text, memory_type, scope, entity, property, value, confidence, importance
            FROM memory_discarded
            ORDER BY first_seen DESC NULLS LAST
            LIMIT %s
            """,
            (limit,),
        ),
    }


def recent_conflicts(limit: int = 10) -> list[dict[str, Any]]:
    groups = db_query(
        """
        SELECT entity, property, COUNT(*) AS variants
        FROM memory_items
        WHERE status = 'active'
          AND entity IS NOT NULL
          AND property IS NOT NULL
        GROUP BY entity, property
        HAVING COUNT(*) > 1
        ORDER BY variants DESC, entity, property
        LIMIT %s
        """,
        (limit,),
    )

    out = []
    for g in groups:
        variants = db_query(
            """
            SELECT id, text, value, confidence, importance, last_confirmed
            FROM memory_items
            WHERE status = 'active'
              AND entity = %s
              AND property = %s
            ORDER BY importance DESC NULLS LAST, confidence DESC NULLS LAST, last_confirmed DESC NULLS LAST
            """,
            (g["entity"], g["property"]),
        )
        out.append(
            {
                "entity": g["entity"],
                "property": g["property"],
                "variants": g["variants"],
                "items": variants,
            }
        )

    return out

def fmt_mtime(path: Path) -> str | None:
    ts = safe_mtime(path)
    if ts is None:
        return None
    try:
        return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return str(ts)


def heartbeat_status() -> dict[str, Any]:
    now_ts = datetime.now().timestamp()

    hb_raw = safe_mtime(HEARTBEAT_FILE)
    hb_agent_raw = safe_mtime(HEARTBEAT_AGENT_FILE)
    log_raw = safe_mtime(DECISIONS_LOG)

    hb_mtime = fmt_mtime(HEARTBEAT_FILE)
    hb_agent_mtime = fmt_mtime(HEARTBEAT_AGENT_FILE)
    log_mtime = fmt_mtime(DECISIONS_LOG)

    raw_times = [x for x in [hb_raw, hb_agent_raw, log_raw] if x is not None]
    latest_raw = max(raw_times) if raw_times else None
    stale_seconds = None if latest_raw is None else int(now_ts - latest_raw)

    healthy = latest_raw is not None and stale_seconds < 900
    warning = latest_raw is not None and 900 <= stale_seconds < 1800
    critical = latest_raw is None or stale_seconds >= 1800

    return {
        "heartbeat_file_exists": HEARTBEAT_FILE.exists(),
        "heartbeat_agent_file_exists": HEARTBEAT_AGENT_FILE.exists(),
        "heartbeat_file_mtime": hb_mtime,
        "heartbeat_agent_file_mtime": hb_agent_mtime,
        "decisions_log_exists": DECISIONS_LOG.exists(),
        "decisions_log_mtime": log_mtime,
        "stale_seconds": stale_seconds,
        "healthy": healthy,
        "warning": warning,
        "critical": critical,
    }

def queue_counts() -> dict[str, int]:
    return {
        "inbox": db_scalar("SELECT COUNT(*) FROM memory_inbox"),
        "pending_stable": db_scalar("SELECT COUNT(*) FROM memory_pending_stable"),
        "discarded": db_scalar("SELECT COUNT(*) FROM memory_discarded"),
        "canonical_active": db_scalar("SELECT COUNT(*) FROM memory_items WHERE status = 'active'"),
        "embeddings": db_scalar("SELECT COUNT(*) FROM memory_item_embeddings"),
        "feedback": db_scalar("SELECT COUNT(*) FROM retrieval_feedback"),
    }


def get_live_status() -> dict[str, Any]:
    return {
        "queues": queue_counts(),
        "recent_memory": recent_queue_items(),
        "conflicts": recent_conflicts(),
        "heartbeat": heartbeat_status(),
    }
