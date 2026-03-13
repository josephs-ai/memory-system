from __future__ import annotations

import json
import os
from contextlib import contextmanager
from typing import Any

import psycopg
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from pathlib import Path

WORKSPACE = Path.home() / ".openclaw" / "workspace"
MEMORY_INDEX_DIR = WORKSPACE / ".memory-index"
ENV_FILE = MEMORY_INDEX_DIR / ".env"


def load_env_file(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not path.exists():
        return out

    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            out[key] = value
    return out


for _k, _v in load_env_file(ENV_FILE).items():
    os.environ.setdefault(_k, _v)

DEFAULT_DSN = os.environ.get("OPENCLAW_MEMORY_DB_DSN", "dbname=openclaw_memory user=postgres")
CHECKPOINT_LOCK_KEY = 913004271
STALE_PROCESSING_SECONDS = int(os.environ.get("OPENCLAW_CHECKPOINT_STALE_SECONDS", "300"))

POOL = ConnectionPool(
    conninfo=DEFAULT_DSN,
    min_size=1,
    max_size=8,
    kwargs={"row_factory": dict_row},
    open=True,
)


def close_pool() -> None:
    try:
        POOL.close()
    except Exception:
        pass


def ensure_checkpoint_tables() -> None:
    with POOL.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS memory_checkpoints (
                    id BIGSERIAL PRIMARY KEY,
                    trigger_type TEXT NOT NULL,
                    trigger_reason TEXT,
                    status TEXT NOT NULL,
                    started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    finished_at TIMESTAMPTZ,
                    delta_meta JSONB NOT NULL DEFAULT '{}'::jsonb,
                    counts JSONB NOT NULL DEFAULT '{}'::jsonb,
                    error_text TEXT
                )
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS memory_runtime_state (
                    key TEXT PRIMARY KEY,
                    value JSONB NOT NULL,
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS memory_agent_load_state (
                    agent_name TEXT PRIMARY KEY,
                    last_loaded_checkpoint_id BIGINT,
                    loaded_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
                """
            )
        conn.commit()
   
    mark_stale_processing_checkpoints()
    
    ensure_trigger_tables()

def mark_stale_processing_checkpoints(max_age_seconds: int = STALE_PROCESSING_SECONDS) -> int:
    with POOL.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE memory_checkpoints
                SET status = 'failed',
                    finished_at = NOW(),
                    error_text = COALESCE(error_text, '') ||
                        CASE
                            WHEN COALESCE(error_text, '') = '' THEN ''
                            ELSE E'\n'
                        END ||
                        %s
                WHERE status = 'processing'
                  AND started_at < NOW() - (%s::text || ' seconds')::interval
                RETURNING id
                """,
                (f"janitor_marked_stale_after_{max_age_seconds}s", str(max_age_seconds)),
            )
            rows = cur.fetchall()
        conn.commit()
        return len(rows)


def get_state(key: str, default: Any = None) -> Any:
    with POOL.connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT value FROM memory_runtime_state WHERE key = %s", (key,))
            row = cur.fetchone()
            return row["value"] if row else default


def set_state(key: str, value: Any) -> None:
    with POOL.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO memory_runtime_state (key, value, updated_at)
                VALUES (%s, %s::jsonb, NOW())
                ON CONFLICT (key)
                DO UPDATE SET value = EXCLUDED.value, updated_at = NOW()
                """,
                (key, json.dumps(value)),
            )
        conn.commit()


def begin_checkpoint(trigger_type: str, trigger_reason: str, delta_meta: dict[str, Any]) -> int:
    with POOL.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO memory_checkpoints (trigger_type, trigger_reason, status, delta_meta)
                VALUES (%s, %s, 'processing', %s::jsonb)
                RETURNING id
                """,
                (trigger_type, trigger_reason, json.dumps(delta_meta)),
            )
            checkpoint_id = cur.fetchone()["id"]
        conn.commit()
        return checkpoint_id


def update_checkpoint(
    checkpoint_id: int,
    *,
    status: str,
    counts: dict[str, Any] | None = None,
    error_text: str | None = None,
) -> None:
    with POOL.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE memory_checkpoints
                SET status = %s,
                    finished_at = CASE
                        WHEN %s IN ('committed', 'failed') THEN NOW()
                        ELSE finished_at
                    END,
                    counts = COALESCE(%s::jsonb, counts),
                    error_text = %s
                WHERE id = %s
                """,
                (
                    status,
                    status,
                    json.dumps(counts) if counts is not None else None,
                    error_text,
                    checkpoint_id,
                ),
            )
        conn.commit()


def latest_committed_checkpoint_id() -> int | None:
    with POOL.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id
                FROM memory_checkpoints
                WHERE status = 'committed'
                ORDER BY id DESC
                LIMIT 1
                """
            )
            row = cur.fetchone()
            return row["id"] if row else None


def get_agent_last_loaded_checkpoint(agent_name: str) -> int | None:
    with POOL.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT last_loaded_checkpoint_id FROM memory_agent_load_state WHERE agent_name = %s",
                (agent_name,),
            )
            row = cur.fetchone()
            return row["last_loaded_checkpoint_id"] if row else None


def mark_agent_loaded(agent_name: str, checkpoint_id: int | None) -> None:
    with POOL.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO memory_agent_load_state (agent_name, last_loaded_checkpoint_id, loaded_at)
                VALUES (%s, %s, NOW())
                ON CONFLICT (agent_name)
                DO UPDATE SET last_loaded_checkpoint_id = EXCLUDED.last_loaded_checkpoint_id,
                              loaded_at = NOW()
                """,
                (agent_name, checkpoint_id),
            )
        conn.commit()

def list_recent_checkpoints(limit: int = 20) -> list[dict[str, Any]]:
    with POOL.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    id,
                    trigger_type,
                    trigger_reason,
                    status,
                    started_at,
                    finished_at,
                    delta_meta,
                    counts,
                    error_text
                FROM memory_checkpoints
                ORDER BY id DESC
                LIMIT %s
                """,
                (limit,),
            )
            return list(cur.fetchall())


def latest_checkpoint() -> dict[str, Any] | None:
    with POOL.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    id,
                    trigger_type,
                    trigger_reason,
                    status,
                    started_at,
                    finished_at,
                    delta_meta,
                    counts,
                    error_text
                FROM memory_checkpoints
                ORDER BY id DESC
                LIMIT 1
                """
            )
            row = cur.fetchone()
            return dict(row) if row else None

def ensure_trigger_tables() -> None:
    with POOL.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS memory_triggers (
                    id BIGSERIAL PRIMARY KEY,
                    source_type TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending',
                    priority INTEGER NOT NULL DEFAULT 100,
                    dedupe_key TEXT,
                    payload JSONB NOT NULL DEFAULT '{}'::jsonb,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    started_at TIMESTAMPTZ,
                    finished_at TIMESTAMPTZ,
                    error_text TEXT
                )
                """
            )
            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_memory_triggers_status_priority_created
                ON memory_triggers (status, priority, created_at)
                """
            )
            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_memory_triggers_dedupe_key
                ON memory_triggers (dedupe_key)
                """
            )
        conn.commit()


def enqueue_trigger(
    *,
    source_type: str,
    reason: str,
    payload: dict[str, Any],
    priority: int = 100,
    dedupe_key: str | None = None,
) -> int:
    with POOL.connection() as conn:
        with conn.cursor() as cur:
            if dedupe_key:
                cur.execute(
                    """
                    SELECT id
                    FROM memory_triggers
                    WHERE status IN ('pending', 'processing')
                      AND dedupe_key = %s
                    ORDER BY id DESC
                    LIMIT 1
                    """,
                    (dedupe_key,),
                )
                existing = cur.fetchone()
                if existing:
                    conn.commit()
                    return int(existing["id"])

            cur.execute(
                """
                INSERT INTO memory_triggers (
                    source_type,
                    reason,
                    status,
                    priority,
                    dedupe_key,
                    payload
                )
                VALUES (%s, %s, 'pending', %s, %s, %s::jsonb)
                RETURNING id
                """,
                (
                    source_type,
                    reason,
                    priority,
                    dedupe_key,
                    json.dumps(payload),
                ),
            )
            trigger_id = int(cur.fetchone()["id"])
        conn.commit()
        return trigger_id


def claim_next_trigger() -> dict[str, Any] | None:
    with POOL.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                WITH picked AS (
                    SELECT id
                    FROM memory_triggers
                    WHERE status = 'pending'
                    ORDER BY priority ASC, created_at ASC, id ASC
                    LIMIT 1
                    FOR UPDATE SKIP LOCKED
                )
                UPDATE memory_triggers t
                SET status = 'processing',
                    started_at = NOW()
                FROM picked
                WHERE t.id = picked.id
                RETURNING
                    t.id,
                    t.source_type,
                    t.reason,
                    t.status,
                    t.priority,
                    t.dedupe_key,
                    t.payload,
                    t.created_at,
                    t.started_at,
                    t.finished_at,
                    t.error_text
                """
            )
            row = cur.fetchone()
        conn.commit()
        return dict(row) if row else None


def complete_trigger(trigger_id: int) -> None:
    with POOL.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE memory_triggers
                SET status = 'done',
                    finished_at = NOW(),
                    error_text = NULL
                WHERE id = %s
                """,
                (trigger_id,),
            )
        conn.commit()


def fail_trigger(trigger_id: int, error_text: str) -> None:
    with POOL.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE memory_triggers
                SET status = 'failed',
                    finished_at = NOW(),
                    error_text = %s
                WHERE id = %s
                """,
                (error_text, trigger_id),
            )
        conn.commit()


def mark_stale_processing_triggers(max_age_seconds: int = 300) -> int:
    with POOL.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE memory_triggers
                SET status = 'failed',
                    finished_at = NOW(),
                    error_text = COALESCE(error_text, '') ||
                        CASE
                            WHEN COALESCE(error_text, '') = '' THEN ''
                            ELSE E'\n'
                        END ||
                        %s
                WHERE status = 'processing'
                  AND started_at < NOW() - (%s::text || ' seconds')::interval
                RETURNING id
                """,
                (f"janitor_marked_stale_after_{max_age_seconds}s", str(max_age_seconds)),
            )
            rows = cur.fetchall()
        conn.commit()
        return len(rows)


def list_recent_triggers(limit: int = 20) -> list[dict[str, Any]]:
    with POOL.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    id,
                    source_type,
                    reason,
                    status,
                    priority,
                    dedupe_key,
                    payload,
                    created_at,
                    started_at,
                    finished_at,
                    error_text
                FROM memory_triggers
                ORDER BY id DESC
                LIMIT %s
                """,
                (limit,),
            )
            return [dict(row) for row in cur.fetchall()]


def latest_trigger() -> dict[str, Any] | None:
    with POOL.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    id,
                    source_type,
                    reason,
                    status,
                    priority,
                    dedupe_key,
                    payload,
                    created_at,
                    started_at,
                    finished_at,
                    error_text
                FROM memory_triggers
                ORDER BY id DESC
                LIMIT 1
                """
            )
            row = cur.fetchone()
        conn.commit()
        return dict(row) if row else None

@contextmanager
def advisory_lock():
    with POOL.connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT pg_try_advisory_lock(%s)", (CHECKPOINT_LOCK_KEY,))
            locked = cur.fetchone()["pg_try_advisory_lock"]
            if not locked:
                raise RuntimeError("checkpoint worker lock already held")
        try:
            yield
        finally:
            with conn.cursor() as cur:
                cur.execute("SELECT pg_advisory_unlock(%s)", (CHECKPOINT_LOCK_KEY,))
            conn.commit()
