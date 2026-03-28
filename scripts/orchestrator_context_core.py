from __future__ import annotations

import os
from typing import Any

import psycopg


ROLE_MEMORY_TYPES = {
    "builder": [
        "implementation_pattern",
        "architecture_rule",
        "project_summary",
        "subsystem_summary",
        "feature_summary",
        "guardrail",
    ],
    "reviewer": [
        "review_finding",
        "decision_record",
        "architecture_rule",
        "guardrail",
        "workflow_rule",
    ],
    "tester": [
        "test_outcome",
        "bug_history",
        "validation_pattern",
        "workflow_rule",
        "guardrail",
    ],
    "writeback": [
        "project_summary",
        "subsystem_summary",
        "feature_summary",
    ],
}


def _get_dsn() -> str:
    dsn = os.environ.get("OPENCLAW_MEMORY_DB_DSN") or os.environ.get("DB")
    if not dsn:
        raise RuntimeError("OPENCLAW_MEMORY_DB_DSN or DB is not set")
    return dsn

def fetch_orchestrator_context(
    *,
    project_id: str,
    subproject_id: str | None,
    role: str,
    mode: str,
    work_id: str,
    kind: str,
    tags: list[str] | None = None,
    limit: int = 6,
) -> dict:
    """
    Step 8 initial real implementation:
    - canonical Postgres-first retrieval
    - role-aware type filtering
    - compact summaries + memory refs
    - safe fallback if exact schema differs
    Later you can enrich this behind the same function with Qdrant/Neo4j/reranker.
    """
    tags = tags or []
    preferred_types = ROLE_MEMORY_TYPES.get(role, ["project_summary", "architecture_rule", "guardrail"])
    limit = max(1, min(limit, 12))

    scopes = [project_id, "global"]
    if subproject_id:
        scopes.insert(1, subproject_id)

    with psycopg.connect(_get_dsn()) as conn:
        with conn.cursor() as cur:
            work_row = None

            cur.execute(
                """
                SELECT text
                FROM memory_items
                WHERE scope = %s
                  AND memory_type = 'project_summary'
                  AND lifecycle_state IN ('reinforced', 'durable')
                ORDER BY last_confirmed DESC NULLS LAST, first_seen DESC NULLS LAST
                LIMIT 1
                """,
                (project_id,),
            )
            project_row = cur.fetchone()

            subproject_row = None
            if subproject_id:
                cur.execute(
                    """
                    SELECT text
                    FROM memory_items
                    WHERE scope = %s
                      AND memory_type IN ('subsystem_summary', 'feature_summary')
                      AND lifecycle_state IN ('reinforced', 'durable')
                    ORDER BY last_confirmed DESC NULLS LAST, first_seen DESC NULLS LAST
                    LIMIT 1
                    """,
                    (subproject_id,),
                )
                subproject_row = cur.fetchone()

            cur.execute(
                """
                SELECT id, memory_type, lifecycle_state, text
                FROM memory_items
                WHERE scope = ANY(%s)
                  AND memory_type = ANY(%s)
                  AND lifecycle_state IN ('reinforced', 'durable')
                ORDER BY last_confirmed DESC NULLS LAST, first_seen DESC NULLS LAST
                LIMIT %s
                """,
                (scopes, preferred_types, limit),
            )
            rows = cur.fetchall()

    return {
        "ok": True,
        "source": "postgres_canonical_initial",
        "role": role,
        "mode": mode,
        "summaries": {
            "work_summary": work_row[0] if work_row else None,
            "project_summary": project_row[0] if project_row else None,
            "subproject_summary": subproject_row[0] if subproject_row else None,
            "feature_summary": None,
        },
        "memory_refs": [
            {
                "memory_id": row[0],
                "memory_type": row[1],
                "lifecycle_state": row[2],
                "content": row[3],
                "score": None,
            }
            for row in rows
        ],
        "meta": {
            "project_id": project_id,
            "subproject_id": subproject_id,
            "kind": kind,
            "tags": tags,
            "limit": limit,
        },
    }
