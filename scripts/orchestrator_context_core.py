"""
Core context assembly for the orchestrator — retrieves and formats
memory context for orchestrator decision-making.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import psycopg

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from orchestrator.memory_contract import (  # noqa: E402
    CONTRACT_VERSION,
    UNSCOPED_FALLBACK_SCOPE,
    contract_for_role,
    eligible_categories_for_role,
    eligible_lifecycle_states_for_role,
    eligible_types_for_role,
    memory_category,
    preferred_types_for_role,
    scope_plan,
    scope_rank,
    scope_tier,
)


def _get_dsn() -> str:
    # Honor both canonical env names (plus the legacy `DB` alias). Previously
    # this read only OPENCLAW_MEMORY_DB_DSN, so a deployment that set only
    # OPENCLAW_MEMORY_DSN would raise here even though the rest of the system
    # was configured correctly. Preserve the strict no-silent-default behavior:
    # this path REQUIRES an explicit DSN and must not fall back to a bare
    # default, so we resolve the names directly rather than via resolve_db_dsn().
    dsn = (
        os.environ.get("OPENCLAW_MEMORY_DSN")
        or os.environ.get("OPENCLAW_MEMORY_DB_DSN")
        or os.environ.get("DB")
    )
    if not dsn:
        raise RuntimeError(
            "No DSN configured: set OPENCLAW_MEMORY_DSN, OPENCLAW_MEMORY_DB_DSN, or DB"
        )
    return dsn


def _summarize_scope(rows: list[tuple], desired_types: tuple[str, ...], project_id: str, subproject_id: str | None) -> str | None:
    for row in rows:
        _id, memory_type, _lifecycle_state, text, _tags, _updated_at, _created_at, _work_id, scope = row
        if memory_type in desired_types and scope_tier(scope, project_id, subproject_id) not in {"fallback_global", "unscoped_fallback"}:
            return text
    return None


def _execute_scope_query(cur, scopes: list[str], eligible_types: list[str], eligible_lifecycle_states: list[str], limit: int) -> tuple[list[tuple], dict]:
    if not scopes:
        return [], {"query_mode": "skipped", "scope_column_present": None, "scope_filter_applied": False}

    scoped_query = """
        SELECT id, memory_type, lifecycle_state, text, tags, updated_at, created_at, work_id, COALESCE(scope, '') AS scope
        FROM memory_items
        WHERE scope = ANY(%s)
          AND memory_type = ANY(%s)
          AND lifecycle_state = ANY(%s)
          AND (retrieval_eligibility IS NULL OR retrieval_eligibility <> 'excluded')
        ORDER BY updated_at DESC NULLS LAST, created_at DESC NULLS LAST
        LIMIT %s
    """
    unscoped_query = """
        SELECT id, memory_type, lifecycle_state, text, tags, updated_at, created_at, work_id, %s AS scope
        FROM memory_items
        WHERE memory_type = ANY(%s)
          AND lifecycle_state = ANY(%s)
          AND (retrieval_eligibility IS NULL OR retrieval_eligibility <> 'excluded')
        ORDER BY updated_at DESC NULLS LAST, created_at DESC NULLS LAST
        LIMIT %s
    """

    try:
        cur.execute(scoped_query, (scopes, eligible_types, eligible_lifecycle_states, limit))
        return cur.fetchall(), {
            "query_mode": "scoped",
            "scope_column_present": True,
            "scope_filter_applied": True,
            "requested_scopes": list(scopes),
        }
    except Exception as exc:
        if "scope" not in str(exc).lower():
            raise

    cur.execute(unscoped_query, (UNSCOPED_FALLBACK_SCOPE, eligible_types, eligible_lifecycle_states, limit))
    return cur.fetchall(), {
        "query_mode": "unscoped_fallback",
        "scope_column_present": False,
        "scope_filter_applied": False,
        "requested_scopes": list(scopes),
    }


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
    tags = tags or []
    _contract = contract_for_role(role)
    preferred_types = preferred_types_for_role(role)
    eligible_types = eligible_types_for_role(role)
    eligible_lifecycle_states = eligible_lifecycle_states_for_role(role)
    primary_scopes, fallback_scopes = scope_plan(project_id, subproject_id)
    limit = max(1, min(limit, 12))
    candidate_limit = max(limit * 4, limit)

    with psycopg.connect(_get_dsn()) as conn:
        with conn.cursor() as cur:
            primary_rows, primary_query_meta = _execute_scope_query(cur, primary_scopes, eligible_types, eligible_lifecycle_states, candidate_limit)
            fallback_rows = []
            fallback_query_meta = {"query_mode": "skipped", "scope_column_present": primary_query_meta.get("scope_column_present")}
            if len(primary_rows) < limit and fallback_scopes and primary_query_meta.get("query_mode") != "unscoped_fallback":
                fallback_rows, fallback_query_meta = _execute_scope_query(cur, fallback_scopes, eligible_types, eligible_lifecycle_states, candidate_limit)

    deduped_rows = []
    seen_ids: set[str] = set()
    for row in [*primary_rows, *fallback_rows]:
        memory_id = str(row[0])
        if memory_id in seen_ids:
            continue
        seen_ids.add(memory_id)
        deduped_rows.append(row)

    deduped_rows.sort(
        key=lambda row: (
            scope_rank(row[8], project_id, subproject_id),
            preferred_types.index(row[1]) if row[1] in preferred_types else len(preferred_types),
            str(row[0]),
        )
    )
    selected_rows = deduped_rows[:limit]

    scope_contract_status = "scoped"
    if primary_query_meta.get("query_mode") == "unscoped_fallback":
        scope_contract_status = "unscoped_fallback_only"
    elif fallback_query_meta.get("query_mode") == "unscoped_fallback":
        scope_contract_status = "scoped_then_unscoped_fallback"

    fallback_source = None
    if fallback_rows:
        fallback_source = fallback_query_meta.get("query_mode")

    return {
        "ok": True,
        "source": "postgres_canonical_scoped_contract",
        "contract_version": CONTRACT_VERSION,
        "role": role,
        "mode": mode,
        "summaries": {
            "work_summary": None,
            "project_summary": _summarize_scope(selected_rows, ("project_summary",), project_id, subproject_id),
            "subproject_summary": _summarize_scope(selected_rows, ("subsystem_summary", "feature_summary"), project_id, subproject_id),
            "feature_summary": None,
        },
        "memory_refs": [
            {
                "memory_id": row[0],
                "memory_type": row[1],
                "category": memory_category(row[1]),
                "lifecycle_state": row[2],
                "content": row[3],
                "tags": row[4] or [],
                "score": None,
                "source_work_id": row[7],
                "scope": row[8],
                "scope_tier": scope_tier(row[8], project_id, subproject_id),
                "retrieval_contract_role": role,
            }
            for row in selected_rows
        ],
        "meta": {
            "project_id": project_id,
            "subproject_id": subproject_id,
            "kind": kind,
            "tags": tags,
            "limit": limit,
            "candidate_limit": candidate_limit,
            "primary_scopes": primary_scopes,
            "fallback_scopes": fallback_scopes,
            "fallback_used": bool(fallback_rows),
            "fallback_source": fallback_source,
            "scope_contract_status": scope_contract_status,
            "scope_column_present": primary_query_meta.get("scope_column_present") if primary_query_meta.get("scope_column_present") is not None else fallback_query_meta.get("scope_column_present"),
            "primary_query_mode": primary_query_meta.get("query_mode"),
            "fallback_query_mode": fallback_query_meta.get("query_mode"),
            "eligible_categories": eligible_categories_for_role(role),
            "eligible_types": eligible_types,
            "eligible_lifecycle_states": eligible_lifecycle_states,
            "primary_candidate_count": len(primary_rows),
            "fallback_candidate_count": len(fallback_rows),
        },
    }
