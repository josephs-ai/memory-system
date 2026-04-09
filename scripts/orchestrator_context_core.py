from __future__ import annotations

import os
from typing import Any

import psycopg


CONTRACT_VERSION = "2026-04-09.memory-runtime-truth.v1"
GLOBAL_FALLBACK_SCOPES = ("global", "archive")
ROLE_RETRIEVAL_CONTRACTS: dict[str, dict[str, Any]] = {
    "builder": {
        "preferred_types": [
            "implementation_pattern",
            "architecture_rule",
            "project_summary",
            "subsystem_summary",
            "feature_summary",
            "guardrail",
            "decision",
            "decision_record",
        ],
        "eligible_categories": [
            "implementation_pattern",
            "architecture_rule",
            "project_summary",
            "subproject_summary",
            "guardrail",
            "decision",
            "recent_change",
        ],
        "eligible_types": [
            "implementation_pattern",
            "architecture_rule",
            "project_summary",
            "subsystem_summary",
            "feature_summary",
            "guardrail",
            "decision",
            "decision_record",
        ],
    },
    "reviewer": {
        "preferred_types": [
            "review_finding",
            "decision_record",
            "architecture_rule",
            "guardrail",
            "workflow_rule",
            "bug_history",
            "decision",
            "feature_summary",
        ],
        "eligible_categories": [
            "review_finding",
            "architecture_rule",
            "guardrail",
            "workflow_rule",
            "bug_history",
            "decision",
            "recent_change",
        ],
        "eligible_types": [
            "review_finding",
            "decision_record",
            "architecture_rule",
            "guardrail",
            "workflow_rule",
            "bug_history",
            "decision",
            "feature_summary",
            "subsystem_summary",
        ],
    },
    "tester": {
        "preferred_types": [
            "test_outcome",
            "bug_history",
            "validation_pattern",
            "workflow_rule",
            "guardrail",
            "decision",
            "review_finding",
        ],
        "eligible_categories": [
            "test_outcome",
            "bug_history",
            "validation_pattern",
            "workflow_rule",
            "guardrail",
            "recent_change",
        ],
        "eligible_types": [
            "test_outcome",
            "bug_history",
            "validation_pattern",
            "workflow_rule",
            "guardrail",
            "decision",
            "review_finding",
        ],
    },
    "writeback": {
        "preferred_types": [
            "project_summary",
            "subsystem_summary",
            "feature_summary",
            "decision",
            "test_outcome",
            "bug_history",
        ],
        "eligible_categories": [
            "project_summary",
            "subproject_summary",
            "decision",
            "test_outcome",
            "bug_history",
            "recent_change",
        ],
        "eligible_types": [
            "project_summary",
            "subsystem_summary",
            "feature_summary",
            "decision",
            "decision_record",
            "test_outcome",
            "bug_history",
            "review_finding",
        ],
    },
}


def _contract_for_role(role: str) -> dict[str, Any]:
    return ROLE_RETRIEVAL_CONTRACTS.get(
        role,
        {
            "preferred_types": ["project_summary", "architecture_rule", "guardrail", "decision"],
            "eligible_categories": ["project_summary", "architecture_rule", "guardrail", "decision", "recent_change"],
            "eligible_types": ["project_summary", "architecture_rule", "guardrail", "decision", "decision_record"],
        },
    )


def _get_dsn() -> str:
    dsn = os.environ.get("OPENCLAW_MEMORY_DB_DSN") or os.environ.get("DB")
    if not dsn:
        raise RuntimeError("OPENCLAW_MEMORY_DB_DSN or DB is not set")
    return dsn


def _scope_plan(project_id: str, subproject_id: str | None) -> tuple[list[str], list[str]]:
    primary: list[str] = []
    if subproject_id:
        primary.append(subproject_id)
    if project_id:
        primary.append(project_id)
    fallback = [scope for scope in GLOBAL_FALLBACK_SCOPES if scope not in primary]
    return primary, fallback


def _scope_tier(scope: str | None, project_id: str, subproject_id: str | None) -> str:
    scope = str(scope or "")
    if subproject_id and scope == subproject_id:
        return "subproject"
    if project_id and scope == project_id:
        return "project"
    if scope in GLOBAL_FALLBACK_SCOPES:
        return "fallback_global"
    return "other"


def _memory_category(memory_type: str | None) -> str:
    memory_type = str(memory_type or "")
    if memory_type in {"project_summary"}:
        return "project_summary"
    if memory_type in {"subsystem_summary", "feature_summary"}:
        return "subproject_summary"
    if memory_type in {"architecture_rule"}:
        return "architecture_rule"
    if memory_type in {"implementation_pattern"}:
        return "implementation_pattern"
    if memory_type in {"guardrail"}:
        return "guardrail"
    if memory_type in {"workflow_rule"}:
        return "workflow_rule"
    if memory_type in {"bug_history"}:
        return "bug_history"
    if memory_type in {"test_outcome"}:
        return "test_outcome"
    if memory_type in {"validation_pattern"}:
        return "validation_pattern"
    if memory_type in {"review_finding"}:
        return "review_finding"
    if memory_type in {"decision", "decision_record"}:
        return "decision"
    return "other"


def _summarize_scope(rows: list[tuple], desired_types: tuple[str, ...], project_id: str, subproject_id: str | None) -> str | None:
    for row in rows:
        _id, memory_type, _lifecycle_state, text, _tags, _updated_at, _created_at, _work_id, scope = row
        if memory_type in desired_types and _scope_tier(scope, project_id, subproject_id) != "fallback_global":
            return text
    return None


def _fetch_rows(cur, scopes: list[str], eligible_types: list[str], limit: int) -> list[tuple]:
    if not scopes:
        return []

    cur.execute(
        """
        SELECT id, memory_type, lifecycle_state, text, tags, updated_at, created_at, work_id, scope
        FROM memory_items
        WHERE scope = ANY(%s)
          AND memory_type = ANY(%s)
          AND lifecycle_state IN ('candidate', 'reinforced', 'durable')
          AND (retrieval_eligibility IS NULL OR retrieval_eligibility <> 'excluded')
        ORDER BY updated_at DESC NULLS LAST, created_at DESC NULLS LAST
        LIMIT %s
        """,
        (scopes, eligible_types, limit),
    )
    return cur.fetchall()


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
    contract = _contract_for_role(role)
    preferred_types = list(contract["preferred_types"])
    eligible_types = list(contract["eligible_types"])
    primary_scopes, fallback_scopes = _scope_plan(project_id, subproject_id)
    limit = max(1, min(limit, 12))

    with psycopg.connect(_get_dsn()) as conn:
        with conn.cursor() as cur:
            primary_rows = _fetch_rows(cur, primary_scopes, eligible_types, max(limit * 4, limit))
            fallback_rows = []
            if len(primary_rows) < limit and fallback_scopes:
                fallback_rows = _fetch_rows(cur, fallback_scopes, eligible_types, max(limit * 4, limit))

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
            {"subproject": 0, "project": 1, "fallback_global": 2}.get(_scope_tier(row[8], project_id, subproject_id), 3),
            preferred_types.index(row[1]) if row[1] in preferred_types else len(preferred_types),
            str(row[0]),
        )
    )
    selected_rows = deduped_rows[:limit]

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
                "category": _memory_category(row[1]),
                "lifecycle_state": row[2],
                "content": row[3],
                "tags": row[4] or [],
                "score": None,
                "source_work_id": row[7],
                "scope": row[8],
                "scope_tier": _scope_tier(row[8], project_id, subproject_id),
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
            "primary_scopes": primary_scopes,
            "fallback_scopes": fallback_scopes,
            "fallback_used": bool(fallback_rows),
            "eligible_categories": list(contract["eligible_categories"]),
            "eligible_types": eligible_types,
            "primary_candidate_count": len(primary_rows),
            "fallback_candidate_count": len(fallback_rows),
        },
    }
