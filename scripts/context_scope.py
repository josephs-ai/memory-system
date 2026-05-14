"""
Scope resolution for memory context — determines which scoped memories
are visible in the current execution context.
"""
from __future__ import annotations

import hashlib
import json
import re
from typing import Any

PIPELINE_RE = re.compile(r"\b(?:in|within|for)\s+(?:this\s+)?pipeline\s+([A-Za-z0-9_.:-]+)", re.IGNORECASE)
WORKFLOW_RE = re.compile(r"\b(?:in|within|for)\s+(?:this\s+)?workflow\s+([A-Za-z0-9_.:-]+)", re.IGNORECASE)
SUBPROJECT_RE = re.compile(r"\bsubproject\s+([A-Za-z0-9_.:-]+)", re.IGNORECASE)
PROJECT_RE = re.compile(r"\bproject\s+([A-Za-z0-9_.:-]+)", re.IGNORECASE)


def _slug(value: str | None, fallback: str) -> str:
    raw = str(value or "").strip().lower()
    if not raw:
        return fallback
    cleaned = []
    for ch in raw:
        if ch.isalnum() or ch in {"_", "-", ".", ":"}:
            cleaned.append(ch)
        else:
            cleaned.append("-")
    text = "".join(cleaned).strip("-")
    while "--" in text:
        text = text.replace("--", "-")
    return text or fallback



def _extract_scope_markers(text: str) -> dict[str, str]:
    markers: dict[str, str] = {}
    for key, pattern in (
        ("pipeline_id", PIPELINE_RE),
        ("workflow_id", WORKFLOW_RE),
        ("subproject_id", SUBPROJECT_RE),
        ("project_id", PROJECT_RE),
    ):
        match = pattern.search(text or "")
        if match:
            markers[key] = _slug(match.group(1), key)
    return markers



def _scope_type(payload: dict[str, Any]) -> str:
    if payload.get("pipeline_id"):
        return "pipeline"
    if payload.get("workflow_id"):
        return "workflow"
    if payload.get("subproject_id"):
        return "subproject"
    if payload.get("project_id"):
        return "project"
    base_scope = str(payload.get("scope") or "").strip().lower()
    if base_scope in {"global", "stable"}:
        return "global"
    if base_scope:
        return base_scope
    return "global"



def _inheritance_policy(scope_type: str) -> str:
    if scope_type in {"pipeline", "workflow"}:
        return "local_only"
    if scope_type in {"subproject", "project"}:
        return "inherits_downward"
    if scope_type == "global":
        return "promoted_global"
    return "local_only"



def build_context_scope(item: dict[str, Any]) -> dict[str, Any]:
    text = str(item.get("text") or "")
    markers = _extract_scope_markers(text)

    payload = {
        "project_id": _slug(item.get("project_id") or markers.get("project_id"), "") or None,
        "subproject_id": _slug(item.get("subproject_id") or markers.get("subproject_id"), "") or None,
        "workflow_id": _slug(item.get("workflow_id") or markers.get("workflow_id"), "") or None,
        "pipeline_id": _slug(item.get("pipeline_id") or markers.get("pipeline_id"), "") or None,
        "scope": str(item.get("scope") or "stable").strip().lower() or "stable",
        "source_session": str(item.get("source_session") or "").strip() or None,
    }
    scope_type = _scope_type(payload)
    payload["scope_type"] = scope_type
    payload["inheritance_policy"] = str(item.get("inheritance_policy") or _inheritance_policy(scope_type))
    payload["scope_confidence"] = float(item.get("scope_confidence") or (0.96 if markers else 0.75 if any(payload.get(k) for k in ("project_id", "subproject_id", "workflow_id", "pipeline_id")) else 0.35))

    identity_parts = {
        "scope_type": scope_type,
        "project_id": payload.get("project_id"),
        "subproject_id": payload.get("subproject_id"),
        "workflow_id": payload.get("workflow_id"),
        "pipeline_id": payload.get("pipeline_id"),
        "scope": payload.get("scope"),
        "source_session": payload.get("source_session") if scope_type == "session" else None,
    }
    digest = hashlib.sha1(json.dumps(identity_parts, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    payload["scope_id"] = f"scope-{digest}"
    return payload
