"""
Standard agent API interface for memory operations — provides a clean
API surface for agent-to-memory-system interaction.
"""
from __future__ import annotations

import copy
import hashlib
import time
from datetime import datetime
from enum import Enum
from typing import Any, Literal

from pydantic import AliasChoices, BaseModel, Field, field_validator


DEFAULT_ALLOWED_MEMORY_TYPES = [
    "decision",
    "decision_record",
    "implementation_pattern",
    "architecture_rule",
    "guardrail",
    "workflow_rule",
    "bug_history",
    "failure_mode",
    "learned_fix",
    "feature_summary",
]
DEFAULT_ALLOWED_LIFECYCLE_STATES = ["candidate", "reinforced", "durable"]
DEFAULT_MAX_CONTEXT_ITEMS = 8
DEFAULT_MAX_CONTEXT_TOKENS = 1200
NORMAL_AGENT_PACKET_CONTRACT_VERSION = "2026-04-11.normal-agent.packet.v1"
NORMAL_MODE_RETRIEVAL_POLICY_VERSION = "2026-04-11.normal-mode.retrieval.v1"
NORMAL_MODE_FRESHNESS_POLICY_VERSION = "2026-04-11.normal-mode.freshness.v1"
NORMAL_MODE_CONFLICT_POLICY_VERSION = "2026-04-11.normal-mode.conflict.v1"
NORMAL_AGENT_SHAPING_POLICY_VERSION = "2026-04-11.normal-agent.shaping.v1"
NORMAL_AGENT_SECTION_CONTRACT_VERSION = "2026-04-11.normal-agent.sections.v1"
NORMAL_AGENT_BOUNDARY_POLICY_VERSION = "2026-04-11.normal-agent.boundary.v1"
PACKET_CACHE_SCHEMA_VERSION = 1
NORMAL_AGENT_PACKET_CACHE: dict[str, dict[str, Any]] = {}
PACKET_CACHE_TTL_SECONDS = {
    "fresh_only": 300,
    "mixed": 900,
    "durable_only": 3600,
}


class RetrievalBias(str, Enum):
    fresh_only = "fresh_only"
    durable_only = "durable_only"
    mixed = "mixed"


class RequesterIdentity(BaseModel):
    requester_id: str | None = None
    requester_role: str = "normal_agent"

    @field_validator("requester_role")
    @classmethod
    def _validate_requester_role(cls, value: str) -> str:
        item = str(value or "").strip()
        if not item:
            raise ValueError("requester_role must be non-empty")
        return item


class NormalAgentRequest(BaseModel):
    query: str = Field(min_length=1, validation_alias=AliasChoices("query", "task"))
    agent_mode: Literal["normal"] = "normal"
    project_id: str = Field(min_length=1)
    subproject_id: str | None = None
    domain_tags: list[str] = Field(default_factory=list)
    allowed_memory_types: list[str] = Field(default_factory=lambda: list(DEFAULT_ALLOWED_MEMORY_TYPES))
    allowed_lifecycle_states: list[str] = Field(default_factory=lambda: list(DEFAULT_ALLOWED_LIFECYCLE_STATES))
    max_context_items: int = Field(default=DEFAULT_MAX_CONTEXT_ITEMS, ge=1, le=50)
    max_context_tokens: int = Field(default=DEFAULT_MAX_CONTEXT_TOKENS, ge=128, le=16000)
    need_decisions: bool = True
    need_patterns: bool = True
    need_bug_history: bool = True
    need_recent_session_context: bool = False
    requester: RequesterIdentity = Field(default_factory=RequesterIdentity)
    retrieval_bias: RetrievalBias = RetrievalBias.mixed

    @field_validator("query", "project_id")
    @classmethod
    def _validate_required_strings(cls, value: str) -> str:
        item = str(value or "").strip()
        if not item:
            raise ValueError("field must be non-empty")
        return item

    @field_validator("subproject_id")
    @classmethod
    def _normalize_subproject_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        item = str(value).strip()
        return item or None

    @field_validator("domain_tags", "allowed_memory_types", "allowed_lifecycle_states")
    @classmethod
    def _dedupe_string_lists(cls, value: list[str]) -> list[str]:
        out: list[str] = []
        seen: set[str] = set()
        for item in value or []:
            normalized = str(item or "").strip()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            out.append(normalized)
        return out


class PacketSourceRef(BaseModel):
    path: str
    source_type: str | None = None
    memory_type: str | None = None
    lifecycle_state: str | None = None
    score: float | None = None


class NormalAgentPacket(BaseModel):
    packet_id: str
    agent_mode: Literal["normal"] = "normal"
    status: Literal["fulfilled", "partial", "insufficient_context"]
    task_context: dict[str, Any]
    must_follow_rules: list[str] = Field(default_factory=list)
    relevant_decisions: list[str] = Field(default_factory=list)
    implementation_patterns: list[str] = Field(default_factory=list)
    recent_history: list[str] = Field(default_factory=list)
    known_failures_or_bug_history: list[str] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    recommended_next_assumptions: list[str] = Field(default_factory=list)
    conflict_notes: list[str] = Field(default_factory=list)
    source_refs: list[PacketSourceRef] = Field(default_factory=list)
    missing_context: list[str] = Field(default_factory=list)
    meta: dict[str, Any] = Field(default_factory=dict)


class NormalAgentPacketResponse(BaseModel):
    ok: bool
    request: NormalAgentRequest
    packet: NormalAgentPacket
    meta: dict[str, Any]


NORMAL_AGENT_REQUEST_EXAMPLE: dict[str, Any] = {
    "query": "How should I update the stealth suspicion decay behavior without breaking alert escalation?",
    "agent_mode": "normal",
    "project_id": "mygame_prototype",
    "subproject_id": "enemy_ai",
    "domain_tags": ["combat", "enemy_ai", "stealth"],
    "allowed_memory_types": list(DEFAULT_ALLOWED_MEMORY_TYPES),
    "allowed_lifecycle_states": list(DEFAULT_ALLOWED_LIFECYCLE_STATES),
    "max_context_items": DEFAULT_MAX_CONTEXT_ITEMS,
    "max_context_tokens": DEFAULT_MAX_CONTEXT_TOKENS,
    "need_decisions": True,
    "need_patterns": True,
    "need_bug_history": True,
    "need_recent_session_context": False,
    "requester": {"requester_id": "normal-agent-demo", "requester_role": "normal_agent"},
    "retrieval_bias": "mixed",
}


def _approx_tokens(text: str) -> int:
    return max(1, len((text or "").split()))


def _stable_packet_id(req: NormalAgentRequest) -> str:
    raw = "|".join(
        [
            req.project_id,
            req.subproject_id or "",
            req.query,
            req.requester.requester_role,
            req.retrieval_bias.value,
            ",".join(req.domain_tags),
        ]
    )
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]
    return f"normal-agent::{req.project_id}::{digest}"


def _normalize_text(row: dict[str, Any]) -> str:
    return str(row.get("text") or "").strip()


DISALLOWED_NORMAL_AGENT_PATH_PREFIXES = (
    ".memory-index/scripts/",
    ".memory-index/control_panel/",
    "orchestrator/",
    "orchestrator_runtime/",
    "openclaw/control/",
)
INTERNAL_SOURCE_MARKERS = DISALLOWED_NORMAL_AGENT_PATH_PREFIXES



def _is_allowed_normal_agent_source(row: dict[str, Any]) -> bool:
    if bool(row.get("explicitly_attached") or row.get("attachment_authorized")):
        return True

    path = str(row.get("path") or "").strip().replace("\\", "/")
    source_type = str(row.get("source_type") or "").strip().lower()
    if source_type in {"db", "canonical", "canonical_qdrant", "graph", "qdrant"} and path in {"", "db:memory_items", "qdrant:memory_items", "neo4j:memory_graph"}:
        return True
    if not path:
        return True
    return not path.startswith(DISALLOWED_NORMAL_AGENT_PATH_PREFIXES)



def _matches_filters(row: dict[str, Any], req: NormalAgentRequest) -> bool:
    if not _is_allowed_normal_agent_source(row):
        return False

    memory_type = str(row.get("memory_type") or "").strip()
    lifecycle_state = str(row.get("lifecycle_state") or "").strip()

    if req.allowed_memory_types and memory_type and memory_type not in req.allowed_memory_types:
        return False
    if req.allowed_memory_types and not memory_type:
        return False

    if req.allowed_lifecycle_states and lifecycle_state and lifecycle_state not in req.allowed_lifecycle_states:
        return False
    if req.allowed_lifecycle_states and not lifecycle_state:
        return False

    if req.retrieval_bias == RetrievalBias.fresh_only and lifecycle_state == "durable":
        return False
    if req.retrieval_bias == RetrievalBias.durable_only and lifecycle_state != "durable":
        return False
    return True


def _classify_sections(req: NormalAgentRequest, row: dict[str, Any]) -> list[str]:
    memory_type = str(row.get("memory_type") or "").strip().lower()
    text = _normalize_text(row).lower()
    sections: list[str] = []

    if memory_type in {"guardrail", "workflow_rule", "architecture_rule", "rule"} or "must " in text:
        sections.append("must_follow_rules")
    if req.need_decisions and memory_type in {"decision", "decision_record"}:
        sections.append("relevant_decisions")
    if req.need_patterns and memory_type in {"implementation_pattern", "architecture_rule", "feature_summary", "validation_pattern"}:
        sections.append("implementation_patterns")
    if req.need_recent_session_context and memory_type in {"session_summary", "recent_session_context", "feature_summary"}:
        sections.append("recent_history")
    if req.need_bug_history and memory_type in {"bug_history", "failure_mode", "learned_fix", "test_outcome"}:
        sections.append("known_failures_or_bug_history")
    if memory_type in {"open_question", "review_finding"} or "?" in text:
        sections.append("open_questions")
    return sections


SECTION_PRIORITY = {
    "must_follow_rules": 0,
    "relevant_decisions": 1,
    "implementation_patterns": 2,
    "known_failures_or_bug_history": 3,
    "recent_history": 4,
    "open_questions": 5,
    "recommended_next_assumptions": 6,
}


def _push_unique(target: list[str], text: str) -> None:
    item = str(text or "").strip()
    if item and item not in target:
        target.append(item)


def _dedupe_key(text: str) -> str:
    value = str(text or "").lower().strip()
    for prefix in (
        "decision:",
        "rule:",
        "implementation pattern:",
        "bug history:",
        "must ",
        "assume this remains relevant unless fresher evidence contradicts it: ",
    ):
        if value.startswith(prefix):
            value = value[len(prefix):].strip()
    normalized = "".join(ch if ch.isalnum() or ch.isspace() else " " for ch in value)
    return " ".join(normalized.split())


def _is_near_duplicate(text: str, accepted_keys: list[str]) -> bool:
    key = _dedupe_key(text)
    if not key:
        return True
    for existing in accepted_keys:
        if key == existing:
            return True
        if key in existing or existing in key:
            shorter = min(len(key), len(existing))
            if shorter >= 24:
                return True
    return False


def _section_priority(sections: list[str]) -> int:
    return min(SECTION_PRIORITY.get(section, 999) for section in sections)


LIFECYCLE_PRECEDENCE = {"durable": 3, "reinforced": 2, "candidate": 1}
MEMORY_TYPE_PRECEDENCE = {
    "architecture_rule": 5,
    "rule": 5,
    "guardrail": 5,
    "workflow_rule": 5,
    "decision": 4,
    "decision_record": 4,
    "implementation_pattern": 3,
    "validation_pattern": 3,
    "feature_summary": 2,
    "bug_history": 2,
    "failure_mode": 2,
    "learned_fix": 2,
    "test_outcome": 2,
    "session_summary": 1,
    "recent_session_context": 1,
    "open_question": 1,
    "review_finding": 1,
}

CONFLICT_PREFIXES = (
    "decision:",
    "rule:",
    "implementation pattern:",
    "bug history:",
    "open question:",
    "note:",
)
NEGATIVE_CUES = (
    "must not ",
    "should not ",
    "do not ",
    "don't ",
    "never ",
    "avoid ",
    "disable ",
    "remove ",
    "forbid ",
    "disallow ",
    "without ",
)
POSITIVE_CUES = (
    "must ",
    "should ",
    "keep ",
    "preserve ",
    "use ",
    "enable ",
    "allow ",
    "include ",
    "require ",
)
CONFLICT_STOPWORDS = {
    "a",
    "an",
    "the",
    "this",
    "that",
    "these",
    "those",
    "is",
    "are",
    "be",
    "to",
    "for",
    "of",
    "and",
    "or",
    "with",
    "when",
    "while",
    "during",
    "under",
    "within",
    "outside",
    "inside",
    "into",
    "from",
    "by",
    "on",
    "in",
    "it",
    "its",
    "remain",
    "remains",
}


def _parse_timestamp(value: Any) -> float:
    raw = str(value or "").strip()
    if not raw:
        return 0.0
    try:
        normalized = raw.replace("Z", "+00:00")
        return datetime.fromisoformat(normalized).timestamp()
    except ValueError:
        return 0.0



def _scope_precedence(req: NormalAgentRequest, row: dict[str, Any]) -> int:
    row_project_id = str(row.get("project_id") or "").strip()
    row_subproject_id = str(row.get("subproject_id") or "").strip()
    scope = str(row.get("scope") or "").strip().lower()
    path = str(row.get("path") or "").strip().lower().replace("\\", "/")
    project_prefix = f"memory/projects/{req.project_id.lower()}/"

    if req.subproject_id and row_subproject_id == req.subproject_id:
        return 3
    if req.subproject_id and req.subproject_id.lower() in scope:
        return 3
    if row_project_id == req.project_id:
        return 2
    if req.project_id.lower() in scope:
        return 2
    if path.startswith(project_prefix):
        return 2
    if scope in {"global", "stable", "fallback_global", "unscoped_fallback"}:
        return 1
    return 1 if scope else 0



def _memory_type_precedence(row: dict[str, Any]) -> int:
    memory_type = str(row.get("memory_type") or "").strip().lower()
    return MEMORY_TYPE_PRECEDENCE.get(memory_type, 0)



def _row_scope_label(req: NormalAgentRequest, row: dict[str, Any]) -> str:
    row_project_id = str(row.get("project_id") or "").strip()
    row_subproject_id = str(row.get("subproject_id") or "").strip()
    scope = str(row.get("scope") or "").strip().lower()
    path = str(row.get("path") or "").strip().lower().replace("\\", "/")

    if req.subproject_id and (row_subproject_id == req.subproject_id or req.subproject_id.lower() in scope):
        return "subproject"
    if scope in {"session", "work"}:
        return "session"
    if scope == "archive" or "/archive" in path or "/history" in path:
        return "archive_or_history"
    if scope == "global" or "/memory/global/" in path:
        return "global"
    if row_project_id == req.project_id or req.project_id.lower() in scope or f"memory/projects/{req.project_id.lower()}/" in path:
        return "project"
    return "unscoped"



def _retrieval_tier(req: NormalAgentRequest, row: dict[str, Any]) -> int:
    scope_label = _row_scope_label(req, row)
    lifecycle = str(row.get("lifecycle_state") or row.get("lifecycle") or "").strip().lower()
    memory_type = str(row.get("memory_type") or "").strip().lower()

    if scope_label == "subproject" and lifecycle == "durable":
        return 0
    if scope_label == "project" and lifecycle == "durable":
        return 1
    if lifecycle in {"candidate", "reinforced"} or scope_label == "session" or memory_type in {"session_summary", "recent_session_context"}:
        return 2
    if scope_label == "archive_or_history":
        return 3
    if scope_label == "global":
        return 4
    return 5



def _memory_type_weight(row: dict[str, Any]) -> int:
    memory_type = str(row.get("memory_type") or "").strip().lower()
    if memory_type in {"architecture_rule", "guardrail", "rule"}:
        return 0
    if memory_type in {"decision", "decision_record"}:
        return 1
    if memory_type in {"bug_history", "failure_mode", "learned_fix", "test_outcome"}:
        return 2
    if memory_type in {"workflow_rule"}:
        return 3
    if memory_type in {"implementation_pattern", "validation_pattern"}:
        return 4
    if memory_type in {"feature_summary", "session_summary", "recent_session_context"}:
        return 5
    return 6



def _retrieval_policy_tuple(req: NormalAgentRequest, row: dict[str, Any]) -> tuple[int, int, float, float, str]:
    return (
        _retrieval_tier(req, row),
        _memory_type_weight(row),
        -_parse_timestamp(row.get("updated_at") or row.get("created_at")),
        -float(row.get("rerank_score") or row.get("score") or 0),
        str(row.get("path") or ""),
    )



def _retrieval_policy_label(req: NormalAgentRequest, row: dict[str, Any]) -> str:
    return f"tier={_retrieval_tier(req, row)}:{_row_scope_label(req, row)}:{str(row.get('lifecycle_state') or row.get('lifecycle') or '').strip().lower() or 'unknown'}:{str(row.get('memory_type') or '').strip().lower() or 'unknown'}"



def _claim_polarity(text: str) -> str | None:
    lowered = str(text or "").strip().lower()
    if not lowered:
        return None
    for cue in NEGATIVE_CUES:
        if cue in lowered:
            return "negative"
    for cue in POSITIVE_CUES:
        if cue in lowered:
            return "positive"
    return None



def _conflict_topic_key(text: str) -> str | None:
    value = str(text or "").strip().lower()
    if not value:
        return None
    for prefix in CONFLICT_PREFIXES:
        if value.startswith(prefix):
            value = value[len(prefix):].strip()
    for cue in NEGATIVE_CUES + POSITIVE_CUES:
        value = value.replace(cue, " ")
    normalized = "".join(ch if ch.isalnum() or ch.isspace() else " " for ch in value)
    tokens = [token for token in normalized.split() if token and token not in CONFLICT_STOPWORDS]
    if len(tokens) < 2:
        return None
    topic = " ".join(tokens)
    return topic if len(topic) >= 16 else None



def _precedence_tuple(req: NormalAgentRequest, candidate: dict[str, Any]) -> tuple[int, int, int, float, float]:
    row = candidate["row"]
    return (
        _scope_precedence(req, row),
        _memory_type_precedence(row),
        LIFECYCLE_PRECEDENCE.get(str(row.get("lifecycle_state") or "").strip().lower(), 0),
        _parse_timestamp(row.get("updated_at") or row.get("created_at")),
        float(row.get("rerank_score") or row.get("score") or 0),
    )



def _conflict_family(sections: list[str]) -> str:
    primary = sections[0]
    if primary in {"must_follow_rules", "relevant_decisions", "implementation_patterns", "recommended_next_assumptions"}:
        return "guidance"
    return primary



def _conflict_note(section: str, topic: str) -> str:
    label = section.replace("_", " ")
    return f"Conflicting {label} memories disagree about {topic}."



def _resolve_candidate_conflicts(req: NormalAgentRequest, candidates: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str], dict[str, int]]:
    grouped: dict[tuple[str, str], dict[str, list[dict[str, Any]]]] = {}
    passthrough: list[dict[str, Any]] = []

    for candidate in candidates:
        conflict_family = _conflict_family(candidate["sections"])
        polarity = _claim_polarity(candidate["text"])
        topic = _conflict_topic_key(candidate["text"])
        if not polarity or not topic:
            passthrough.append(candidate)
            continue
        grouped.setdefault((conflict_family, topic), {}).setdefault(polarity, []).append(candidate)

    resolved: list[dict[str, Any]] = list(passthrough)
    conflict_notes: list[str] = []
    suppressed_count = 0
    unresolved_count = 0

    for (section, topic), by_polarity in grouped.items():
        if len(by_polarity) <= 1:
            for items in by_polarity.values():
                resolved.extend(items)
            continue

        ranked = sorted(
            (
                {
                    "polarity": polarity,
                    "best": max(items, key=lambda item: (_precedence_tuple(req, item), -int(item["index"]))),
                    "items": items,
                }
                for polarity, items in by_polarity.items()
            ),
            key=lambda item: (_precedence_tuple(req, item["best"]), -int(item["best"]["index"])),
            reverse=True,
        )
        winner = ranked[0]
        runner_up = ranked[1]
        if _precedence_tuple(req, winner["best"]) > _precedence_tuple(req, runner_up["best"]):
            winner_text = winner["best"]["text"]
            for item in winner["items"]:
                if item["text"] == winner_text:
                    resolved.append(item)
                else:
                    suppressed_count += 1
            for losing in ranked[1:]:
                suppressed_count += len(losing["items"])
            continue

        unresolved_count += 1
        conflict_notes.append(_conflict_note(section, topic))

    return resolved, conflict_notes, {
        "conflicts_suppressed_count": suppressed_count,
        "unresolved_conflict_count": unresolved_count,
    }


def build_normal_agent_packet(
    req: NormalAgentRequest,
    rows: list[dict[str, Any]],
    retrieval_meta: dict[str, Any] | None = None,
) -> NormalAgentPacket:
    retrieval_meta = dict(retrieval_meta or {})
    packet_id = _stable_packet_id(req)
    budget_used = 0

    packet = NormalAgentPacket(
        packet_id=packet_id,
        status="insufficient_context",
        task_context={
            "query": req.query,
            "project_id": req.project_id,
            "subproject_id": req.subproject_id,
            "agent_mode": req.agent_mode,
            "domain_tags": list(req.domain_tags),
            "allowed_memory_types": list(req.allowed_memory_types),
            "allowed_lifecycle_states": list(req.allowed_lifecycle_states),
            "max_context_items": req.max_context_items,
            "max_context_tokens": req.max_context_tokens,
            "requester_role": req.requester.requester_role,
            "requester_id": req.requester.requester_id,
            "retrieval_bias": req.retrieval_bias.value,
            "need_decisions": req.need_decisions,
            "need_patterns": req.need_patterns,
            "need_bug_history": req.need_bug_history,
            "need_recent_session_context": req.need_recent_session_context,
        },
        meta={
            "packet_contract_version": NORMAL_AGENT_PACKET_CONTRACT_VERSION,
            "packet_compiler": "normal_agent_packet_compiler",
            "max_context_items": req.max_context_items,
            "max_context_tokens": req.max_context_tokens,
            "applied_memory_type_filters": list(req.allowed_memory_types),
            "applied_lifecycle_filters": list(req.allowed_lifecycle_states),
            "retrieval_meta": retrieval_meta,
        },
    )

    usable_rows: list[dict[str, Any]] = []
    filtered_out = 0
    blocked_internal_source_paths: list[str] = []
    for row in rows:
        if not _is_allowed_normal_agent_source(row):
            filtered_out += 1
            blocked_internal_source_paths.append(str(row.get("path") or ""))
            continue
        if not _matches_filters(row, req):
            filtered_out += 1
            continue
        usable_rows.append(row)

    usable_rows.sort(key=lambda row: _retrieval_policy_tuple(req, row))

    candidates: list[dict[str, Any]] = []
    for index, row in enumerate(usable_rows):
        text = _normalize_text(row)
        if not text:
            continue
        sections = _classify_sections(req, row)
        if not sections:
            sections = ["recommended_next_assumptions"]
        candidates.append(
            {
                "row": row,
                "text": text,
                "sections": sections,
                "priority": _section_priority(sections),
                "tokens": _approx_tokens(text),
                "score": float(row.get("rerank_score") or row.get("score") or 0),
                "index": index,
                "policy_tuple": _retrieval_policy_tuple(req, row),
                "policy_label": _retrieval_policy_label(req, row),
            }
        )

    candidates, conflict_notes, conflict_meta = _resolve_candidate_conflicts(req, candidates)
    candidates.sort(key=lambda item: (item["priority"], item["policy_tuple"], item["index"], item["text"]))

    accepted_dedupe_keys: list[str] = []
    duplicates_collapsed = 0
    skipped_for_budget = 0
    item_budget_exhausted = False
    token_budget_exhausted = False

    for candidate in candidates:
        text = candidate["text"]
        if _is_near_duplicate(text, accepted_dedupe_keys):
            duplicates_collapsed += 1
            continue

        if len(packet.source_refs) >= req.max_context_items:
            item_budget_exhausted = True
            skipped_for_budget += 1
            continue
        if budget_used + candidate["tokens"] > req.max_context_tokens:
            token_budget_exhausted = True
            skipped_for_budget += 1
            continue

        accepted_dedupe_keys.append(_dedupe_key(text))

        candidate_sections = list(candidate["sections"])
        if "must_follow_rules" in candidate_sections and "implementation_patterns" in candidate_sections:
            candidate_sections = [section for section in candidate_sections if section != "implementation_patterns"]

        for section in candidate_sections:
            if section == "must_follow_rules":
                _push_unique(packet.must_follow_rules, text)
            elif section == "relevant_decisions":
                _push_unique(packet.relevant_decisions, text)
            elif section == "implementation_patterns":
                _push_unique(packet.implementation_patterns, text)
            elif section == "recent_history":
                _push_unique(packet.recent_history, text)
            elif section == "known_failures_or_bug_history":
                _push_unique(packet.known_failures_or_bug_history, text)
            elif section == "open_questions":
                _push_unique(packet.open_questions, text)
            elif section == "recommended_next_assumptions":
                _push_unique(packet.recommended_next_assumptions, f"Assume this remains relevant unless fresher evidence contradicts it: {text}")

        row = candidate["row"]
        packet.source_refs.append(
            PacketSourceRef(
                path=str(row.get("path") or ""),
                source_type=str(row.get("source_type") or ""),
                memory_type=str(row.get("memory_type") or "") or None,
                lifecycle_state=str(row.get("lifecycle_state") or "") or None,
                score=float(row.get("rerank_score") or row.get("score") or 0) if (row.get("rerank_score") is not None or row.get("score") is not None) else None,
            )
        )
        budget_used += candidate["tokens"]

    packet.conflict_notes.extend(conflict_notes)

    if not packet.must_follow_rules:
        packet.missing_context.append("No rule/guardrail memories met the bounded filter.")
    if req.need_decisions and not packet.relevant_decisions:
        packet.missing_context.append("No decision memories met the bounded filter.")
    if req.need_patterns and not packet.implementation_patterns:
        packet.missing_context.append("No implementation-pattern memories met the bounded filter.")
    if req.need_bug_history and not packet.known_failures_or_bug_history:
        packet.missing_context.append("No bug-history memories met the bounded filter.")

    if packet.source_refs:
        packet.status = "fulfilled" if not packet.missing_context else "partial"
    packet.meta.update(
        {
            "budget_used_tokens": budget_used,
            "selected_item_count": len(packet.source_refs),
            "filtered_out_count": filtered_out,
            "blocked_internal_source_count": len(blocked_internal_source_paths),
            "blocked_internal_source_paths": blocked_internal_source_paths[:8],
            "normal_agent_boundary_policy_version": NORMAL_AGENT_BOUNDARY_POLICY_VERSION,
            "normal_agent_boundary_policy": {
                "no_script_reading": True,
                "blocked_markers": list(DISALLOWED_NORMAL_AGENT_PATH_PREFIXES),
                "explicit_attachment_override": True,
            },
            "duplicates_collapsed_count": duplicates_collapsed,
            "skipped_for_budget_count": skipped_for_budget,
            "item_budget_exhausted": item_budget_exhausted,
            "token_budget_exhausted": token_budget_exhausted,
            "selection_order": [ref.memory_type or ref.source_type or "unknown" for ref in packet.source_refs],
            "normal_agent_section_contract_version": NORMAL_AGENT_SECTION_CONTRACT_VERSION,
            "normal_agent_section_contract": {
                "agent_mode": "normal",
                "stable_sections": [
                    "must_follow_rules",
                    "relevant_decisions",
                    "implementation_patterns",
                    "recent_history",
                    "known_failures_or_bug_history",
                    "open_questions",
                    "recommended_next_assumptions",
                    "conflict_notes",
                    "source_refs",
                    "missing_context",
                ],
            },
            "normal_agent_shaping_policy_version": NORMAL_AGENT_SHAPING_POLICY_VERSION,
            "normal_agent_shaping_policy": {
                "applied": True,
                "stage": "post_retrieval_pre_packet_emit",
                "budget_fields": {
                    "max_context_items": req.max_context_items,
                    "max_context_tokens": req.max_context_tokens,
                },
                "section_priority": [
                    "must_follow_rules",
                    "relevant_decisions",
                    "implementation_patterns",
                    "known_failures_or_bug_history",
                    "recent_history",
                    "open_questions",
                    "recommended_next_assumptions",
                ],
                "truncate_behavior": "respect_item_and_token_budgets_and_surface_skips_in_meta",
            },
            "retrieval_policy_order": [_retrieval_policy_label(req, row) for row in usable_rows[:12]],
            "retrieval_policy": {
                "scope_order": [
                    "subproject durable memory",
                    "project durable memory",
                    "recent candidate/session memory",
                    "archived/project history",
                    "global reusable patterns",
                ],
                "weight_order": [
                    "architecture rules: high",
                    "explicit decisions: high",
                    "bug history: medium-high",
                    "workflow rules: medium",
                    "summaries: low unless nothing else exists",
                ],
            },
            "normal_mode_retrieval_policy_version": NORMAL_MODE_RETRIEVAL_POLICY_VERSION,
            "normal_mode_retrieval_policy": {
                "applied": True,
                "stage": "pre_packet_shaping",
                "order": [
                    "subproject durable memory",
                    "project durable memory",
                    "recent candidate/session memory",
                    "archived/project history",
                    "global reusable patterns",
                ],
                "weighting": {
                    "architecture_rule": "high",
                    "decision": "high",
                    "bug_history": "medium-high",
                    "workflow_rule": "medium",
                    "summary": "low_unless_sparse",
                },
            },
            "conflict_notes_count": len(packet.conflict_notes),
            "normal_mode_conflict_policy_version": NORMAL_MODE_CONFLICT_POLICY_VERSION,
            "normal_mode_conflict_policy": {
                "applied": True,
                "winner_rules": [
                    "project_scoped_over_global",
                    "explicit_architecture_rule_over_inferred_pattern",
                    "latest_durable_decision_over_older_candidate_note",
                    "newer_timestamp_within_same_precedence_tier",
                    "durable_over_reinforced_over_candidate",
                ],
                "unresolved_behavior": "emit_conflict_notes_and_do_not_mix_conflicting_guidance",
            },
            "precedence_policy": [
                "project_scoped_over_global",
                "explicit_architecture_rule_over_inferred_pattern",
                "latest_durable_decision_over_older_candidate_note",
                "newer_timestamp_within_same_precedence_tier",
                "durable_over_reinforced_over_candidate",
            ],
            **conflict_meta,
        }
    )
    return packet



def _current_time() -> float:
    return time.time()



def _request_cache_key(req: NormalAgentRequest) -> str:
    raw = "|".join(
        [
            req.project_id,
            req.subproject_id or "",
            req.query,
            req.requester.requester_role,
            req.requester.requester_id or "",
            req.retrieval_bias.value,
            ",".join(req.domain_tags),
            ",".join(req.allowed_memory_types),
            ",".join(req.allowed_lifecycle_states),
            str(req.max_context_items),
            str(req.max_context_tokens),
            str(req.need_decisions),
            str(req.need_patterns),
            str(req.need_bug_history),
            str(req.need_recent_session_context),
        ]
    )
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()



def _row_fingerprint(row: dict[str, Any]) -> str:
    payload = {
        "path": str(row.get("path") or ""),
        "source_type": str(row.get("source_type") or ""),
        "memory_type": str(row.get("memory_type") or ""),
        "lifecycle_state": str(row.get("lifecycle_state") or ""),
        "text": _normalize_text(row),
        "updated_at": str(row.get("updated_at") or ""),
        "created_at": str(row.get("created_at") or ""),
        "scope": str(row.get("scope") or ""),
        "project_id": str(row.get("project_id") or ""),
        "subproject_id": str(row.get("subproject_id") or ""),
        "score": row.get("rerank_score") if row.get("rerank_score") is not None else row.get("score"),
    }
    return hashlib.sha1(repr(payload).encode("utf-8")).hexdigest()



def _build_freshness_snapshot(req: NormalAgentRequest, rows: list[dict[str, Any]]) -> dict[str, Any]:
    usable_rows: list[dict[str, Any]] = []
    for row in rows:
        if not _is_allowed_normal_agent_source(row):
            continue
        if not _matches_filters(row, req):
            continue
        usable_rows.append(row)

    latest_update = 0.0
    row_fingerprints: list[str] = []
    for row in usable_rows:
        latest_update = max(latest_update, _parse_timestamp(row.get("updated_at") or row.get("created_at")))
        row_fingerprints.append(_row_fingerprint(row))

    joined = "|".join(sorted(row_fingerprints))
    return {
        "usable_row_count": len(usable_rows),
        "latest_update_at": latest_update,
        "rows_fingerprint": hashlib.sha1(joined.encode("utf-8")).hexdigest() if joined else hashlib.sha1(b"empty").hexdigest(),
    }



def _evaluate_packet_reuse(req: NormalAgentRequest, freshness_snapshot: dict[str, Any], cached_entry: dict[str, Any] | None) -> dict[str, Any]:
    if not cached_entry:
        return {"decision": "regenerate", "reason": "no_prior_cached_packet"}

    if cached_entry.get("schema_version") != PACKET_CACHE_SCHEMA_VERSION:
        return {"decision": "regenerate", "reason": "cache_schema_version_changed"}

    if cached_entry.get("request_cache_key") != _request_cache_key(req):
        return {"decision": "regenerate", "reason": "request_key_changed"}

    cache_age_seconds = max(0.0, _current_time() - float(cached_entry.get("cached_at") or 0.0))
    ttl_seconds = PACKET_CACHE_TTL_SECONDS.get(req.retrieval_bias.value, PACKET_CACHE_TTL_SECONDS["mixed"])
    if cache_age_seconds > ttl_seconds:
        return {
            "decision": "regenerate",
            "reason": "cache_ttl_expired",
            "cache_age_seconds": round(cache_age_seconds, 3),
            "ttl_seconds": ttl_seconds,
        }

    cached_freshness = cached_entry.get("freshness_snapshot") or {}
    if cached_freshness.get("rows_fingerprint") != freshness_snapshot.get("rows_fingerprint"):
        return {"decision": "regenerate", "reason": "memory_rows_fingerprint_changed"}
    if cached_freshness.get("usable_row_count") != freshness_snapshot.get("usable_row_count"):
        return {"decision": "regenerate", "reason": "usable_row_count_changed"}
    if float(cached_freshness.get("latest_update_at") or 0.0) != float(freshness_snapshot.get("latest_update_at") or 0.0):
        return {"decision": "regenerate", "reason": "latest_memory_update_changed"}

    return {
        "decision": "reuse",
        "reason": "request_and_memory_freshness_match",
        "cache_age_seconds": round(cache_age_seconds, 3),
        "ttl_seconds": ttl_seconds,
    }



def _cache_packet(req: NormalAgentRequest, packet: NormalAgentPacket, freshness_snapshot: dict[str, Any]) -> None:
    NORMAL_AGENT_PACKET_CACHE[_request_cache_key(req)] = {
        "schema_version": PACKET_CACHE_SCHEMA_VERSION,
        "request_cache_key": _request_cache_key(req),
        "packet": packet.model_dump(mode="python"),
        "freshness_snapshot": dict(freshness_snapshot),
        "cached_at": _current_time(),
    }



def _load_cached_packet(cached_entry: dict[str, Any]) -> NormalAgentPacket:
    return NormalAgentPacket.model_validate(copy.deepcopy(cached_entry["packet"]))



def _apply_packet_reuse_meta(packet: NormalAgentPacket, *, decision: str, reason: str, cache_key: str, freshness_snapshot: dict[str, Any], cache_age_seconds: float | None = None, ttl_seconds: int | None = None) -> None:
    reuse_meta: dict[str, Any] = {
        "decision": decision,
        "reason": reason,
        "cache_key": cache_key,
        "freshness": dict(freshness_snapshot),
        "policy_version": NORMAL_MODE_FRESHNESS_POLICY_VERSION,
        "policy": {
            "ttl_by_bias": dict(PACKET_CACHE_TTL_SECONDS),
            "reuse_requires": [
                "same_request_cache_key",
                "same_cache_schema_version",
                "cache_age_within_ttl",
                "same_usable_row_count",
                "same_latest_memory_update",
                "same_rows_fingerprint",
            ],
        },
    }
    if cache_age_seconds is not None:
        reuse_meta["cache_age_seconds"] = cache_age_seconds
    if ttl_seconds is not None:
        reuse_meta["ttl_seconds"] = ttl_seconds
    packet.meta["packet_reuse"] = reuse_meta



def _reset_normal_agent_packet_cache() -> None:
    NORMAL_AGENT_PACKET_CACHE.clear()



def build_normal_agent_packet_response(
    req: NormalAgentRequest,
    rows: list[dict[str, Any]],
    retrieval_meta: dict[str, Any] | None = None,
) -> NormalAgentPacketResponse:
    cache_key = _request_cache_key(req)
    freshness_snapshot = _build_freshness_snapshot(req, rows)
    reuse_decision = _evaluate_packet_reuse(req, freshness_snapshot, NORMAL_AGENT_PACKET_CACHE.get(cache_key))

    if reuse_decision["decision"] == "reuse":
        packet = _load_cached_packet(NORMAL_AGENT_PACKET_CACHE[cache_key])
        _apply_packet_reuse_meta(
            packet,
            decision="reused",
            reason=str(reuse_decision["reason"]),
            cache_key=cache_key,
            freshness_snapshot=freshness_snapshot,
            cache_age_seconds=float(reuse_decision.get("cache_age_seconds") or 0.0),
            ttl_seconds=int(reuse_decision.get("ttl_seconds") or 0),
        )
    else:
        packet = build_normal_agent_packet(req, rows, retrieval_meta=retrieval_meta)
        _apply_packet_reuse_meta(
            packet,
            decision="regenerated",
            reason=str(reuse_decision["reason"]),
            cache_key=cache_key,
            freshness_snapshot=freshness_snapshot,
            cache_age_seconds=float(reuse_decision.get("cache_age_seconds") or 0.0) if reuse_decision.get("cache_age_seconds") is not None else None,
            ttl_seconds=int(reuse_decision.get("ttl_seconds") or 0) if reuse_decision.get("ttl_seconds") is not None else None,
        )
        _cache_packet(req, packet, freshness_snapshot)

    return NormalAgentPacketResponse(
        ok=True,
        request=req,
        packet=packet,
        meta={
            "entrypoint": "normal_agent_packet",
            "status": packet.status,
            "selected_item_count": packet.meta.get("selected_item_count", 0),
            "packet_reuse": packet.meta.get("packet_reuse", {}),
        },
    )
