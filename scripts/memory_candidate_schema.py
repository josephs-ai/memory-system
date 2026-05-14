"""
Pydantic schema definitions for memory candidates — the structured format
used throughout the extraction and judgment pipeline.
"""
from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ClaimType(str, Enum):
    rule = "rule"
    decision = "decision"
    preference = "preference"
    fact = "fact"
    implementation_pattern = "implementation_pattern"
    bug_history = "bug_history"
    open_question = "open_question"
    summary_only = "summary_only"


class AuthorityBasis(str, Enum):
    user_explicit = "user_explicit"
    developer_explicit = "developer_explicit"
    tester_explicit = "tester_explicit"
    system_tool_verified = "system_tool_verified"
    assistant_inferred = "assistant_inferred"
    summary_inferred = "summary_inferred"
    unknown = "unknown"


class DurabilityClass(str, Enum):
    ephemeral = "ephemeral"
    session = "session"
    candidate = "candidate"
    review_required = "review_required"
    durable_eligible = "durable_eligible"


class ImpactLevel(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"
    critical = "critical"


class ScopeEnvelope(BaseModel):
    project_id: str | None = None
    subproject_id: str | None = None
    workflow_id: str | None = None
    pipeline_id: str | None = None
    session_id: str | None = None
    scope_id: str | None = None
    scope_type: str | None = None
    inheritance_policy: str | None = None
    scope_confidence: float = 0.0
    raw: dict[str, Any] = Field(default_factory=dict)


class StructuredMemoryCandidate(BaseModel):
    candidate_id: str
    source_event_ids: list[str] = Field(default_factory=list)
    source_span_ids: list[str] = Field(default_factory=list)
    source_agent: str = "unknown"
    source_session: str = "unknown"
    source_chunk: str = "unknown"
    claim_text: str
    claim_type: ClaimType = ClaimType.fact
    subject: str | None = None
    predicate: str | None = None
    object: str | None = None
    scope_envelope: ScopeEnvelope = Field(default_factory=ScopeEnvelope)
    authority_basis: AuthorityBasis = AuthorityBasis.unknown
    durability_class: DurabilityClass = DurabilityClass.candidate
    impact_level: ImpactLevel = ImpactLevel.low
    requires_approval: bool = True
    contains_reasoning: bool = False
    contains_speculation: bool = False
    is_atomic: bool = True
    extractor_confidence: float = 0.0
    normalization_notes: list[str] = Field(default_factory=list)
    validation_status: str = "unvalidated"
    validation_reasons: list[str] = Field(default_factory=list)

    def to_legacy_item(self) -> dict[str, Any]:
        scope_payload = self.scope_envelope.model_dump(mode="python")
        return {
            "candidate_id": self.candidate_id,
            "raw_text": self.claim_text,
            "normalized_text": self.claim_text,
            "source_agent": self.source_agent,
            "source_session": self.source_session,
            "source_chunk": self.source_chunk,
            "candidate_score": round(float(self.extractor_confidence or 0.0) * 10, 2),
            "candidate_reasons": list(self.validation_reasons or []),
            "claim_type": self.claim_type.value,
            "authority_basis": self.authority_basis.value,
            "durability_class": self.durability_class.value,
            "impact_level": self.impact_level.value,
            "requires_approval": self.requires_approval,
            "contains_reasoning": self.contains_reasoning,
            "contains_speculation": self.contains_speculation,
            "is_atomic": self.is_atomic,
            "scope_envelope": scope_payload,
            "context_scope_id": self.scope_envelope.scope_id,
            "context_scope_type": self.scope_envelope.scope_type,
            "context_scope_payload": scope_payload,
            "inheritance_policy": self.scope_envelope.inheritance_policy,
        }
