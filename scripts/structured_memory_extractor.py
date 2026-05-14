"""
Memory system utility: structured memory extractor.

Key functions: extract_structured_candidates
"""
from __future__ import annotations

import hashlib
import re
from typing import Iterable

from context_scope import build_context_scope
from memory_candidate_schema import (
    AuthorityBasis,
    ClaimType,
    DurabilityClass,
    ImpactLevel,
    ScopeEnvelope,
    StructuredMemoryCandidate,
)
from memory_validation_gate import validate_candidate


PROMPT_TEMPLATE = """You are a structural memory extractor. Return ONLY valid JSON matching the StructuredMemoryCandidate schema. Do not return prose. Never include chain-of-thought. Extract only atomic claims with explicit authority/provenance."""


def _clean_line(text: str) -> str:
    text = re.sub(r"^(USER|ASSISTANT):\s*", "", text.strip(), flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", text).strip()



def _looks_structural(text: str) -> bool:
    lowered = text.lower()
    return any(token in lowered for token in ["must", "should", "decision", "prefers", "supports", "uses", "pipeline", "workflow", "approved", "rule"])



def _claim_type(text: str) -> ClaimType:
    lowered = text.lower()
    if "prefers" in lowered:
        return ClaimType.preference
    if "decision" in lowered:
        return ClaimType.decision
    if any(token in lowered for token in ["must", "should", "rule"]):
        return ClaimType.rule
    return ClaimType.fact



def _authority_basis(text: str) -> AuthorityBasis:
    lowered = text.lower()
    if lowered.startswith("user:"):
        return AuthorityBasis.user_explicit
    if lowered.startswith("system:"):
        return AuthorityBasis.system_tool_verified
    return AuthorityBasis.unknown



def extract_structured_candidates(text: str, *, source_agent: str, source_session: str, source_chunk: str) -> list[StructuredMemoryCandidate]:
    candidates: list[StructuredMemoryCandidate] = []
    for raw_line in text.splitlines():
        line = _clean_line(raw_line)
        if not line or not _looks_structural(line):
            continue
        scope = build_context_scope({"text": line, "source_session": source_session, "scope": "stable"})
        candidate = StructuredMemoryCandidate(
            candidate_id=f"cand-{hashlib.sha1(f'{source_chunk}|{line}'.encode('utf-8')).hexdigest()[:12]}",
            source_agent=source_agent,
            source_session=source_session,
            source_chunk=source_chunk,
            claim_text=line,
            claim_type=_claim_type(line),
            authority_basis=_authority_basis(line),
            durability_class=DurabilityClass.candidate,
            impact_level=ImpactLevel.medium,
            scope_envelope=ScopeEnvelope(
                project_id=scope.get("project_id"),
                subproject_id=scope.get("subproject_id"),
                workflow_id=scope.get("workflow_id"),
                pipeline_id=scope.get("pipeline_id"),
                session_id=source_session,
                scope_id=scope.get("scope_id"),
                scope_type=scope.get("scope_type"),
                inheritance_policy=scope.get("inheritance_policy"),
                scope_confidence=float(scope.get("scope_confidence") or 0.0),
                raw=scope,
            ),
            extractor_confidence=0.5,
        )
        candidates.append(validate_candidate(candidate))
    return candidates
