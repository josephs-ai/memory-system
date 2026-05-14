"""
Schema and content validation for memory items before storage.
Enforces required fields, value constraints, and structural integrity.
"""
from __future__ import annotations

from memory_candidate_schema import StructuredMemoryCandidate


REASONING_MARKERS = (
    "let's think",
    "step by step",
    "i think",
    "i suspect",
    "brainstorm",
)
SPECULATION_MARKERS = (
    "maybe",
    "might",
    "could",
    "perhaps",
    "probably",
)


def validate_candidate(candidate: StructuredMemoryCandidate) -> StructuredMemoryCandidate:
    text = candidate.claim_text.strip().lower()
    reasons: list[str] = []

    if not text:
        reasons.append("empty_claim_text")
    if any(marker in text for marker in REASONING_MARKERS):
        candidate.contains_reasoning = True
        reasons.append("contains_reasoning")
    if any(marker in text for marker in SPECULATION_MARKERS):
        candidate.contains_speculation = True
        reasons.append("contains_speculation")
    if "?" in text:
        reasons.append("question_not_durable")
    if len(text.split()) > 80:
        reasons.append("not_atomic_enough")
        candidate.is_atomic = False
    if candidate.authority_basis.value in {"assistant_inferred", "summary_inferred", "unknown"}:
        reasons.append("authority_not_explicit")
    if not candidate.scope_envelope.scope_type:
        reasons.append("missing_scope_type")

    candidate.validation_reasons = reasons
    candidate.validation_status = "valid" if not reasons else "needs_review"
    return candidate
