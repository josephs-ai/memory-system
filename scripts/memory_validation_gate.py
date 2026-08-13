"""
Schema and content validation for memory items before storage.
Enforces required fields, value constraints, and structural integrity.
"""
from __future__ import annotations

import re

from memory_candidate_schema import StructuredMemoryCandidate


# Instructions addressed TO an agent are not claims about the world. They read
# like durable rules ("must", "only", imperative voice) and so score highly,
# but they describe a prompt, not a fact. 32% of the first promoted batch was
# this single class -- mostly "Return JSON only: {...}" reviewer scaffolding.
# \b is scoped to each word-initial alternative rather than wrapped around the
# whole group: a word boundary cannot match before "[", so a leading \b silently
# disabled the bracketed-tag alternative entirely.
PROMPT_DIRECTIVE_RE = re.compile(
    r"(?i)(\breturn\s+json\s+only|\brespond\s+with|\breply\s+with|\boutput\s+only"
    r"|\byou\s+are\s+the\s+(builder|reviewer|tester|coder)\b"
    r"|\[\s*(subagent\s+task|task|system|instructions?)\s*\]\s*:)"
)

# A JSON object embedded in a claim means the prompt payload came along with it.
EMBEDDED_PAYLOAD_RE = re.compile(r"\{\s*[\"']")

# Digests defeat deduplication: the same scaffolding with a different hash is a
# brand-new "memory" every time, so this noise compounds instead of collapsing.
OPAQUE_DIGEST_RE = re.compile(r"\b[0-9a-f]{16,}\b")


def prompt_noise_reasons(text: str) -> list[str]:
    """Reasons this text is agent scaffolding rather than a durable claim.

    Shared by the validation gate and the promotion gate so that items already
    sitting in the queue -- scored before these checks existed -- are still
    filtered without needing to be re-extracted.
    """
    text = str(text or "")
    reasons = []
    if PROMPT_DIRECTIVE_RE.search(text):
        reasons.append("prompt_directive")
    if EMBEDDED_PAYLOAD_RE.search(text):
        reasons.append("embedded_payload")
    if OPAQUE_DIGEST_RE.search(text):
        reasons.append("opaque_digest")
    return reasons


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
    reasons.extend(prompt_noise_reasons(candidate.claim_text))
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
