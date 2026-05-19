"""
Memory system utility: structured memory extractor.

Parses dehydrated transcript chunks (JSON message envelopes) and extracts
atomic, durable memory candidates with dynamic confidence/importance scoring.

Key functions: extract_structured_candidates
"""
from __future__ import annotations

import hashlib
import json
import re
from typing import Any

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


# ---------------------------------------------------------------------------
# Text extraction from JSON message envelopes
# ---------------------------------------------------------------------------

def _extract_text_from_message(raw_line: str) -> list[dict[str, Any]]:
    """Parse a JSON message envelope or dehydrated text line and return text segments.

    Returns list of dicts: {"text": str, "role": str, "message_id": str}
    Handles both JSON envelopes and dehydrated plain text (USER:/ASSISTANT:/A: prefixes).
    Skips thinking blocks, tool calls, encrypted signatures, and non-text content.
    """
    try:
        msg = json.loads(raw_line)
    except (json.JSONDecodeError, TypeError):
        # Not JSON — check for dehydrated role prefixes
        text = raw_line.strip()
        if not text:
            return []
        role = "unknown"
        # Detect role from dehydrated transcript format
        for prefix, detected_role in [
            ("USER: ", "user"),
            ("ASSISTANT: ", "assistant"),
            ("A: ", "assistant"),
            ("SYSTEM: ", "system"),
        ]:
            if text.startswith(prefix):
                text = text[len(prefix):]
                role = detected_role
                break
        if text:
            return [{"text": text, "role": role, "message_id": ""}]
        return []

    message = msg.get("message") or {}
    role = message.get("role", "unknown")
    message_id = msg.get("id", "")
    content = message.get("content")

    # Handle simple string content
    if isinstance(content, str):
        text = content.strip()
        if text:
            return [{"text": text, "role": role, "message_id": message_id}]
        return []

    # Handle content array
    if not isinstance(content, list):
        return []

    results = []
    for block in content:
        if not isinstance(block, dict):
            continue
        block_type = block.get("type", "")

        # Only extract actual text blocks
        if block_type == "text":
            text = (block.get("text") or "").strip()
            if text and len(text) > 10:  # Skip trivially short fragments
                results.append({"text": text, "role": role, "message_id": message_id})
        # Skip: thinking, toolCall, toolResult, thinkingSignature, textSignature, etc.

    return results


# ---------------------------------------------------------------------------
# Sentence splitting — break text into atomic claim-sized segments
# ---------------------------------------------------------------------------

# Noise patterns that should be filtered out entirely
_NOISE_PATTERNS = [
    re.compile(r"^```", re.MULTILINE),  # Code fence markers
    re.compile(r"^\s*import\s+", re.MULTILINE),  # Code imports
    re.compile(r"^\s*(def|class|if|for|while|return|try|except)\s", re.MULTILINE),  # Code
    re.compile(r"^\s*\{", re.MULTILINE),  # JSON/dict literals
    re.compile(r"^(HEARTBEAT_OK|NO_REPLY|NONE)\s*$", re.IGNORECASE),  # Bot commands
    re.compile(r"^\[\[reply_to", re.IGNORECASE),  # Reply tags
    re.compile(r"^https?://", re.IGNORECASE),  # Bare URLs
]

# Structural claim indicators — text containing these is more likely durable
_STRUCTURAL_KEYWORDS = {
    "high": [
        "must", "always", "never", "rule", "decision", "approved", "rejected",
        "architecture", "pattern", "lesson learned", "guardrail", "constraint",
        "root cause", "fix was", "the fix", "solution is", "resolved by",
    ],
    "medium": [
        "should", "prefers", "preference", "workflow", "pipeline", "supports",
        "uses", "configured", "implemented", "completed", "milestone",
        "bug", "issue", "error", "failure", "broken", "fixed",
    ],
    "low": [
        "works", "running", "tested", "verified", "checked", "confirmed",
        "deployed", "installed", "updated", "changed", "modified",
    ],
}


def _split_into_claims(text: str) -> list[str]:
    """Split text into sentence-level segments suitable for atomic claims."""
    # Remove markdown headers but keep content
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    # Remove markdown bold/italic markers
    text = re.sub(r"\*{1,3}([^*]+)\*{1,3}", r"\1", text)
    # Remove markdown bullet markers
    text = re.sub(r"^\s*[-*+]\s+", "", text, flags=re.MULTILINE)

    # Split on sentence boundaries and newlines
    segments = re.split(r"(?<=[.!?])\s+|\n+", text)

    claims = []
    for seg in segments:
        seg = seg.strip()
        if not seg or len(seg) < 15:  # Too short to be a meaningful claim
            continue
        if len(seg) > 500:  # Too long — likely a paragraph, not atomic
            continue
        # Filter noise
        if any(p.search(seg) for p in _NOISE_PATTERNS):
            continue
        claims.append(seg)

    return claims


def _is_noise(text: str) -> bool:
    """Detect text that's clearly not a durable memory claim."""
    lowered = text.lower()
    # Bot/system noise
    if any(phrase in lowered for phrase in [
        "let me know if", "here's what i found", "i'll check",
        "sure, i can", "i've opened", "the browser",
        "searching for", "reading file", "writing to",
    ]):
        return True
    # Pure questions without assertions
    if text.strip().endswith("?") and not any(
        kw in lowered for kw in ["must", "should", "rule", "decision"]
    ):
        return True
    return False


# ---------------------------------------------------------------------------
# Claim classification and scoring
# ---------------------------------------------------------------------------

def _claim_type(text: str) -> ClaimType:
    lowered = text.lower()
    if any(kw in lowered for kw in ["prefers", "preference", "likes to", "wants to"]):
        return ClaimType.preference
    if any(kw in lowered for kw in ["decided", "decision", "chose", "chosen", "approved"]):
        return ClaimType.decision
    if any(kw in lowered for kw in ["must", "should", "rule", "never", "always", "constraint"]):
        return ClaimType.rule
    if any(kw in lowered for kw in ["pattern", "implementation"]):
        return ClaimType.implementation_pattern
    if any(kw in lowered for kw in ["bug", "broke", "broken", "regression", "crash"]):
        return ClaimType.bug_history
    return ClaimType.fact


def _authority_basis(text: str, role: str) -> AuthorityBasis:
    if role == "user":
        return AuthorityBasis.user_explicit
    if role == "system":
        return AuthorityBasis.system_tool_verified
    if role == "assistant":
        return AuthorityBasis.assistant_inferred
    return AuthorityBasis.unknown


def _durability_class(text: str, claim_type: ClaimType) -> DurabilityClass:
    """Determine durability based on claim type and structural markers."""
    # Rules, decisions, and patterns are inherently more durable
    if claim_type in (ClaimType.rule, ClaimType.decision):
        return DurabilityClass.durable_eligible
    if claim_type == ClaimType.implementation_pattern:
        return DurabilityClass.durable_eligible
    if claim_type == ClaimType.preference:
        return DurabilityClass.review_required
    # Bug history is durable (lessons learned)
    if claim_type == ClaimType.bug_history:
        return DurabilityClass.durable_eligible

    lowered = text.lower()
    # Explicit durable signals
    if any(kw in lowered for kw in [
        "lesson learned", "root cause", "architecture", "guardrail",
        "milestone", "completed", "approved",
    ]):
        return DurabilityClass.durable_eligible
    # Review-worthy signals
    if any(kw in lowered for kw in [
        "fixed", "resolved", "implemented", "configured", "deployed",
    ]):
        return DurabilityClass.review_required

    return DurabilityClass.candidate


def _impact_level(text: str, claim_type: ClaimType) -> ImpactLevel:
    """Determine impact from claim type and content signals."""
    lowered = text.lower()
    if any(kw in lowered for kw in [
        "critical", "breaking", "security", "data loss", "production",
        "must", "never", "always", "guardrail",
    ]):
        return ImpactLevel.critical
    if claim_type in (ClaimType.rule, ClaimType.decision):
        return ImpactLevel.high
    if any(kw in lowered for kw in [
        "architecture", "pipeline", "workflow", "milestone",
        "pattern", "root cause", "lesson",
    ]):
        return ImpactLevel.high
    if claim_type == ClaimType.bug_history:
        return ImpactLevel.medium
    return ImpactLevel.medium


def _compute_confidence(
    text: str,
    role: str,
    claim_type: ClaimType,
    authority: AuthorityBasis,
    durability: DurabilityClass,
    is_atomic: bool,
    has_speculation: bool,
    has_reasoning: bool,
) -> float:
    """Compute a dynamic confidence score (0.0 - 1.0) based on structural markers."""
    score = 0.50  # Baseline

    # Authority boost
    if authority in (AuthorityBasis.user_explicit, AuthorityBasis.developer_explicit):
        score += 0.20
    elif authority == AuthorityBasis.system_tool_verified:
        score += 0.15
    elif authority == AuthorityBasis.assistant_inferred:
        score += 0.05

    # Structural keyword boost
    lowered = text.lower()
    if any(kw in lowered for kw in _STRUCTURAL_KEYWORDS["high"]):
        score += 0.15
    elif any(kw in lowered for kw in _STRUCTURAL_KEYWORDS["medium"]):
        score += 0.08

    # Claim type boost — rules and decisions are inherently higher confidence
    if claim_type in (ClaimType.rule, ClaimType.decision):
        score += 0.08
    elif claim_type == ClaimType.implementation_pattern:
        score += 0.05

    # Durability boost
    if durability == DurabilityClass.durable_eligible:
        score += 0.05

    # Atomicity bonus
    word_count = len(text.split())
    if is_atomic and 5 <= word_count <= 40:
        score += 0.05

    # Penalties
    if has_speculation:
        score -= 0.15
    if has_reasoning:
        score -= 0.10
    if word_count > 60:
        score -= 0.05  # Verbose
    if word_count < 5:
        score -= 0.10  # Too terse

    return round(max(0.0, min(1.0, score)), 3)


# ---------------------------------------------------------------------------
# Main extraction entry point
# ---------------------------------------------------------------------------

def extract_structured_candidates(
    text: str,
    *,
    source_agent: str,
    source_session: str,
    source_chunk: str,
) -> list[StructuredMemoryCandidate]:
    """Extract structured memory candidates from a dehydrated transcript chunk.

    Parses JSON message envelopes, extracts text content, splits into atomic
    claims, and scores each with dynamic confidence/importance.
    """
    candidates: list[StructuredMemoryCandidate] = []
    seen_hashes: set[str] = set()

    for raw_line in text.splitlines():
        raw_line = raw_line.strip()
        if not raw_line:
            continue

        # Step 1: Extract text from JSON message envelope
        text_segments = _extract_text_from_message(raw_line)
        if not text_segments:
            continue

        for segment in text_segments:
            seg_text = segment["text"]
            role = segment["role"]
            message_id = segment.get("message_id", "")

            # Step 2: Split into atomic claims
            claims = _split_into_claims(seg_text)

            for claim_text in claims:
                # Step 3: Filter noise
                if _is_noise(claim_text):
                    continue

                # Deduplicate within this chunk
                claim_hash = hashlib.sha1(
                    f"{source_chunk}|{claim_text}".encode("utf-8")
                ).hexdigest()[:12]
                if claim_hash in seen_hashes:
                    continue
                seen_hashes.add(claim_hash)

                # Step 4: Classify
                ct = _claim_type(claim_text)
                authority = _authority_basis(claim_text, role)
                durability = _durability_class(claim_text, ct)
                impact = _impact_level(claim_text, ct)

                # Pre-check for speculation/reasoning (needed for scoring)
                lowered = claim_text.lower()
                has_speculation = any(m in lowered for m in [
                    "maybe", "might", "could", "perhaps", "probably",
                ])
                has_reasoning = any(m in lowered for m in [
                    "let's think", "step by step", "i think", "i suspect",
                ])
                is_atomic = len(claim_text.split()) <= 80

                # Step 5: Dynamic confidence scoring
                confidence = _compute_confidence(
                    claim_text, role, ct, authority, durability,
                    is_atomic, has_speculation, has_reasoning,
                )

                # Step 6: Build scope envelope
                scope = build_context_scope({
                    "text": claim_text,
                    "source_session": source_session,
                    "scope": "stable",
                })

                candidate = StructuredMemoryCandidate(
                    candidate_id=f"cand-{claim_hash}",
                    source_event_ids=[message_id] if message_id else [],
                    source_agent=source_agent,
                    source_session=source_session,
                    source_chunk=source_chunk,
                    claim_text=claim_text,
                    claim_type=ct,
                    authority_basis=authority,
                    durability_class=durability,
                    impact_level=impact,
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
                    extractor_confidence=confidence,
                )
                candidates.append(validate_candidate(candidate))

    return candidates
