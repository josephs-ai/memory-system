"""
extraction_quality_gate.py — Precision-first extraction layer.

Phase 2, Item 4: Build a precision-first extraction layer.

The fastest way to ruin a memory system is to remember too much garbage.
This module enforces strict quality standards before any candidate enters
the memory pipeline.

Responsibilities:
1. Multi-signal quality scoring (not just keyword heuristics)
2. Confidence decay for weakly supported items
3. Evidence/rationale requirement enforcement
4. Source provenance validation
5. Atomicity and specificity checks
6. Extraction precision metrics

Design principles:
- Conservative by default — reject ambiguous items
- Every rejection has an explicit reason
- Scores are deterministic (no LLM calls)
- Integrates with existing StructuredMemoryCandidate schema
"""
from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

LOGGER = logging.getLogger("openclaw.extraction_quality_gate")


# ---------------------------------------------------------------------------
# Quality dimensions
# ---------------------------------------------------------------------------

class QualityDimension(str, Enum):
    ATOMICITY = "atomicity"
    SPECIFICITY = "specificity"
    AUTHORITY = "authority"
    EVIDENCE = "evidence"
    NOVELTY = "novelty"
    CLARITY = "clarity"
    ACTIONABILITY = "actionability"


@dataclass
class QualityScore:
    """Score for a single quality dimension."""
    dimension: QualityDimension
    score: float  # 0.0 to 1.0
    reason: str
    weight: float = 1.0

    def weighted(self) -> float:
        return self.score * self.weight


@dataclass
class ExtractionVerdict:
    """Full quality assessment for a memory candidate."""
    candidate_id: str
    overall_score: float
    passed: bool
    tier: str  # "high_quality", "acceptable", "needs_review", "rejected"
    dimension_scores: list[QualityScore] = field(default_factory=list)
    rejection_reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    evidence_present: bool = False
    provenance_valid: bool = False
    confidence_after_decay: float = 0.0

    def to_dict(self) -> dict:
        return {
            "candidate_id": self.candidate_id,
            "overall_score": round(self.overall_score, 4),
            "passed": self.passed,
            "tier": self.tier,
            "dimension_scores": [
                {"dimension": s.dimension.value, "score": round(s.score, 4), "reason": s.reason}
                for s in self.dimension_scores
            ],
            "rejection_reasons": self.rejection_reasons,
            "warnings": self.warnings,
            "evidence_present": self.evidence_present,
            "provenance_valid": self.provenance_valid,
            "confidence_after_decay": round(self.confidence_after_decay, 4),
        }


# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

# Minimum overall score to pass
PASS_THRESHOLD = 0.55
HIGH_QUALITY_THRESHOLD = 0.80
ACCEPTABLE_THRESHOLD = 0.55

# Per-dimension minimum scores (below = hard rejection)
HARD_MINIMUMS = {
    QualityDimension.ATOMICITY: 0.3,
    QualityDimension.SPECIFICITY: 0.2,
    QualityDimension.AUTHORITY: 0.1,
}

# Dimension weights for overall score
DIMENSION_WEIGHTS = {
    QualityDimension.ATOMICITY: 1.5,
    QualityDimension.SPECIFICITY: 1.3,
    QualityDimension.AUTHORITY: 1.2,
    QualityDimension.EVIDENCE: 1.0,
    QualityDimension.NOVELTY: 0.8,
    QualityDimension.CLARITY: 1.0,
    QualityDimension.ACTIONABILITY: 0.7,
}

# Confidence decay parameters
CONFIDENCE_DECAY_HALF_LIFE_DAYS = 30
CONFIDENCE_FLOOR = 0.1
CONFIDENCE_BOOST_PER_MENTION = 0.05
CONFIDENCE_MAX = 0.99


# ---------------------------------------------------------------------------
# Scoring functions
# ---------------------------------------------------------------------------

# Patterns that indicate low-quality candidates
_VAGUE_PATTERNS = re.compile(
    r'\b(something|stuff|things|various|etc|some kind of|sort of|a lot|'
    r'certain|particular|specific|in general|overall|basically|generally)\b',
    re.IGNORECASE,
)

_FILLER_PATTERNS = re.compile(
    r'\b(actually|honestly|obviously|clearly|definitely|absolutely|'
    r'literally|essentially|fundamentally|interestingly)\b',
    re.IGNORECASE,
)

_REASONING_PATTERNS = re.compile(
    r'\b(let\'s think|step by step|i think|i believe|i suspect|'
    r'brainstorm|consider|perhaps|wondering|seems like|appears to)\b',
    re.IGNORECASE,
)

_SPECULATION_PATTERNS = re.compile(
    r'\b(maybe|might|could|perhaps|probably|possibly|'
    r'not sure|uncertain|unclear|hypothetically)\b',
    re.IGNORECASE,
)

_QUESTION_PATTERN = re.compile(r'\?\s*$')

_TEMPORAL_SELF_REF = re.compile(
    r'\b(today|yesterday|just now|earlier|recently|this morning|tonight|'
    r'right now|at the moment|currently working on)\b',
    re.IGNORECASE,
)

# Patterns that indicate high-quality candidates
_DECISION_MARKERS = re.compile(
    r'\b(decided|chose|selected|will use|switched to|migrated to|'
    r'settled on|adopted|committed to|standard is|rule is|policy is)\b',
    re.IGNORECASE,
)

_FACT_MARKERS = re.compile(
    r'\b(is located at|runs on|port \d+|version \d|uses? |requires? |'
    r'depends on|configured with|set to|equals?|password is|key is)\b',
    re.IGNORECASE,
)

_PREFERENCE_MARKERS = re.compile(
    r'\b(prefer|always use|never use|like to|dislike|avoid|'
    r'want|don\'t want|should always|should never)\b',
    re.IGNORECASE,
)

EXPLICIT_AUTHORITY_BASES = {
    "user_explicit", "developer_explicit", "tester_explicit", "system_tool_verified",
}


def score_atomicity(text: str) -> QualityScore:
    """Score how atomic/self-contained the claim is."""
    words = text.split()
    word_count = len(words)
    sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
    sentence_count = len(sentences)

    score = 1.0
    reason_parts = []

    # Too long = not atomic
    if word_count > 60:
        score -= 0.4
        reason_parts.append(f"too_long ({word_count} words)")
    elif word_count > 40:
        score -= 0.2
        reason_parts.append(f"somewhat_long ({word_count} words)")

    # Too short = probably not useful
    if word_count < 3:
        score -= 0.5
        reason_parts.append("too_short")

    # Multiple sentences = compound claim
    if sentence_count > 2:
        score -= 0.3
        reason_parts.append(f"compound_claim ({sentence_count} sentences)")

    # Contains lists or enumerations
    if re.search(r'\d\.\s|\n-\s|\n\*\s', text):
        score -= 0.2
        reason_parts.append("contains_list")

    score = max(0.0, min(1.0, score))
    reason = "; ".join(reason_parts) if reason_parts else "atomic"
    return QualityScore(QualityDimension.ATOMICITY, score, reason, DIMENSION_WEIGHTS[QualityDimension.ATOMICITY])


def score_specificity(text: str) -> QualityScore:
    """Score how specific and concrete the claim is."""
    text_lower = text.lower()
    score = 0.7  # Start at decent baseline

    reason_parts = []

    # Vague language penalization
    vague_matches = _VAGUE_PATTERNS.findall(text_lower)
    if vague_matches:
        penalty = min(0.4, len(vague_matches) * 0.1)
        score -= penalty
        reason_parts.append(f"vague_language ({len(vague_matches)} markers)")

    # Filler word penalization (lighter)
    filler_matches = _FILLER_PATTERNS.findall(text_lower)
    if filler_matches:
        penalty = min(0.2, len(filler_matches) * 0.05)
        score -= penalty
        reason_parts.append(f"filler_words ({len(filler_matches)})")

    # Concrete markers boost
    has_numbers = bool(re.search(r'\d+', text))
    has_names = bool(re.search(r'[A-Z][a-z]+', text))
    has_paths = bool(re.search(r'[/\\][\w./-]+', text))
    has_code = bool(re.search(r'`[^`]+`|[\w_]+\(|[\w_]+\.[\w_]+', text))

    concrete_count = sum([has_numbers, has_names, has_paths, has_code])
    if concrete_count >= 2:
        score += 0.2
        reason_parts.append("highly_concrete")
    elif concrete_count == 1:
        score += 0.1
        reason_parts.append("somewhat_concrete")

    # Temporal self-references are bad (not durable)
    if _TEMPORAL_SELF_REF.search(text):
        score -= 0.3
        reason_parts.append("temporal_self_reference")

    score = max(0.0, min(1.0, score))
    reason = "; ".join(reason_parts) if reason_parts else "specific"
    return QualityScore(QualityDimension.SPECIFICITY, score, reason, DIMENSION_WEIGHTS[QualityDimension.SPECIFICITY])


def score_authority(item: dict) -> QualityScore:
    """Score the authority basis of the claim."""
    authority = str(item.get("authority_basis", "unknown")).lower()
    role = str(item.get("role", "unknown")).lower()
    str(item.get("source_agent", "unknown")).lower()

    score = 0.5  # Default moderate
    reason_parts = []

    if authority in EXPLICIT_AUTHORITY_BASES:
        score = 0.95
        reason_parts.append(f"explicit_authority ({authority})")
    elif authority == "assistant_inferred":
        score = 0.35
        reason_parts.append("assistant_inferred (low trust)")
    elif authority == "summary_inferred":
        score = 0.25
        reason_parts.append("summary_inferred (very low trust)")
    elif authority == "unknown":
        score = 0.3
        reason_parts.append("unknown_authority")

    # User-role text has higher authority than assistant-generated
    if role == "user":
        score = min(1.0, score + 0.15)
        reason_parts.append("user_role_boost")
    elif role == "system":
        score = min(1.0, score + 0.1)
        reason_parts.append("system_role_boost")

    score = max(0.0, min(1.0, score))
    reason = "; ".join(reason_parts) if reason_parts else "moderate_authority"
    return QualityScore(QualityDimension.AUTHORITY, score, reason, DIMENSION_WEIGHTS[QualityDimension.AUTHORITY])


def score_evidence(item: dict) -> QualityScore:
    """Score whether the candidate has supporting evidence/rationale."""
    text = str(item.get("text", "") or item.get("claim_text", ""))
    source_chunk = str(item.get("source_chunk", ""))
    source_session = str(item.get("source_session", ""))
    str(item.get("source_agent", ""))

    score = 0.3  # Default low — evidence should be earned
    reason_parts = []

    # Has source provenance
    has_provenance = bool(source_chunk and source_chunk != "unknown"
                          and source_session and source_session != "unknown")
    if has_provenance:
        score += 0.3
        reason_parts.append("has_provenance")

    # Decision markers suggest deliberate choice (self-documenting)
    if _DECISION_MARKERS.search(text):
        score += 0.15
        reason_parts.append("decision_marker")

    # Fact markers suggest verifiable claim
    if _FACT_MARKERS.search(text):
        score += 0.1
        reason_parts.append("fact_marker")

    # Contains a "because" or "reason:" → has rationale
    if re.search(r'\bbecause\b|\breason[:\s]|\bdue to\b|\bsince\b', text, re.IGNORECASE):
        score += 0.15
        reason_parts.append("has_rationale")

    score = max(0.0, min(1.0, score))
    reason = "; ".join(reason_parts) if reason_parts else "no_evidence"
    return QualityScore(QualityDimension.EVIDENCE, score, reason, DIMENSION_WEIGHTS[QualityDimension.EVIDENCE])


def score_clarity(text: str) -> QualityScore:
    """Score how clear and unambiguous the text is."""
    score = 0.8
    reason_parts = []

    # Reasoning/speculation = unclear for durable storage
    if _REASONING_PATTERNS.search(text):
        score -= 0.35
        reason_parts.append("contains_reasoning")

    if _SPECULATION_PATTERNS.search(text):
        score -= 0.3
        reason_parts.append("contains_speculation")

    # Questions are not statements of truth
    if _QUESTION_PATTERN.search(text):
        score -= 0.4
        reason_parts.append("is_question")

    # Ambiguous pronouns without clear referent
    pronoun_density = len(re.findall(r'\b(it|this|that|they|those|these)\b', text, re.IGNORECASE))
    word_count = max(len(text.split()), 1)
    if pronoun_density / word_count > 0.15:
        score -= 0.2
        reason_parts.append(f"high_pronoun_density ({pronoun_density}/{word_count})")

    score = max(0.0, min(1.0, score))
    reason = "; ".join(reason_parts) if reason_parts else "clear"
    return QualityScore(QualityDimension.CLARITY, score, reason, DIMENSION_WEIGHTS[QualityDimension.CLARITY])


def score_actionability(item: dict) -> QualityScore:
    """Score whether the memory is actionable/useful for future retrieval."""
    text = str(item.get("text", "") or item.get("claim_text", ""))
    claim_type = str(item.get("claim_type", "") or item.get("memory_type", ""))

    score = 0.5
    reason_parts = []

    # High-value claim types
    high_value = {"decision", "preference", "rule", "bug_history", "learned_fix", "implementation_pattern"}
    if claim_type in high_value:
        score += 0.3
        reason_parts.append(f"high_value_type ({claim_type})")

    # Low-value claim types
    low_value = {"summary_only", "open_question", "ephemeral"}
    if claim_type in low_value:
        score -= 0.3
        reason_parts.append(f"low_value_type ({claim_type})")

    # Preference/decision markers boost actionability
    if _PREFERENCE_MARKERS.search(text):
        score += 0.15
        reason_parts.append("preference_marker")

    score = max(0.0, min(1.0, score))
    reason = "; ".join(reason_parts) if reason_parts else "moderate_actionability"
    return QualityScore(QualityDimension.ACTIONABILITY, score, reason, DIMENSION_WEIGHTS[QualityDimension.ACTIONABILITY])


# ---------------------------------------------------------------------------
# Confidence decay
# ---------------------------------------------------------------------------

def compute_confidence_decay(
    base_confidence: float,
    first_seen: datetime | None,
    last_confirmed: datetime | None,
    mention_count: int = 1,
    now: datetime | None = None,
) -> float:
    """
    Apply logarithmic confidence decay based on time since last confirmation.

    - Decay is gentle (log-based, not exponential)
    - Multiple mentions boost confidence
    - Recent confirmation resets the decay clock
    - Confidence never drops below CONFIDENCE_FLOOR
    """
    if now is None:
        now = datetime.now(timezone.utc)

    reference_time = last_confirmed or first_seen
    if reference_time is None:
        return max(CONFIDENCE_FLOOR, base_confidence * 0.5)

    # Ensure timezone-aware comparison
    if reference_time.tzinfo is None:
        reference_time = reference_time.replace(tzinfo=timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)

    days_elapsed = max(0, (now - reference_time).total_seconds() / 86400)

    # Logarithmic decay: confidence * (1 / (1 + ln(1 + days/half_life)))
    decay_factor = 1.0 / (1.0 + math.log(1.0 + days_elapsed / CONFIDENCE_DECAY_HALF_LIFE_DAYS))

    # Mention count boost (diminishing returns)
    mention_boost = min(
        CONFIDENCE_BOOST_PER_MENTION * math.log(1 + mention_count),
        0.2,  # cap
    )

    decayed = base_confidence * decay_factor + mention_boost
    return max(CONFIDENCE_FLOOR, min(CONFIDENCE_MAX, decayed))


# ---------------------------------------------------------------------------
# Main gate function
# ---------------------------------------------------------------------------

def evaluate_candidate(item: dict) -> ExtractionVerdict:
    """
    Run full quality evaluation on a memory candidate.

    Args:
        item: dict with at least 'text' or 'claim_text', plus optional
              'authority_basis', 'source_agent', 'source_session', 'source_chunk',
              'claim_type'/'memory_type', 'confidence', 'first_seen', 'last_confirmed',
              'mention_count', 'role'.

    Returns:
        ExtractionVerdict with pass/fail, tier, dimension scores, and reasons.
    """
    candidate_id = str(item.get("candidate_id", "") or item.get("id", "") or "unknown")
    text = str(item.get("text", "") or item.get("claim_text", ""))

    # Immediate rejection for empty text
    if not text.strip():
        return ExtractionVerdict(
            candidate_id=candidate_id,
            overall_score=0.0,
            passed=False,
            tier="rejected",
            rejection_reasons=["empty_text"],
        )

    # Score all dimensions
    scores = [
        score_atomicity(text),
        score_specificity(text),
        score_authority(item),
        score_evidence(item),
        score_clarity(text),
        score_actionability(item),
    ]

    # Check hard minimums
    rejection_reasons = []
    for s in scores:
        if s.dimension in HARD_MINIMUMS:
            if s.score < HARD_MINIMUMS[s.dimension]:
                rejection_reasons.append(f"{s.dimension.value}_below_minimum ({s.score:.2f} < {HARD_MINIMUMS[s.dimension]})")

    # Compute weighted overall score
    total_weight = sum(s.weight for s in scores)
    overall = sum(s.weighted() for s in scores) / max(total_weight, 1e-9)

    # Confidence decay
    confidence = float(item.get("confidence", 0.7) or 0.7)
    first_seen = _parse_dt(item.get("first_seen"))
    last_confirmed = _parse_dt(item.get("last_confirmed"))
    mention_count = int(item.get("mention_count", 1) or 1)
    decayed_confidence = compute_confidence_decay(confidence, first_seen, last_confirmed, mention_count)

    # Evidence presence
    evidence_present = any(s.dimension == QualityDimension.EVIDENCE and s.score > 0.5 for s in scores)

    # Provenance validation
    source_chunk = str(item.get("source_chunk", ""))
    source_session = str(item.get("source_session", ""))
    provenance_valid = bool(source_chunk and source_chunk != "unknown"
                            and source_session and source_session != "unknown")

    # Determine tier
    if rejection_reasons or overall < ACCEPTABLE_THRESHOLD:
        tier = "rejected"
        passed = False
        if overall < ACCEPTABLE_THRESHOLD and not rejection_reasons:
            rejection_reasons.append(f"overall_score_too_low ({overall:.3f} < {ACCEPTABLE_THRESHOLD})")
    elif overall >= HIGH_QUALITY_THRESHOLD:
        tier = "high_quality"
        passed = True
    elif overall >= ACCEPTABLE_THRESHOLD:
        tier = "acceptable"
        passed = True
    else:
        tier = "needs_review"
        passed = False

    # Collect warnings
    warnings = []
    if not evidence_present:
        warnings.append("no_evidence_attached")
    if not provenance_valid:
        warnings.append("missing_provenance")
    if decayed_confidence < 0.3:
        warnings.append(f"low_confidence_after_decay ({decayed_confidence:.2f})")

    return ExtractionVerdict(
        candidate_id=candidate_id,
        overall_score=overall,
        passed=passed,
        tier=tier,
        dimension_scores=scores,
        rejection_reasons=rejection_reasons,
        warnings=warnings,
        evidence_present=evidence_present,
        provenance_valid=provenance_valid,
        confidence_after_decay=decayed_confidence,
    )


def _parse_dt(val: Any) -> datetime | None:
    """Parse a datetime value from various formats."""
    if val is None:
        return None
    if isinstance(val, datetime):
        return val
    try:
        s = str(val).replace("Z", "+00:00")
        return datetime.fromisoformat(s)
    except (ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# Batch evaluation
# ---------------------------------------------------------------------------

def evaluate_batch(items: list[dict]) -> dict:
    """
    Evaluate a batch of candidates and return summary statistics.

    Returns:
        {
            "total": int,
            "passed": int,
            "rejected": int,
            "pass_rate": float,
            "tier_counts": {"high_quality": int, "acceptable": int, "rejected": int},
            "avg_score": float,
            "common_rejections": {reason: count},
            "verdicts": [ExtractionVerdict.to_dict(), ...],
        }
    """
    verdicts = [evaluate_candidate(item) for item in items]

    passed = sum(1 for v in verdicts if v.passed)
    rejected = sum(1 for v in verdicts if not v.passed)
    tier_counts: dict[str, int] = {}
    rejection_counts: dict[str, int] = {}
    scores = []

    for v in verdicts:
        tier_counts[v.tier] = tier_counts.get(v.tier, 0) + 1
        scores.append(v.overall_score)
        for reason in v.rejection_reasons:
            # Normalize reason for counting
            base_reason = reason.split("(")[0].strip()
            rejection_counts[base_reason] = rejection_counts.get(base_reason, 0) + 1

    total = len(verdicts)
    avg_score = sum(scores) / max(total, 1)

    return {
        "total": total,
        "passed": passed,
        "rejected": rejected,
        "pass_rate": round(passed / max(total, 1), 4),
        "tier_counts": tier_counts,
        "avg_score": round(avg_score, 4),
        "common_rejections": dict(sorted(rejection_counts.items(), key=lambda x: -x[1])),
        "verdicts": [v.to_dict() for v in verdicts],
    }
