"""
verification_pipeline.py — Memory verification and confirmation system.

Phase 2, Item 5: Add memory verification / confirmation pipelines.

Before important items become durable, this module enforces:
1. Repeated mention tracking (corroboration)
2. Structured confidence thresholds by claim type
3. Confirmation requirements for sensitive categories
4. Promotion eligibility computation

Design principles:
- Sensitive categories (preferences, personal facts, decisions, architectural truths)
  require higher evidence thresholds
- Corroboration = same claim appearing across multiple sessions/contexts
- No LLM calls — deterministic scoring from DB state
- Integrates with existing memory_items table and promotion pipeline
"""
from __future__ import annotations

import hashlib
import logging
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

LOGGER = logging.getLogger("openclaw.verification_pipeline")

DB_DSN = os.environ.get("OPENCLAW_MEMORY_DSN", "dbname=openclaw_memory")


# ---------------------------------------------------------------------------
# Sensitivity categories and thresholds
# ---------------------------------------------------------------------------

class SensitivityCategory(str, Enum):
    """How sensitive a memory type is — higher sensitivity = stricter promotion."""
    PERSONAL_FACT = "personal_fact"
    PREFERENCE = "preference"
    ARCHITECTURAL_TRUTH = "architectural_truth"
    LONG_TERM_DECISION = "long_term_decision"
    BUG_FIX = "bug_fix"
    IMPLEMENTATION_PATTERN = "implementation_pattern"
    GENERAL_FACT = "general_fact"
    EPHEMERAL = "ephemeral"


@dataclass
class PromotionRequirements:
    """What's required before a candidate can become durable."""
    min_mentions: int = 1
    min_confidence: float = 0.5
    min_corroboration_sessions: int = 1
    min_age_hours: int = 0
    requires_user_confirmation: bool = False
    requires_explicit_authority: bool = False
    max_contradiction_count: int = 0  # 0 = no contradictions allowed


# Per-category promotion requirements
CATEGORY_REQUIREMENTS: dict[SensitivityCategory, PromotionRequirements] = {
    SensitivityCategory.PERSONAL_FACT: PromotionRequirements(
        min_mentions=2,
        min_confidence=0.75,
        min_corroboration_sessions=1,
        requires_user_confirmation=False,  # Can be auto-promoted if mentioned twice
        requires_explicit_authority=True,
    ),
    SensitivityCategory.PREFERENCE: PromotionRequirements(
        min_mentions=1,
        min_confidence=0.70,
        min_corroboration_sessions=1,
        requires_explicit_authority=True,
    ),
    SensitivityCategory.ARCHITECTURAL_TRUTH: PromotionRequirements(
        min_mentions=2,
        min_confidence=0.80,
        min_corroboration_sessions=2,
        min_age_hours=1,
        max_contradiction_count=0,
    ),
    SensitivityCategory.LONG_TERM_DECISION: PromotionRequirements(
        min_mentions=1,
        min_confidence=0.80,
        requires_explicit_authority=True,
        max_contradiction_count=0,
    ),
    SensitivityCategory.BUG_FIX: PromotionRequirements(
        min_mentions=1,
        min_confidence=0.65,
    ),
    SensitivityCategory.IMPLEMENTATION_PATTERN: PromotionRequirements(
        min_mentions=1,
        min_confidence=0.60,
    ),
    SensitivityCategory.GENERAL_FACT: PromotionRequirements(
        min_mentions=1,
        min_confidence=0.50,
    ),
    SensitivityCategory.EPHEMERAL: PromotionRequirements(
        min_mentions=999,  # Effectively never promoted
        min_confidence=1.0,
    ),
}


# Mapping from claim_type/memory_type to sensitivity category
TYPE_TO_CATEGORY: dict[str, SensitivityCategory] = {
    "preference": SensitivityCategory.PREFERENCE,
    "decision": SensitivityCategory.LONG_TERM_DECISION,
    "rule": SensitivityCategory.ARCHITECTURAL_TRUTH,
    "fact": SensitivityCategory.GENERAL_FACT,
    "implementation_pattern": SensitivityCategory.IMPLEMENTATION_PATTERN,
    "bug_history": SensitivityCategory.BUG_FIX,
    "learned_fix": SensitivityCategory.BUG_FIX,
    "open_question": SensitivityCategory.EPHEMERAL,
    "summary_only": SensitivityCategory.EPHEMERAL,
}

EXPLICIT_AUTHORITIES = {
    "user_explicit", "developer_explicit", "tester_explicit", "system_tool_verified",
}


# ---------------------------------------------------------------------------
# Corroboration tracking
# ---------------------------------------------------------------------------

@dataclass
class CorroborationRecord:
    """Tracks how many times a claim has been mentioned/confirmed."""
    item_id: str
    mention_count: int = 0
    distinct_sessions: int = 0
    distinct_agents: int = 0
    first_seen: datetime | None = None
    last_mentioned: datetime | None = None
    session_ids: list[str] = field(default_factory=list)
    agent_ids: list[str] = field(default_factory=list)

    def age_hours(self, now: datetime | None = None) -> float:
        if not self.first_seen:
            return 0.0
        if now is None:
            now = datetime.now(timezone.utc)
        return max(0, (now - self.first_seen).total_seconds() / 3600)


def compute_content_hash(text: str) -> str:
    """Compute a normalized content hash for deduplication and corroboration."""
    normalized = re.sub(r'\s+', ' ', text.lower().strip())
    # Remove punctuation for fuzzy matching
    normalized = re.sub(r'[^\w\s]', '', normalized)
    return hashlib.sha256(normalized.encode()).hexdigest()[:16]


def get_corroboration(item_id: str, text: str, conn: Any = None) -> CorroborationRecord:
    """
    Query the database for corroboration evidence for this claim.

    Looks for:
    1. Exact or near-duplicate items (same content hash)
    2. Items with the same entity+property+value slot
    3. Items from different sessions mentioning the same thing
    """
    compute_content_hash(text)
    record = CorroborationRecord(item_id=item_id)

    should_close = False
    if conn is None:
        import psycopg
        conn = psycopg.connect(DB_DSN)
        should_close = True

    try:
        with conn.cursor() as cur:
            # Find items with similar content (using text similarity or exact entity/property match)
            cur.execute("""
                SELECT id, source_session, source_agent, first_seen, last_confirmed
                FROM memory_items
                WHERE id != %s
                  AND status IN ('active', 'candidate', 'pending')
                  AND (
                    -- Same entity+property slot
                    (entity IS NOT NULL AND property IS NOT NULL
                     AND entity = (SELECT entity FROM memory_items WHERE id = %s)
                     AND property = (SELECT property FROM memory_items WHERE id = %s))
                    OR
                    -- Similar text (trigram would be better, but this works for exact matches)
                    LOWER(text) = LOWER(%s)
                  )
                ORDER BY first_seen DESC
                LIMIT 50
            """, (item_id, item_id, item_id, text.strip()))

            rows = cur.fetchall()

            sessions = set()
            agents = set()
            earliest = None
            latest = None

            for row_id, session, agent, first_seen, last_confirmed in rows:
                if session:
                    sessions.add(session)
                if agent:
                    agents.add(agent)
                ts = last_confirmed or first_seen
                if ts:
                    if earliest is None or ts < earliest:
                        earliest = ts
                    if latest is None or ts > latest:
                        latest = ts

            # Include the item itself
            cur.execute("""
                SELECT source_session, source_agent, first_seen
                FROM memory_items WHERE id = %s
            """, (item_id,))
            self_row = cur.fetchone()
            if self_row:
                if self_row[0]:
                    sessions.add(self_row[0])
                if self_row[1]:
                    agents.add(self_row[1])
                if self_row[2]:
                    if earliest is None or self_row[2] < earliest:
                        earliest = self_row[2]

            record.mention_count = len(rows) + 1  # +1 for the item itself
            record.distinct_sessions = len(sessions)
            record.distinct_agents = len(agents)
            record.first_seen = earliest
            record.last_mentioned = latest
            record.session_ids = list(sessions)
            record.agent_ids = list(agents)

    finally:
        if should_close:
            conn.close()

    return record


# ---------------------------------------------------------------------------
# Promotion eligibility check
# ---------------------------------------------------------------------------

@dataclass
class PromotionEligibility:
    """Result of checking whether an item is eligible for promotion."""
    item_id: str
    eligible: bool
    category: SensitivityCategory
    requirements: PromotionRequirements
    corroboration: CorroborationRecord
    checks_passed: list[str] = field(default_factory=list)
    checks_failed: list[str] = field(default_factory=list)
    confidence_sufficient: bool = False
    mentions_sufficient: bool = False
    corroboration_sufficient: bool = False
    age_sufficient: bool = False
    authority_sufficient: bool = False
    contradiction_free: bool = True

    def to_dict(self) -> dict:
        return {
            "item_id": self.item_id,
            "eligible": self.eligible,
            "category": self.category.value,
            "checks_passed": self.checks_passed,
            "checks_failed": self.checks_failed,
            "mention_count": self.corroboration.mention_count,
            "distinct_sessions": self.corroboration.distinct_sessions,
            "confidence_sufficient": self.confidence_sufficient,
            "mentions_sufficient": self.mentions_sufficient,
            "corroboration_sufficient": self.corroboration_sufficient,
            "age_sufficient": self.age_sufficient,
            "authority_sufficient": self.authority_sufficient,
            "contradiction_free": self.contradiction_free,
        }


def check_promotion_eligibility(
    item: dict,
    corroboration: CorroborationRecord | None = None,
    contradiction_count: int = 0,
    conn: Any = None,
) -> PromotionEligibility:
    """
    Check whether a memory item meets all requirements for promotion to durable.

    Args:
        item: memory item dict (from DB or candidate)
        corroboration: pre-computed corroboration record (or will be computed)
        contradiction_count: number of active contradictions for this item
        conn: optional psycopg connection

    Returns:
        PromotionEligibility with detailed check results.
    """
    item_id = str(item.get("id", "") or item.get("candidate_id", ""))
    text = str(item.get("text", "") or item.get("claim_text", ""))
    claim_type = str(item.get("claim_type", "") or item.get("memory_type", "fact"))
    confidence = float(item.get("confidence", 0.5) or 0.5)
    authority = str(item.get("authority_basis", "unknown")).lower()

    # Determine category
    category = TYPE_TO_CATEGORY.get(claim_type, SensitivityCategory.GENERAL_FACT)
    reqs = CATEGORY_REQUIREMENTS[category]

    # Get corroboration if not provided
    if corroboration is None:
        corroboration = get_corroboration(item_id, text, conn)

    checks_passed = []
    checks_failed = []

    # Check 1: Confidence threshold
    confidence_ok = confidence >= reqs.min_confidence
    if confidence_ok:
        checks_passed.append(f"confidence ({confidence:.2f} >= {reqs.min_confidence})")
    else:
        checks_failed.append(f"confidence ({confidence:.2f} < {reqs.min_confidence})")

    # Check 2: Mention count
    mentions_ok = corroboration.mention_count >= reqs.min_mentions
    if mentions_ok:
        checks_passed.append(f"mentions ({corroboration.mention_count} >= {reqs.min_mentions})")
    else:
        checks_failed.append(f"mentions ({corroboration.mention_count} < {reqs.min_mentions})")

    # Check 3: Corroboration (distinct sessions)
    corroboration_ok = corroboration.distinct_sessions >= reqs.min_corroboration_sessions
    if corroboration_ok:
        checks_passed.append(f"sessions ({corroboration.distinct_sessions} >= {reqs.min_corroboration_sessions})")
    else:
        checks_failed.append(f"sessions ({corroboration.distinct_sessions} < {reqs.min_corroboration_sessions})")

    # Check 4: Minimum age
    age_hours = corroboration.age_hours()
    age_ok = age_hours >= reqs.min_age_hours
    if age_ok:
        checks_passed.append(f"age ({age_hours:.1f}h >= {reqs.min_age_hours}h)")
    else:
        checks_failed.append(f"age ({age_hours:.1f}h < {reqs.min_age_hours}h)")

    # Check 5: Authority
    if reqs.requires_explicit_authority:
        authority_ok = authority in EXPLICIT_AUTHORITIES
        if authority_ok:
            checks_passed.append(f"authority ({authority})")
        else:
            checks_failed.append(f"authority ({authority} not explicit)")
    else:
        authority_ok = True
        checks_passed.append("authority (not required)")

    # Check 6: Contradictions
    contradiction_ok = contradiction_count <= reqs.max_contradiction_count
    if contradiction_ok:
        checks_passed.append(f"contradictions ({contradiction_count} <= {reqs.max_contradiction_count})")
    else:
        checks_failed.append(f"contradictions ({contradiction_count} > {reqs.max_contradiction_count})")

    eligible = all([confidence_ok, mentions_ok, corroboration_ok, age_ok, authority_ok, contradiction_ok])

    return PromotionEligibility(
        item_id=item_id,
        eligible=eligible,
        category=category,
        requirements=reqs,
        corroboration=corroboration,
        checks_passed=checks_passed,
        checks_failed=checks_failed,
        confidence_sufficient=confidence_ok,
        mentions_sufficient=mentions_ok,
        corroboration_sufficient=corroboration_ok,
        age_sufficient=age_ok,
        authority_sufficient=authority_ok,
        contradiction_free=contradiction_ok,
    )


# ---------------------------------------------------------------------------
# Batch processing
# ---------------------------------------------------------------------------

def check_promotion_batch(items: list[dict], conn: Any = None) -> dict:
    """
    Check promotion eligibility for a batch of candidates.

    Returns summary stats and per-item results.
    """
    results = []
    for item in items:
        eligibility = check_promotion_eligibility(item, conn=conn)
        results.append(eligibility)

    eligible_count = sum(1 for r in results if r.eligible)
    total = len(results)

    # Failure breakdown
    failure_reasons: dict[str, int] = {}
    for r in results:
        for reason in r.checks_failed:
            base = reason.split("(")[0].strip()
            failure_reasons[base] = failure_reasons.get(base, 0) + 1

    return {
        "total": total,
        "eligible": eligible_count,
        "ineligible": total - eligible_count,
        "eligibility_rate": round(eligible_count / max(total, 1), 4),
        "common_failures": dict(sorted(failure_reasons.items(), key=lambda x: -x[1])),
        "results": [r.to_dict() for r in results],
    }
