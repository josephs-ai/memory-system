"""
The producer's vocabulary and the consumers' vocabularies must agree.

This is the bug class that quietly broke promotion for months. The extractor
emits DurabilityClass.durable_eligible; the scoring map only knew "durable".
The miss was invisible because an unmapped key does not raise -- it silently
takes the 0.5 unknown default. That pinned importance at 0.775 against a 0.85
gate, so nothing could ever be auto-promoted, while extraction kept running and
the review queue grew past 300,000 rows with no error anywhere.

Nothing checked that the two sides matched, so nothing reported it. These tests
are that check. They fail on the *addition* of an enum value too, which forces
a deliberate decision about new vocabulary instead of a silent default.
"""
from __future__ import annotations

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from memory_candidate_schema import ClaimType, DurabilityClass, ImpactLevel
from route_memory_items_batch import DURABILITY_SCORES, IMPACT_SCORES, normalize_item
from auto_promote_safe_items import PROMOTABLE_TYPES, has_identity, identity_key

# Types deliberately held back from auto-promotion. Listed explicitly so that
# adding a ClaimType fails the coverage test below until someone decides which
# side it belongs on.
HELD_FOR_REVIEW = {"bug_history", "open_question", "summary_only"}


def test_every_durability_class_is_scored():
    missing = {d.value for d in DurabilityClass} - set(DURABILITY_SCORES)
    assert not missing, f"DurabilityClass values with no score (silently 0.5): {missing}"


def test_every_impact_level_is_scored():
    missing = {i.value for i in ImpactLevel} - set(IMPACT_SCORES)
    assert not missing, f"ImpactLevel values with no score (silently 0.5): {missing}"


def test_every_claim_type_has_a_promotion_decision():
    undecided = {c.value for c in ClaimType} - PROMOTABLE_TYPES - HELD_FOR_REVIEW
    assert not undecided, f"ClaimType values with no promotion decision: {undecided}"


def test_durable_eligible_can_actually_clear_the_promotion_gate():
    """The regression itself: the best possible item must be promotable.

    If the most durable, most critical claim the extractor can emit cannot
    reach the importance gate, promotion is mathematically dead no matter what
    else is correct.
    """
    item = normalize_item(
        {
            "candidate_id": "x",
            "durability_class": DurabilityClass.durable_eligible.value,
            "impact_level": ImpactLevel.critical.value,
        }
    )
    assert item["importance"] >= 0.85, (
        f"best-case importance is {item['importance']}, below the 0.85 gate -- "
        "auto-promotion cannot happen for any item"
    )


def test_durability_ordering_is_monotonic():
    """More durable must never score lower than less durable."""
    order = ["ephemeral", "session", "candidate", "durable_eligible"]
    scores = [DURABILITY_SCORES[k] for k in order]
    assert scores == sorted(scores), f"durability scores not monotonic: {dict(zip(order, scores))}"


def test_different_items_never_share_an_identity_key():
    """Guards the dedup collision that deleted a 300k-row queue.

    No queued item, and 10,161 stored ones, have entity/property/value. Keying
    on that triple alone made every one of them compare equal, so unrelated
    rows were treated as the same item.
    """
    a = {"memory_type": "fact", "text": "the sky is blue"}
    b = {"memory_type": "fact", "text": "the build is broken"}
    assert identity_key(a) is not None
    assert identity_key(a) != identity_key(b), "unrelated items collapsed to one key"


def test_identical_text_is_deduped():
    a = {"memory_type": "fact", "text": "The Sky Is Blue"}
    b = {"memory_type": "fact", "text": "the sky is  blue "}
    assert identity_key(a) == identity_key(b), "dedup must ignore case and whitespace"


def test_text_key_cannot_collide_with_structured_key():
    structured = {"memory_type": "fact", "entity": "e", "property": "p", "value": "v"}
    textual = {"memory_type": "fact", "text": "e p v"}
    assert has_identity(structured)
    assert identity_key(structured) != identity_key(textual)


def test_unidentifiable_item_returns_none():
    """None means 'matches nothing' -- it must never be used as a key value."""
    assert identity_key({"memory_type": "fact"}) is None


# ---------------------------------------------------------------------------
# Prompt scaffolding must never be mistaken for a durable claim.
#
# Instructions addressed to an agent are imperative and declarative, so they
# score like rules. 40 of the first 123 promoted items (32%) were reviewer
# scaffolding of the form 'Return JSON only: {"reviewerId":...}', each carrying
# a unique digest so dedup could not collapse them.
# ---------------------------------------------------------------------------

from memory_validation_gate import prompt_noise_reasons
from auto_promote_safe_items import safe_to_promote


def _promotable(text):
    return {
        "scope": "stable",
        "confidence": 0.95,
        "importance": 0.9,
        "memory_type": "rule",
        "text": text,
    }


def test_reviewer_scaffolding_is_rejected():
    text = 'Return JSON only: {"reviewerId":"memory-reviewer","evidenceDigest":"87ef00e1b37c1770"}'
    assert prompt_noise_reasons(text)
    assert not safe_to_promote(_promotable(text), [])


def test_real_rules_still_promote():
    """The filter must not eat legitimate technical claims."""
    for text in (
        "Failure handling must be explicit (clean fail or explicit fallback path, tested).",
        "Helper payload must contain exactly one script artifact, exact requested name",
        "Slice 11A is approved as the first implementation area with this revised scope.",
    ):
        assert not prompt_noise_reasons(text), f"false positive on: {text}"
        assert safe_to_promote(_promotable(text), [])


def test_opaque_digests_are_flagged():
    """Digests defeat dedup, so identical scaffolding compounds forever."""
    assert "opaque_digest" in prompt_noise_reasons("evidence 3d560db1f0f0a367b2c4d5e6")


def test_validation_gate_records_the_noise_reasons():
    import inspect
    # Only assert the shared helper is wired in; constructing a full candidate
    # requires the whole schema and is covered by test_structural_extractor.
    src = inspect.getsource(__import__("memory_validation_gate").validate_candidate)
    assert "prompt_noise_reasons" in src, "validation gate no longer applies the noise filter"
