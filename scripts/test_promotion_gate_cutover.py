from __future__ import annotations

from memory_promotion_gate import explicit_promotion_gate
from memory_routing_rules import stable_safe_auto



def test_explicit_promotion_gate_blocks_unknown_authority_even_with_high_scores():
    allowed, reasons = explicit_promotion_gate(
        {
            "authority_basis": "unknown",
            "validation_status": "valid",
            "contains_reasoning": False,
            "contains_speculation": False,
            "requires_approval": True,
            "confidence": 0.99,
            "importance": 0.99,
        }
    )

    assert allowed is False
    assert "authority_basis:unknown" in reasons
    assert "missing_explicit_approval" in reasons



def test_stable_safe_auto_requires_explicit_authority_or_approval():
    item = {
        "scope": "stable",
        "confidence": 0.99,
        "importance": 0.99,
        "entity": "browser",
        "property": "capability",
        "value": "available",
        "text": "Browser capability is available.",
        "memory_type": "fact",
        "validation_status": "valid",
        "authority_basis": "unknown",
        "contains_reasoning": False,
        "contains_speculation": False,
        "requires_approval": True,
    }

    assert stable_safe_auto(item) is False

    item["authority_basis"] = "system_tool_verified"
    item["requires_approval"] = False

    assert stable_safe_auto(item) is True
