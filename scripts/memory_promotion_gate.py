"""
Quality gate for memory promotion — validates candidates meet minimum
standards before entering durable storage.
"""
from __future__ import annotations

EXPLICIT_AUTHORITY = {"user_explicit", "developer_explicit", "tester_explicit", "system_tool_verified"}


def explicit_promotion_gate(item: dict) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    authority_basis = str(item.get("authority_basis") or "").strip().lower()
    validation_status = str(item.get("validation_status") or "").strip().lower()
    approved_by = str(item.get("approved_by") or "").strip().lower()
    approval_source = str(item.get("approval_source") or "").strip().lower()
    contains_reasoning = bool(item.get("contains_reasoning"))
    contains_speculation = bool(item.get("contains_speculation"))
    requires_approval = bool(item.get("requires_approval", True))

    if contains_reasoning:
        reasons.append("contains_reasoning")
    if contains_speculation:
        reasons.append("contains_speculation")
    if validation_status not in {"valid", ""}:
        reasons.append(f"validation_status:{validation_status}")
    if authority_basis not in EXPLICIT_AUTHORITY:
        reasons.append(f"authority_basis:{authority_basis or 'unknown'}")

    explicit_approval_present = bool(approved_by or approval_source)
    if requires_approval and authority_basis != "system_tool_verified" and not explicit_approval_present:
        reasons.append("missing_explicit_approval")

    return (not reasons), reasons
