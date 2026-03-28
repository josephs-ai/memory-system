def stable_safe_auto(item: dict) -> bool:
    confidence = float(item.get("confidence", 0.0) or 0.0)
    importance = float(item.get("importance", item.get("retention_priority", 0.0)) or 0.0)
    scope = item.get("scope", "stable")

    entity = (item.get("entity") or "").strip()
    prop = (item.get("property") or "").strip()
    value = (item.get("value") or "").strip()
    text = (item.get("text") or "").strip()
    memory_type = (item.get("memory_type") or "").strip().lower()

    has_full_structure = bool(entity and prop and value)
    looks_atomic = bool(text) and len(text.split()) <= 30

    return (
        str(scope).strip().lower() not in {"session", "transient", "ephemeral"}
        and confidence >= 0.93
        and importance >= 0.90
        and has_full_structure
        and looks_atomic
        and memory_type in {
            "fact",
            "decision",
            "preference",
            "architecture_rule",
            "implementation_pattern",
            "rule",
            "lesson_learned",
            "observation",
            "guardrail",
            "workflow_rule",
        }
    )


def classify_memory(item: dict) -> str:
    memory_type = (item.get("memory_type") or "").strip().lower()
    if memory_type in {"fact", "preference", "decision", "architecture_rule", "rule", "feature_summary", "decision_record", "guardrail", "workflow_rule"}:
        return "semantic"
    if memory_type in {"lesson_learned", "learned_fix", "pattern", "implementation_pattern", "workaround", "risk_pattern", "validation_pattern", "failure_mode"}:
        return "procedural"
    if memory_type in {"observation", "review_finding", "bug_history", "test_outcome"}:
        return "episodic"
    return "episodic"


def apply_lifecycle_gates(item: dict) -> tuple[str, str]:
    confidence = float(item.get("confidence", 0.0) or 0.0)
    retention_priority = float(item.get("retention_priority", item.get("importance", 0.0)) or 0.0)

    item["memory_class"] = classify_memory(item)

    source = str(item.get("source_agent") or "")
    approval = str(item.get("approved_by") or "")
    if approval == "human" or source in ("system_compaction", "human"):
        item["source_quality"] = "high"
        item["is_inferred"] = False
    elif confidence >= 0.90:
        item["source_quality"] = "high"
        item["is_inferred"] = True
    else:
        item["source_quality"] = "unverified"
        item["is_inferred"] = True

    memory_type = (item.get("memory_type") or "").strip().lower()
    if memory_type == "guardrail":
        item["retrieval_eligibility"] = "always"
    elif memory_type in {"architecture_rule", "decision", "preference", "rule", "workflow_rule"}:
        item["retrieval_eligibility"] = "normal"
    elif memory_type == "observation":
        age_days = float(item.get("age_days", 0) or 0)
        item["retrieval_eligibility"] = "rare" if age_days < 7 else "suppressed"
    else:
        item["retrieval_eligibility"] = "normal"

    if confidence < 0.60:
        return "discarded", "failed_trust_gate_low_confidence"

    scope = str(item.get("scope") or "").strip().lower()
    if retention_priority < 0.50 or scope in {"session", "transient", "ephemeral"}:
        return "session-only", "failed_retention_gate_transient"

    if confidence >= 0.85 and retention_priority >= 0.80:
        return "durable", "passed_lifecycle_gate_high_value"

    reinforcement_count = int(item.get("reinforcement_count", 0) or 0)
    if reinforcement_count > 3:
        return "reinforced", "passed_lifecycle_gate_reinforced"

    if memory_type in {"decision", "architecture_rule", "lesson_learned", "rule", "guardrail", "workflow_rule"} and confidence >= 0.80:
        return "durable", "passed_lifecycle_gate_high_value_type"

    if memory_type == "observation" and reinforcement_count == 0:
        if float(item.get("age_days", 0) or 0) > 7:
            return "archived", "decayed_unreinforced_observation"

    if float(item.get("age_days", 0) or 0) > 30 and reinforcement_count == 0:
        return "discarded", "decayed_unreinforced_candidate"

    return "candidate", "passed_lifecycle_gate_needs_reinforcement"
