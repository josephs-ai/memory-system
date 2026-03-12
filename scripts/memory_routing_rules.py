def stable_safe_auto(item: dict) -> bool:
    confidence = float(item.get("confidence", 0.0) or 0.0)
    importance = float(item.get("importance", 0.0) or 0.0)
    scope = item.get("scope", "stable")

    entity = (item.get("entity") or "").strip()
    prop = (item.get("property") or "").strip()
    value = (item.get("value") or "").strip()
    text = (item.get("text") or "").strip()
    memory_type = (item.get("memory_type") or "").strip().lower()

    has_full_structure = bool(entity and prop and value)
    looks_atomic = bool(text) and len(text.split()) <= 30

    return (
        scope == "stable"
        and confidence >= 0.93
        and importance >= 0.90
        and has_full_structure
        and looks_atomic
        and memory_type in {"fact", "decision", "preference"}
    )
