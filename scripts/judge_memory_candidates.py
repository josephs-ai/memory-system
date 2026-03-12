import json
import hashlib
import re
import argparse
from datetime import datetime, timezone
from pathlib import Path
from memory_routing_rules import stable_safe_auto
from extract_memory_fields import extract_fields

WORKSPACE = Path.home() / ".openclaw" / "workspace"
SCRIPTS_DIR = WORKSPACE / ".memory-index" / "scripts"
ENTITY_CONFIG_FILE = SCRIPTS_DIR / "entity_keyword_map.json"


def now_iso():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def load_jsonl(path: Path):
    items = []
    if not path.exists():
        return items
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s or s == "NONE":
                continue
            try:
                items.append(json.loads(s))
            except Exception:
                pass
    return items


def load_entity_keyword_config() -> dict:
    default = {
        "browser": ["browser", "devtools", "cdp", "vnc"],
        "gateway": ["gateway"],
        "worker": ["worker", "upload api", "couchdb"],
        "checkpoint_pipeline": ["checkpoint", "memory pipeline", "canonical memory", "registry"],
        "user_preference": ["prefers", "preference", "likes", "dislikes", "wants"],
    }

    if not ENTITY_CONFIG_FILE.exists():
        return default

    try:
        loaded = json.loads(ENTITY_CONFIG_FILE.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            cleaned = {}
            for k, v in loaded.items():
                if isinstance(k, str) and isinstance(v, list):
                    cleaned[k] = [str(x).lower() for x in v]
            return cleaned or default
    except Exception:
        pass

    return default


ENTITY_KEYWORDS = load_entity_keyword_config()


def make_memory_id(source_chunk: str, text: str) -> str:
    # Use full SHA1, not a short prefix, to reduce collision risk.
    h = hashlib.sha1(f"{source_chunk}|{text}".encode("utf-8")).hexdigest()
    return f"mem-{h}"


def detect_reference_loss(text: str) -> list[str]:
    s = (text or "").strip().lower()
    hits = []

    vague_terms = [
        " he ", " she ", " they ", " it ", " this ", " that ", " these ", " those ",
        " his ", " her ", " their ", " its ",
        " the blue one", " the red one", " the green one",
        " that file", " this file", " that project", " this project",
        " that issue", " this issue", " that bug", " this bug",
        " that fix", " this fix", " that change", " this change",
        " that one", " this one", " it was", " it is", " it uses", " it changed",
    ]

    padded = f" {s} "
    for term in vague_terms:
        if term in padded:
            hits.append(term.strip())

    return sorted(set(hits))


def looks_like_raw_transcript_blob(text: str) -> bool:
    t = (text or "").lower()
    markers = [
        '"type":"message"',
        '"role":"assistant"',
        '"role":"user"',
        '"thinking"',
        '"tool_call"',
        '"provider"',
        '"model"',
        '"usage"',
        '"timestamp"',
        "```tool_call",
        "sender (untrusted metadata)",
    ]
    hit_count = sum(1 for m in markers if m in t)
    return hit_count >= 2 or (len(text) > 500 and "{" in text and "}" in text and '"message"' in t)


def normalize_scalar(value):
    if value is None:
        return None
    s = str(value).strip()
    return s or None


def detect_entity_from_config(text_lower: str) -> str | None:
    best_entity = None
    best_kw_len = -1

    for entity, keywords in ENTITY_KEYWORDS.items():
        for kw in keywords:
            kw = str(kw).strip().lower()
            if not kw:
                continue
            if kw in text_lower and len(kw) > best_kw_len:
                best_entity = entity
                best_kw_len = len(kw)

    return best_entity

def extract_entity_property_value(text: str, memory_type: str) -> tuple[str, str | None, str | None]:
    t = (text or "").lower()

    fields = {}
    try:
        fields = extract_fields(text) or {}
    except Exception:
        fields = {}

    entity = normalize_scalar(fields.get("entity"))
    prop = normalize_scalar(fields.get("property"))
    value = normalize_scalar(fields.get("value"))

    if not entity:
        entity = detect_entity_from_config(t)

    if entity == "browser":
        if not prop and "port" in t:
            prop = "forwarded_devtools_port"
            for token in text.replace(",", " ").split():
                if token.isdigit():
                    value = token
                    break
        elif not prop and ("tool access" in t or "cli" in t):
            prop = "tool_access_mode"
            if "cli" in t and not value:
                value = "cli_fallback"

    elif entity == "checkpoint_pipeline":
        if ("structured registry" in t or "markdown bullets" in t) and not prop:
            prop = "canonical_memory_backend"
            value = "structured_registry_not_markdown_inference"
        elif "approval" in t and "stable" in t and not prop:
            prop = "stable_promotion_policy"
            value = "manual_approval_required"
        elif "support" in t and "cross-agent" in t and not prop:
            prop = "processing_capability"
            value = "cross_agent_supported"

    elif entity == "user_preference":
        if not prop:
            prop = "style_or_behavior"

    if not entity:
        entity = "general"

    return entity, prop, value


def rewrite_atomic_fact(
    text: str,
    entity: str | None,
    prop: str | None,
    value: str | None,
    source_chunk: str | None,
    source_session: str | None,
) -> tuple[str, list[str]]:
    rewritten = (text or "").strip()
    notes = []

    if not rewritten:
        return rewritten, notes

    if "this project" in rewritten.lower() or "that project" in rewritten.lower():
        if source_session:
            rewritten = rewritten.replace("this project", f"session {source_session}")
            rewritten = rewritten.replace("that project", f"session {source_session}")
            rewritten = rewritten.replace("This project", f"Session {source_session}")
            rewritten = rewritten.replace("That project", f"Session {source_session}")
            notes.append("Replaced vague project reference with explicit session identifier.")

    if "this file" in rewritten.lower() or "that file" in rewritten.lower():
        if source_chunk:
            rewritten = rewritten.replace("this file", f"source chunk {source_chunk}")
            rewritten = rewritten.replace("that file", f"source chunk {source_chunk}")
            rewritten = rewritten.replace("This file", f"Source chunk {source_chunk}")
            rewritten = rewritten.replace("That file", f"Source chunk {source_chunk}")
            notes.append("Replaced vague file reference with explicit source chunk.")

    if entity:
        entity_text = entity.replace("_", " ")

        replacement_patterns = [
            (r"^\s*(?:actually,\s*)?it\s+was\s+", f"{entity_text} was "),
            (r"^\s*(?:actually,\s*)?it\s+is\s+", f"{entity_text} is "),
            (r"^\s*(?:actually,\s*)?it\s+uses\s+", f"{entity_text} uses "),
            (r"^\s*(?:actually,\s*)?it\s+changed\s+", f"{entity_text} changed "),
            (r"^\s*(?:actually,\s*)?it\s+has\s+", f"{entity_text} has "),
            (r"^\s*(?:actually,\s*)?it\s+needs\s+", f"{entity_text} needs "),
            (r"^\s*(?:actually,\s*)?it\s+supports\s+", f"{entity_text} supports "),
            (r"^\s*(?:actually,\s*)?it\s+", f"{entity_text} "),
        ]

        for pattern, repl in replacement_patterns:
            new_rewritten = re.sub(pattern, repl, rewritten, flags=re.I)
            if new_rewritten != rewritten:
                rewritten = new_rewritten
                notes.append("Replaced leading 'it' with extracted entity.")
                break

    if entity and prop and value and len(rewritten.split()) <= 8:
        entity_text = entity.replace("_", " ")
        prop_text = prop.replace("_", " ")
        value_text = value.replace("_", " ")

        rewritten = f"{entity_text} {prop_text} is {value_text}."
        notes.append("Expanded short fact into cleaner atomic form.")
    return rewritten, notes

def classify_candidate(cand: dict) -> dict:
    text = cand.get("normalized_text") or cand.get("raw_text") or ""
    t = text.lower()

    source_agent = cand.get("source_agent", "unknown")
    source_session = cand.get("source_session", "unknown")
    source_chunk = cand.get("source_chunk", "unknown")

    tags = []
    confidence = 0.72
    importance = 0.55

    # 1) Memory type
    if any(x in t for x in [
        "prefers", "preference", "likes", "dislikes", "wants", "avoid overly formal"
    ]):
        memory_type = "preference"
        tags.append("preference")
        confidence = 0.88
        importance = 0.78

    elif any(x in t for x in [
        "now uses", "decision", "policy", "switched to", "migrated to", "requires",
        "canonical memory", "architecture", "structured registry"
    ]):
        memory_type = "decision"
        tags.append("decision")
        confidence = 0.92
        importance = 0.90

    elif any(x in t for x in [
        "fix", "fixed", "workaround", "resolved", "support", "supports", "capability"
    ]):
        memory_type = "learned_fix"
        tags.append("learned-fix")
        confidence = 0.88
        importance = 0.82

    else:
        memory_type = "fact"
        confidence = 0.78
        importance = 0.70

    # 2) Entity / property / value
    entity, prop, value = extract_entity_property_value(text, memory_type)

    # 3) Scope
    scope = "stable"

    if any(x in t for x in [
        "current", "currently", "for now", "temporary", "pending", "next step", "todo",
        "active issue", "workaround", "compatibility path", "compatibility mode",
        "interim", "in progress"
    ]):
        scope = "project"
        importance = max(importance, 0.65)

    if any(x in t for x in [
        "test output", "checkpoint run", "one-off observation", "experiment result",
        "observed", "observation", "ran a test", "test run", "log output", "measured"
    ]):
        if memory_type not in {"decision", "preference", "learned_fix"}:
            scope = "daily"

    if (
        scope == "stable"
        and memory_type == "fact"
        and confidence < 0.85
        and entity not in {"user_preference"}
    ):
        scope = "daily"

    # 4) Atomic fact rewrite / explicitness
    rewritten_text, rewrite_notes = rewrite_atomic_fact(
        text=text,
        entity=entity,
        prop=prop,
        value=value,
        source_chunk=source_chunk,
        source_session=source_session,
    )
    text = rewritten_text

    ref_loss_hits = detect_reference_loss(text)
    notes_extra = []

    if rewrite_notes:
        notes_extra.extend(rewrite_notes)

    if looks_like_raw_transcript_blob(text):
        tags.append("raw_transcript_blob")
        notes_extra.append("Looks like raw transcript/tool payload, not a clean atomic fact.")
        confidence = max(0.0, round(confidence - 0.25, 2))
        importance = max(0.0, round(importance - 0.10, 2))
        if memory_type == "fact":
            scope = "daily"

    if ref_loss_hits:
        tags.append("needs_explicit_entity_resolution")
        notes_extra.append("Reference-loss risk: " + ", ".join(ref_loss_hits[:8]))
        confidence = max(0.0, round(confidence - 0.10, 2))
        importance = max(0.0, round(importance - 0.05, 2))

    if prop is None and value is None and entity not in {"user_preference"}:
        tags.append("underspecified_fact")
        confidence = max(0.0, round(confidence - 0.08, 2))
        if memory_type in {"fact", "decision"}:
            importance = max(0.0, round(importance - 0.04, 2))

    # 5) Suggested routing
    # 5) Suggested routing
    stable_safe_auto_flag = stable_safe_auto(
        {
            "scope": scope,
            "confidence": confidence,
            "importance": importance,
            "entity": entity,
            "property": prop,
            "value": value,
            "text": text,
            "memory_type": memory_type,
        }
    )

    if "raw_transcript_blob" in tags and prop is None and value is None:
        suggested_route = "discard"
    elif confidence < 0.65:
        suggested_route = "discard"
    elif scope == "project":
        suggested_route = "project"
    elif scope == "daily":
        suggested_route = "daily"
    elif "needs_explicit_entity_resolution" in tags and scope == "stable":
        suggested_route = "inbox"
    elif "underspecified_fact" in tags and scope == "stable":
        suggested_route = "inbox"
    elif stable_safe_auto_flag:
        suggested_route = "auto"
    elif scope == "stable" and confidence >= 0.90 and importance >= 0.90:
        suggested_route = "pending_stable"
    elif confidence >= 0.90 and importance >= 0.80:
        suggested_route = "auto"
    elif confidence >= 0.65:
        suggested_route = "inbox"
    else:
        suggested_route = "discard"

    # 6) Freshness class
    if memory_type == "preference":
        freshness_class = "long_lived"
    elif memory_type == "decision" and entity == "checkpoint_pipeline":
        freshness_class = "permanent"
    elif memory_type == "learned_fix":
        freshness_class = "permanent"
    elif scope == "project":
        freshness_class = "volatile"
    elif entity in {"browser", "gateway", "worker"} and prop in {
        "forwarded_devtools_port",
        "local_devtools_port",
        "tool_access_mode",
    }:
        freshness_class = "volatile"
    else:
        freshness_class = "long_lived"

    base_notes = "Judged from generated memory candidate."
    if notes_extra:
        base_notes += " " + " ".join(notes_extra)

    return {
        "id": make_memory_id(cand.get("source_chunk", "unknown"), text),
        "text": text,
        "memory_type": memory_type,
        "scope": scope,
        "entity": entity,
        "property": prop,
        "value": value,
        "status": "active",
        "confidence": round(confidence, 2),
        "importance": round(importance, 2),
        "freshness_class": freshness_class,
        "source_agent": source_agent,
        "source_session": source_session,
        "source_chunk": source_chunk,
        "first_seen": now_iso(),
        "last_confirmed": now_iso(),
        "supersedes": None,
        "tags": sorted(set(tags)),
        "notes": base_notes,
        "candidate_id": cand.get("candidate_id"),
        "candidate_score": cand.get("candidate_score"),
        "candidate_reasons": cand.get("candidate_reasons", []),
        "suggested_route": suggested_route,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file")
    args = parser.parse_args()

    path = Path(args.input_file)
    if not path.exists():
        print("NONE")
        return

    text = path.read_text(encoding="utf-8", errors="ignore").strip()
    if not text or text == "NONE":
        print("NONE")
        return

    lines = []
    for line in text.splitlines():
        s = line.strip()
        if not s or s == "NONE":
            continue
        try:
            cand = json.loads(s)
        except Exception:
            continue
        lines.append(classify_candidate(cand))

    if not lines:
        print("NONE")
        return

    for item in lines:
        print(json.dumps(item, ensure_ascii=False))

if __name__ == "__main__":
    main()
