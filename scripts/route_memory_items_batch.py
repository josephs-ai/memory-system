"""
Batch routing of memory items through the routing rules engine.
Processes multiple candidates in a single pass for efficiency.
"""
import json
import argparse
import subprocess
import tempfile
from pathlib import Path
from datetime import datetime, timezone
from memory_promotion_gate import explicit_promotion_gate
from memory_routing_rules import stable_safe_auto

from memory_db import (
    upsert_pending_stable,
    upsert_inbox,
    upsert_discarded,
    upsert_memory_item,
    queue_item_exists_anywhere,
    candidate_matches_rejected,
    close_pool,
)

WORKSPACE = Path.home() / ".openclaw" / "workspace"
REVIEW_DIR = WORKSPACE / "memory" / "review"
DECISIONS_LOG = REVIEW_DIR / "decisions.log"

SCRIPTS_DIR = WORKSPACE / ".memory-index" / "scripts"
AGENT_POLICY_SCRIPT = SCRIPTS_DIR / "check_agent_memory_policy.py"


def now_iso():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def load_jsonl_text(text: str):
    items = []
    if not text or text.strip() == "NONE":
        return items

    for line in text.splitlines():
        s = line.strip()
        if not s or s == "NONE":
            continue
        try:
            items.append(json.loads(s))
        except Exception:
            pass
    return items


def append_log(line: str):
    with open(DECISIONS_LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def json_safe(value):
    if hasattr(value, "isoformat"):
        return value.isoformat()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def write_temp_json(item: dict) -> Path:
    tmp = Path(tempfile.mkstemp(suffix=".json")[1])
    tmp.write_text(
        json.dumps(item, ensure_ascii=False, indent=2, default=json_safe),
        encoding="utf-8",
    )
    return tmp


def apply_agent_policy(item: dict) -> tuple[str, str]:
    agent = item.get("source_agent", "default")
    tmp = write_temp_json(item)

    try:
        result = subprocess.run(
            [
                "python3",
                str(AGENT_POLICY_SCRIPT),
                "--agent",
                agent,
                "--item",
                str(tmp),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        text = result.stdout.strip()

        if text == "AUTO":
            return "AUTO", "agent_policy_auto"
        if text == "INBOX":
            return "INBOX", "agent_policy_inbox"
        if text.startswith("DISCARD"):
            return "DISCARDED", f"agent_policy_{text.replace(' ', '_').lower()}"

        return "INBOX", "agent_policy_unknown_fallback"

    finally:
        try:
            tmp.unlink()
        except Exception:
            pass


def decide_route(item: dict) -> tuple[str, str]:

    suggested = item.get("suggested_route")
    confidence = float(item.get("confidence", 0.0) or 0.0)
    importance = float(item.get("importance", 0.0) or 0.0)
    scope = item.get("scope", "stable")
    explicit_gate_ok, explicit_gate_reasons = explicit_promotion_gate(item)

    if stable_safe_auto(item):
        return "AUTO", "explicit_structural_auto"

    if scope == "stable" and confidence >= 0.90 and importance >= 0.90:
        if explicit_gate_ok:
            return "PENDING_STABLE", "stable_high_importance_explicit_gate"
        return "INBOX", "stable_high_importance_missing_explicit_gate:" + ",".join(explicit_gate_reasons)

    policy_route, policy_reason = apply_agent_policy(item)

    if suggested == "discard":
        return "DISCARDED", "judge_discard"

    if suggested == "pending_stable":
        return "PENDING_STABLE", "judge_pending_stable"

    if suggested == "project":
        judge_route, judge_reason = "INBOX", "judge_project_to_inbox"
    elif suggested == "daily":
        judge_route, judge_reason = "INBOX", "judge_daily_to_inbox"
    elif suggested == "auto":
        judge_route, judge_reason = "AUTO", "judge_auto"
    elif suggested == "inbox":
        judge_route, judge_reason = "INBOX", "judge_inbox"
    else:
        if scope == "project":
            judge_route, judge_reason = "INBOX", "fallback_project"
        elif confidence >= 0.90 and importance >= 0.80:
            judge_route, judge_reason = "AUTO", "fallback_high_conf_high_importance"
        elif confidence >= 0.65:
            judge_route, judge_reason = "INBOX", "fallback_review"
        else:
            judge_route, judge_reason = "DISCARDED", "fallback_low_confidence"

    if policy_route == "DISCARDED":
        return "DISCARDED", policy_reason

    if policy_route == "INBOX" and judge_route == "AUTO":
        return "INBOX", f"{policy_reason}_caps_{judge_reason}"

    return judge_route, judge_reason


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    args = parser.parse_args()

    input_path = Path(args.input)
    text = input_path.read_text(encoding="utf-8", errors="ignore")
    items = load_jsonl_text(text)

    if not items:
        print("NONE")
        return

    any_output = False

    for item in items:
        # Normalize schema drift between extraction output and routing expectations:
        # 1. id: extraction produces candidate_id, routing expects id
        if not item.get("id") and item.get("candidate_id"):
            item["id"] = item["candidate_id"]
        # 2. confidence: extraction produces candidate_score (0-10), routing expects confidence (0-1)
        if item.get("confidence") is None and item.get("candidate_score") is not None:
            item["confidence"] = min(float(item["candidate_score"]) / 10.0, 1.0)
        # 3. importance: derive from durability_class + impact_level if missing
        if item.get("importance") is None:
            durability = (item.get("durability_class") or "").lower()
            impact = (item.get("impact_level") or "").lower()
            dur_map = {"durable": 0.9, "stable": 0.8, "candidate": 0.6, "ephemeral": 0.2}
            imp_map = {"critical": 0.95, "high": 0.85, "medium": 0.65, "low": 0.4}
            dur_score = dur_map.get(durability, 0.5)
            imp_score = imp_map.get(impact, 0.5)
            item["importance"] = round((dur_score + imp_score) / 2.0, 3)
        # 4. scope: extraction uses scope_envelope.scope_type, routing expects flat scope
        if item.get("scope") is None:
            scope_env = item.get("scope_envelope") or {}
            raw = scope_env.get("raw") or {}
            item["scope"] = raw.get("scope") or scope_env.get("scope_type") or "stable"
        # 5. text: routing rules check text/normalized_text/claim_text — ensure text is set
        if not item.get("text") and item.get("normalized_text"):
            item["text"] = item["normalized_text"]
        elif not item.get("text") and item.get("raw_text"):
            item["text"] = item["raw_text"]
        # 6. memory_type: routing checks memory_type, extraction uses claim_type
        if not item.get("memory_type") and item.get("claim_type"):
            item["memory_type"] = item["claim_type"]
        item_id = item.get("id")

        if item_id and queue_item_exists_anywhere(item_id):
            print(f"SKIP_DUPLICATE {item_id} existing_db_queue")
            append_log(f"{now_iso()} route=SKIP_DUPLICATE id={item_id} target=db_queue")
            any_output = True
            continue

        matched_rejected, rejected_item = candidate_matches_rejected(item)
        if matched_rejected:
            upsert_discarded([item])
            print("DISCARDED db:memory_discarded")
            append_log(
                f"{now_iso()} route=DISCARDED id={item.get('id')} "
                f"target=memory_discarded reason=previously_rejected_memory "
                f"matched_rejected_id={rejected_item.get('id') if rejected_item else 'unknown'}"
            )
            any_output = True
            continue

        route, reason = decide_route(item)

        if route == "AUTO":
            upsert_memory_item(item)
            print("AUTO db:memory_items")
            append_log(
                f"{now_iso()} route=AUTO id={item.get('id')} "
                f"confidence={item.get('confidence')} importance={item.get('importance')} "
                f"target=memory_items reason={reason}"
            )
            any_output = True

        elif route == "INBOX":
            upsert_inbox([item])
            print("INBOX db:memory_inbox")
            append_log(
                f"{now_iso()} route=INBOX id={item.get('id')} "
                f"confidence={item.get('confidence')} importance={item.get('importance')} "
                f"target=memory_inbox reason={reason}"
            )
            any_output = True

        elif route == "PENDING_STABLE":
            upsert_pending_stable([item])
            print("PENDING_STABLE db:memory_pending_stable")
            append_log(
                f"{now_iso()} route=PENDING_STABLE id={item.get('id')} "
                f"confidence={item.get('confidence')} importance={item.get('importance')} "
                f"target=memory_pending_stable reason={reason}"
            )
            any_output = True

        else:
            upsert_discarded([item])
            print("DISCARDED db:memory_discarded")
            append_log(
                f"{now_iso()} route=DISCARDED id={item.get('id')} "
                f"confidence={item.get('confidence')} importance={item.get('importance')} "
                f"target=memory_discarded reason={reason}"
            )
            any_output = True

    if not any_output:
        print("NONE")

    close_pool()


if __name__ == "__main__":
    main()
