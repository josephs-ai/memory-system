"""
Batch routing of memory items through the routing rules engine.
Processes multiple candidates in a single pass for efficiency.
"""
import json
import os
import argparse
from functools import lru_cache
from pathlib import Path
from datetime import datetime, timezone, timedelta

# --- Re-entry tuning (env-overridable) ---------------------------------------
# How long a discard suppresses a matching re-extraction. After this window a
# claim is allowed back into routing so it can be re-evaluated. 0 disables the
# rejection-match gate entirely.
REJECT_WINDOW_DAYS = int(os.environ.get("OPENCLAW_REJECT_WINDOW_DAYS", "30"))
# If a re-extracted item's confidence exceeds the confidence it was discarded
# with by at least this margin, allow it back in even inside the window — a
# materially better-supported claim should not stay banned.
REJECT_CONF_OVERRIDE_MARGIN = float(
    os.environ.get("OPENCLAW_REJECT_CONF_OVERRIDE_MARGIN", "0.15")
)
from memory_promotion_gate import explicit_promotion_gate
from memory_routing_rules import stable_safe_auto

from memory_db import (
    upsert_pending_stable,
    upsert_inbox,
    upsert_discarded,
    upsert_memory_item,
    upsert_memory_items,
    fetch_discarded_payloads,
    close_pool,
    POOL,
)

WORKSPACE = Path.home() / ".openclaw" / "workspace"
REVIEW_DIR = WORKSPACE / "memory" / "review"
DECISIONS_LOG = REVIEW_DIR / "decisions.log"

SCRIPTS_DIR = WORKSPACE / ".memory-index" / "scripts"
REVIEW_DIR = WORKSPACE / "memory" / "review"
AGENT_POLICY_FILE = REVIEW_DIR / "agent-memory-policy.json"


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


def append_log(line: str, _buf: list | None = None):
    """Append a decision log line. If _buf is a list, buffer there instead of writing."""
    if _buf is not None:
        _buf.append(line)
    else:
        with open(DECISIONS_LOG, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def flush_log_buffer(buf: list):
    """Write all buffered log lines at once."""
    if not buf:
        return
    with open(DECISIONS_LOG, "a", encoding="utf-8") as f:
        f.write("\n".join(buf) + "\n")


def bulk_fetch_existing_ids(item_ids: list[str]) -> set[str]:
    """Check which IDs already exist in a LIVE queue/table, in one query.

    Intentionally excludes memory_discarded. candidate_id is a deterministic
    hash of (chunk|claim_text), so including the discard table here meant any
    once-discarded claim produced the same id forever and was skipped as a
    SKIP_DUPLICATE before decide_route ever ran — permanently dropping memory.
    Discards are handled separately by the confidence/recency-aware rejection
    gate (fast_candidate_matches_rejected), which can let claims back in.
    """
    if not item_ids:
        return set()
    with POOL.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id FROM (
                    SELECT id FROM memory_pending_stable WHERE id = ANY(%s)
                    UNION ALL
                    SELECT id FROM memory_inbox WHERE id = ANY(%s)
                    UNION ALL
                    SELECT id FROM memory_items WHERE id = ANY(%s)
                ) AS combined
                """,
                (item_ids, item_ids, item_ids),
            )
            return {row[0] for row in cur.fetchall()}


def build_rejected_index(discarded_payloads: list[dict]) -> tuple[dict, dict]:
    """Build fast lookup structures from discarded payloads for O(1) matching.

    Maps each rejection key -> the max confidence it was discarded with, so the
    matcher can let a materially-higher-confidence re-extraction back in. The
    recency window is applied upstream in fetch_discarded_payloads(within_days),
    so anything present here is still inside the active suppression window.
    """
    text_map: dict[str, float] = {}
    slot_map: dict[tuple, float] = {}

    def _record(d: dict, key, conf: float):
        if key in d:
            if conf > d[key]:
                d[key] = conf
        else:
            d[key] = conf

    for old in discarded_payloads:
        conf = float(old.get("confidence") or 0.0)
        old_text = (old.get("text") or "").strip().lower()
        if old_text:
            _record(text_map, old_text, conf)
        e, p, v, s = old.get("entity"), old.get("property"), old.get("value"), old.get("scope")
        if e and p and v:
            _record(slot_map, (e, p, v), conf)
            if s is not None:
                _record(slot_map, (e, p, v, s), conf)
    return text_map, slot_map


def fast_candidate_matches_rejected(item: dict, text_map: dict, slot_map: dict) -> bool:
    """Return True only if this item should still be suppressed by a prior reject.

    A match is overridden (allowed back in) when the new item's confidence
    exceeds the discarded confidence by REJECT_CONF_OVERRIDE_MARGIN — a better
    supported claim is no longer banned. REJECT_WINDOW_DAYS<=0 disables the gate.
    """
    if REJECT_WINDOW_DAYS <= 0:
        return False

    new_conf = float(item.get("confidence") or 0.0)

    def _still_suppressed(prior_conf: float) -> bool:
        # Allow re-entry if the new evidence is materially stronger.
        return new_conf < (prior_conf + REJECT_CONF_OVERRIDE_MARGIN)

    text = (item.get("text") or "").strip().lower()
    if text and text in text_map:
        if _still_suppressed(text_map[text]):
            return True

    e, p, v, s = item.get("entity"), item.get("property"), item.get("value"), item.get("scope")
    if e and p and v:
        if (e, p, v) in slot_map and _still_suppressed(slot_map[(e, p, v)]):
            return True
        if s is not None and (e, p, v, s) in slot_map and _still_suppressed(slot_map[(e, p, v, s)]):
            return True
    return False


def safe_group_write(batch_fn, single_fn, items: list[dict], route_name: str, log_buf: list[str]):
    """Batch write when possible; fall back to per-item writes to isolate bad rows."""
    if not items:
        return
    try:
        batch_fn(items)
    except Exception as e:
        append_log(
            f"{now_iso()} route={route_name} target=db_write mode=batch_failed error={type(e).__name__}:{e}",
            log_buf,
        )
        for item in items:
            try:
                single_fn(item)
            except Exception as item_err:
                append_log(
                    f"{now_iso()} route={route_name} id={item.get('id')} target=db_write mode=item_failed "
                    f"error={type(item_err).__name__}:{item_err}",
                    log_buf,
                )


@lru_cache(maxsize=1)
def load_agent_policy() -> dict:
    return json.loads(AGENT_POLICY_FILE.read_text(encoding="utf-8"))


def apply_agent_policy(item: dict) -> tuple[str, str]:
    policy = load_agent_policy()
    agent = item.get("source_agent", "default")
    agent_policy = policy.get(agent, policy["default"])

    memory_type = item.get("memory_type")
    scope = item.get("scope")
    confidence = float(item.get("confidence", 0.0) or 0.0)

    allowed_types = agent_policy["allowed_memory_types"]
    allowed_scopes = agent_policy["allowed_scopes"]
    min_auto = float(agent_policy["min_confidence_auto"])
    min_inbox = float(agent_policy["min_confidence_inbox"])

    if memory_type not in allowed_types:
        return "DISCARDED", "agent_policy_discard_type_not_allowed"

    if scope not in allowed_scopes:
        return "DISCARDED", "agent_policy_discard_scope_not_allowed"

    authority_basis = str(item.get("authority_basis") or "").strip().lower()
    explicit_authority = authority_basis in {
        "user_explicit",
        "developer_explicit",
        "tester_explicit",
        "system_tool_verified",
    }
    explicit_approval = bool(item.get("approved_by") or item.get("approval_source"))

    if confidence >= min_auto and (
        explicit_authority and (explicit_approval or authority_basis == "system_tool_verified")
    ):
        return "AUTO", "agent_policy_auto"

    if confidence >= min_inbox:
        return "INBOX", "agent_policy_inbox"

    # Below the inbox threshold. Previously this was a hard DISCARD, which wrote
    # the item into memory_discarded and (via the rejection gate) permanently
    # banned it — so a claim that merely lacked context once could never return.
    # Route to INBOX as a soft hold instead so it can be promoted when more
    # evidence arrives. Set OPENCLAW_HARD_DISCARD_BELOW_THRESHOLD=1 to restore
    # the old hard-discard behavior.
    if os.environ.get("OPENCLAW_HARD_DISCARD_BELOW_THRESHOLD") == "1":
        return "DISCARDED", "agent_policy_discard_below_threshold"
    return "INBOX", "agent_policy_below_threshold_soft_hold"


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


def normalize_item(item: dict) -> dict:
    """Normalize schema drift between extraction output and routing expectations."""
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
    return item


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

    # --- Phase 1: Normalize all items ---
    for i, item in enumerate(items):
        items[i] = normalize_item(item)

    # --- Phase 2: Bulk prefetch existing IDs (one query) ---
    all_ids = [item.get("id") for item in items if item.get("id")]
    existing_ids = bulk_fetch_existing_ids(all_ids)

    # --- Phase 3: Prefetch discarded payloads once, build fast lookup ---
    # Windowed by recency: only rejections inside REJECT_WINDOW_DAYS suppress
    # re-entry. Old discards age out so reclaimed/again-relevant memory can flow.
    within = REJECT_WINDOW_DAYS if REJECT_WINDOW_DAYS > 0 else None
    discarded_payloads = fetch_discarded_payloads(within_days=within)
    rejected_text_set, rejected_slot_set = build_rejected_index(discarded_payloads)
    del discarded_payloads  # free memory

    # --- Phase 4: Classify each item (no DB writes yet) ---
    log_buf: list[str] = []
    auto_items: list[dict] = []
    inbox_items: list[dict] = []
    pending_stable_items: list[dict] = []
    discarded_items: list[dict] = []
    any_output = False
    seen_batch_ids: set[str] = set()

    for item in items:
        item_id = item.get("id")

        # Preserve old same-batch duplicate behavior: later duplicates skip once an earlier
        # copy has been classified in this run.
        if item_id and item_id in seen_batch_ids:
            print(f"SKIP_DUPLICATE {item_id} existing_batch_queue")
            append_log(f"{now_iso()} route=SKIP_DUPLICATE id={item_id} target=batch_queue", log_buf)
            any_output = True
            continue

        # Duplicate check against prefetched DB state
        if item_id and item_id in existing_ids:
            print(f"SKIP_DUPLICATE {item_id} existing_db_queue")
            append_log(f"{now_iso()} route=SKIP_DUPLICATE id={item_id} target=db_queue", log_buf)
            any_output = True
            continue

        # Rejected match against prebuilt index
        if fast_candidate_matches_rejected(item, rejected_text_set, rejected_slot_set):
            discarded_items.append(item)
            if item_id:
                seen_batch_ids.add(item_id)
            print("DISCARDED db:memory_discarded")
            append_log(
                f"{now_iso()} route=DISCARDED id={item.get('id')} "
                f"target=memory_discarded reason=previously_rejected_memory",
                log_buf,
            )
            any_output = True
            continue

        route, reason = decide_route(item)

        if route == "AUTO":
            auto_items.append(item)
            if item_id:
                seen_batch_ids.add(item_id)
            print("AUTO db:memory_items")
            append_log(
                f"{now_iso()} route=AUTO id={item.get('id')} "
                f"confidence={item.get('confidence')} importance={item.get('importance')} "
                f"target=memory_items reason={reason}",
                log_buf,
            )
        elif route == "INBOX":
            inbox_items.append(item)
            if item_id:
                seen_batch_ids.add(item_id)
            print("INBOX db:memory_inbox")
            append_log(
                f"{now_iso()} route=INBOX id={item.get('id')} "
                f"confidence={item.get('confidence')} importance={item.get('importance')} "
                f"target=memory_inbox reason={reason}",
                log_buf,
            )
        elif route == "PENDING_STABLE":
            pending_stable_items.append(item)
            if item_id:
                seen_batch_ids.add(item_id)
            print("PENDING_STABLE db:memory_pending_stable")
            append_log(
                f"{now_iso()} route=PENDING_STABLE id={item.get('id')} "
                f"confidence={item.get('confidence')} importance={item.get('importance')} "
                f"target=memory_pending_stable reason={reason}",
                log_buf,
            )
        else:
            discarded_items.append(item)
            if item_id:
                seen_batch_ids.add(item_id)
            print("DISCARDED db:memory_discarded")
            append_log(
                f"{now_iso()} route=DISCARDED id={item.get('id')} "
                f"confidence={item.get('confidence')} importance={item.get('importance')} "
                f"target=memory_discarded reason={reason}",
                log_buf,
            )
        any_output = True

    # --- Phase 5: Batch writes with safe per-item fallback ---
    safe_group_write(upsert_memory_items, upsert_memory_item, auto_items, "AUTO", log_buf)
    safe_group_write(upsert_inbox, lambda item: upsert_inbox([item]), inbox_items, "INBOX", log_buf)
    safe_group_write(upsert_pending_stable, lambda item: upsert_pending_stable([item]), pending_stable_items, "PENDING_STABLE", log_buf)
    safe_group_write(upsert_discarded, lambda item: upsert_discarded([item]), discarded_items, "DISCARDED", log_buf)

    # --- Phase 6: Flush log buffer ---
    flush_log_buffer(log_buf)

    if not any_output:
        print("NONE")

    close_pool()


if __name__ == "__main__":
    main()
