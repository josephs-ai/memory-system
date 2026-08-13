"""
Auto Promote Safe Items — utility for the OpenClaw memory system.

Key functions: now_iso, append_log, identity_key, target_file_for_item
"""
import sys
from pathlib import Path
from datetime import datetime, timezone

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from memory_validation_gate import prompt_noise_reasons
from memory_db import (
    fetch_memory_items,
    upsert_memory_item,
    close_pool,
    get_conn,
)

WORKSPACE = Path.home() / ".openclaw" / "workspace"
REVIEW_DIR = WORKSPACE / "memory" / "review"
DECISIONS_LOG = REVIEW_DIR / "decisions.log"


def now_iso():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def append_log(line: str):
    with open(DECISIONS_LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


# Claim types eligible for promotion out of the inbox.
#
# Derived from the types stable_safe_auto() already accepts on the AUTO route,
# intersected with the ClaimType values the extractor actually emits. The two
# gates disagreeing was a bug, not a policy: "rule" is the second most common
# type the extractor produces (2,048 queued) and the AUTO route treats it as
# promotable, but the inbox gate silently rejected every one of them.
#
# Deliberately excluded: bug_history and observation (episodic, not durable
# truth), open_question and summary_only (not claims at all). "learned_fix" is
# kept because other routing code still emits that name, though no ClaimType
# uses it.
PROMOTABLE_TYPES = {
    "fact",
    "decision",
    "preference",
    "rule",
    "implementation_pattern",
    "learned_fix",
}

# Promotion thresholds. These are policy, not correctness -- they trade recall
# against how much unreviewed prose reaches permanent memory, so they are named
# here rather than buried as literals.
#
# Confidence was 0.90. The scorer can reach 1.0, so that gate was never
# unreachable the way the importance one was; it simply admitted only short,
# explicit, user-stated claims. Agent-authored work tops out near 0.88 because
# of the verbosity and reasoning penalties, so in practice nothing the agents
# produced could ever be promoted. Lowered to 0.85 deliberately, to let that
# work through, accepting more agent prose in exchange.
PROMOTION_MIN_CONFIDENCE = 0.85
PROMOTION_MIN_IMPORTANCE = 0.85


def has_identity(item: dict) -> bool:
    """True when an item carries a real (entity, property, value) identity."""
    return bool(item.get("entity")) and bool(item.get("property")) and item.get("value") is not None


def identity_key(item: dict):
    """A key that identifies an item, or None when nothing does.

    The structured (entity, property, value) triple is preferred, but the
    extractor does not populate it -- every queued item and 10,161 stored ones
    have all three NULL. Keying on the triple alone therefore collapsed every
    such item onto ('fact', None, None, None), so unrelated rows compared equal.
    That is not identity, it is a collision, and it made the dedup below reject
    everything as a duplicate.

    Falling back to normalized text gives those items a real identity. The
    variants are tagged so a text key can never accidentally equal a triple key,
    and None is returned when neither exists -- callers must treat None as
    "matches nothing" rather than as a value.
    """
    if has_identity(item):
        return (
            "epv",
            item.get("memory_type"),
            item.get("entity"),
            item.get("property"),
            item.get("value"),
        )

    # Mirrors the router's dedup, which hashes md5(lower(trim(text))).
    text = (item.get("text") or item.get("normalized_text") or item.get("claim_text") or "")
    text = " ".join(str(text).lower().split())
    if text:
        return ("text", item.get("memory_type"), text)

    return None


def target_file_for_item(item: dict) -> str:
    entity = item.get("entity")
    memory_type = item.get("memory_type")

    if entity == "browser":
        return "browser.md"
    if entity == "user_preference":
        return "preferences.md"
    if entity == "checkpoint_pipeline" and memory_type == "decision":
        return "memory-system.md"
    if memory_type == "learned_fix":
        return "learned-fixes.md"
    return "memory-system.md"


def safe_to_promote(item: dict, canonical_items: list[dict]) -> bool:
    if item.get("scope") != "stable":
        return False
    if float(item.get("confidence", 0.0) or 0.0) < PROMOTION_MIN_CONFIDENCE:
        return False
    if float(item.get("importance", 0.0) or 0.0) < PROMOTION_MIN_IMPORTANCE:
        return False
    if item.get("memory_type") not in PROMOTABLE_TYPES:
        return False

    # This path never consulted validation at all, which is how reviewer prompt
    # scaffolding ("Return JSON only: {...}") reached permanent memory: it is
    # imperative and declarative, so it scores like a durable rule. Recomputed
    # from the text rather than read from the payload, so items queued before
    # these checks existed are filtered too.
    text = item.get("text") or item.get("normalized_text") or item.get("claim_text") or ""
    if prompt_noise_reasons(text):
        return False

    # A question is an open item, not a durable truth -- storing "Should Phase 16
    # stay bounded?" as a rule asserts nothing. The validation gate has always
    # flagged these (question_not_durable); this path simply never read it.
    if "?" in str(text):
        return False

    # Nothing identifies this item -- not even its text. It cannot be deduped,
    # so it cannot be promoted safely. Hold it for review.
    ikey = identity_key(item)
    if ikey is None:
        return False

    for existing in canonical_items:
        if existing.get("status") != "active":
            continue

        if identity_key(existing) == ikey:
            return False

        # Contradiction check: only meaningful between two structured items.
        # Without a real triple, "same slot, different value" cannot be
        # established, and comparing NULLs would reject unrelated rows.
        if not (has_identity(existing) and has_identity(item)):
            continue

        if (
            existing.get("memory_type") == item.get("memory_type")
            and existing.get("entity") == item.get("entity")
            and existing.get("property") == item.get("property")
            and existing.get("scope") == item.get("scope")
            and existing.get("value") != item.get("value")
        ):
            return False

    return True


def fetch_queue_rows(table_name: str):
    allowed = {"memory_inbox"}
    if table_name not in allowed:
        raise ValueError(f"unsupported table: {table_name}")

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(f"SELECT id, payload FROM {table_name} ORDER BY queued_at, id")
            return [{"id": row[0], "payload": row[1]} for row in cur.fetchall()]


def delete_queue_item(table_name: str, item_id: str):
    allowed = {"memory_inbox"}
    if table_name not in allowed:
        raise ValueError(f"unsupported table: {table_name}")

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(f"DELETE FROM {table_name} WHERE id = %s", (item_id,))
        conn.commit()


def reconfirm_memory_items(ids: list[str]) -> int:
    """Bump last_confirmed on items we just saw asserted again.

    A re-observed fact is not noise: it is evidence the fact is still true
    *now*. Dropping the duplicate silently -- the old behaviour -- left
    last_confirmed frozen at whenever the fact was first written, so recency
    ranking had nothing fresh to rank and "what did we work on yesterday"
    could only ever surface months-old rows.
    """
    if not ids:
        return 0
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE memory_items
                   SET last_confirmed = now(),
                       reinforcement_count = COALESCE(reinforcement_count, 0) + 1,
                       last_reinforced_at = now()
                 WHERE id = ANY(%s) AND status = 'active'
                """,
                (ids,),
            )
            n = cur.rowcount
        conn.commit()
    return n


def promote_inbox_queue(canonical_items: list[dict]):
    queue_rows = fetch_queue_rows("memory_inbox")
    promoted = 0

    # Identity -> id of the active item asserting it, for re-confirmation.
    #
    # Only well-formed identities take part. identity_key is
    # (memory_type, entity, property, value), so an item missing those collapses
    # to a degenerate key like ('fact', None, None, None) that thousands of
    # unrelated rows share. Matching on that is not identity, it is a collision,
    # and treating it as a duplicate would discard unrelated queue entries.
    active_by_key = {
        identity_key(e): e.get("id")
        for e in canonical_items
        if e.get("status") == "active" and identity_key(e) is not None
    }
    reconfirm_ids: set[str] = set()

    for row in queue_rows:
        item = dict(row["payload"])
        item_id = item.get("id")

        if not safe_to_promote(item, canonical_items):
            ikey = identity_key(item)
            existing_id = active_by_key.get(ikey) if ikey is not None else None
            if existing_id:
                # Same fact, already stored: re-confirm it and clear the queue row
                # so the queue drains instead of replaying forever.
                reconfirm_ids.add(existing_id)
                if item_id:
                    delete_queue_item("memory_inbox", item_id)
            continue

        item["status"] = "active"
        item["target_file"] = target_file_for_item(item)
        item["target_section"] = "Active"
        item["last_confirmed"] = now_iso()
        item["suggested_route"] = None

        upsert_memory_item(item)
        canonical_items.append(item)
        delete_queue_item("memory_inbox", item_id)

        promoted += 1
        append_log(
            f"{now_iso()} auto_promote id={item_id} from=memory_inbox "
            f"target_file={item.get('target_file')}"
        )

    reconfirmed = reconfirm_memory_items(sorted(reconfirm_ids))
    if reconfirmed:
        append_log(f"{now_iso()} reconfirmed count={reconfirmed} from=memory_inbox")
    print(f"RECONFIRMED={reconfirmed}")

    return promoted


def main():
    canonical_items = fetch_memory_items(["active", "superseded"])

    promoted_inbox = promote_inbox_queue(canonical_items)

    total = promoted_inbox
    print(f"AUTO_PROMOTED={total}")
    print("FROM_AUTO=0")
    print(f"FROM_INBOX={promoted_inbox}")
    close_pool()


if __name__ == "__main__":
    main()
