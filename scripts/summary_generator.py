"""
summary_generator.py — Generates summaries from grouped memory items and archives constituents.

Part of P2-S2: Progressive Summarization.

Strategy:
1. Extractive summarization: select most representative items by confidence/importance
2. Concatenate into a structured summary text
3. Store summary as a new memory_item with memory_type='summary'
4. Embed summary in Qdrant
5. Mark constituent items as status='consolidated'

No LLM calls — purely extractive. Keeps costs zero and behavior deterministic.

Design decisions (per tester review):
- Summaries stored in memory_items table (not a separate table)
- Constituents marked 'consolidated' (removed from active retrieval)
- Summary text includes constituent count and time range as context
"""

from __future__ import annotations

import hashlib
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import psycopg

LOGGER = logging.getLogger("openclaw.summary_generator")

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

DB_DSN = (os.environ.get("OPENCLAW_MEMORY_DSN") or os.environ.get("OPENCLAW_MEMORY_DB_DSN") or "dbname=openclaw_memory")  # honor both canonical DSN env names

# How many representative items to include in summary text
MAX_REPRESENTATIVE_ITEMS = 8
# Max chars per representative item in summary
MAX_ITEM_CHARS = 200

RESOLUTION_LABELS = {
    "daily": "Daily",
    "weekly": "Weekly",
    "monthly": "Monthly",
}


def _summary_id(group_key: str) -> str:
    """Generate deterministic summary ID from group key."""
    h = hashlib.sha256(group_key.encode("utf-8")).hexdigest()[:12]
    return f"summary-{h}"


def _select_representative_items(items: list[dict], max_items: int = MAX_REPRESENTATIVE_ITEMS) -> list[dict]:
    """
    Select the most representative items from a group.

    Strategy: sort by confidence (desc), then by text length (prefer more detailed).
    Take top N.
    """
    scored = []
    for item in items:
        conf = float(item.get("confidence") or 0.5)
        text_len = len(item.get("text") or "")
        # Prefer higher confidence and longer (more detailed) text
        score = conf * 0.7 + min(text_len / 500.0, 1.0) * 0.3
        scored.append((score, item))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [item for _, item in scored[:max_items]]


def generate_summary_text(
    group: dict,
    representative_items: list[dict],
) -> str:
    """
    Generate structured summary text from a consolidation group.

    Format:
        [Weekly Summary | scope: project-x | 2026-W15 | 12 items consolidated]
        - Most important point from items
        - Second point
        ...
    """
    resolution = group.get("resolution", "unknown")
    scope = group.get("scope", "global")
    time_label = group.get("time_label", "unknown")
    item_count = len(group.get("items", []))
    res_label = RESOLUTION_LABELS.get(resolution, resolution.title())

    header = f"[{res_label} Summary | scope: {scope} | {time_label} | {item_count} items consolidated]"

    lines = [header]
    for item in representative_items:
        text = (item.get("text") or "").strip()
        # Skip raw JSON / transcript fragments
        if text.startswith("{") or text.startswith("["):
            # Try to extract meaningful content
            continue
        if len(text) > MAX_ITEM_CHARS:
            text = text[:MAX_ITEM_CHARS].rsplit(" ", 1)[0] + "..."
        if text:
            lines.append(f"- {text}")

    if len(lines) == 1:
        # All items were JSON/noise — fall back to count-only summary
        lines.append(f"- [{item_count} memory items from {time_label} (scope: {scope})]")

    return "\n".join(lines)


def create_summary_item(
    group: dict,
    summary_text: str,
) -> dict:
    """Create a memory_items-compatible dict for the summary."""
    now = datetime.now(timezone.utc)
    constituent_ids = [i.get("id") for i in group.get("items", []) if i.get("id")]

    return {
        "id": _summary_id(group["group_key"]),
        "text": summary_text,
        "memory_type": "summary",
        "scope": group.get("scope", "global"),
        "entity": "",
        "property": "",
        "value": "",
        "status": "active",
        "confidence": 0.7,  # Summaries get lower confidence than originals
        "freshness_class": "archival",
        "sensitivity": "public",
        "resolution_level": group["resolution"],
        "constituent_item_ids": constituent_ids,
        "time_range_start": group.get("time_range_start"),
        "time_range_end": group.get("time_range_end"),
        "first_seen": now,
        "last_confirmed": now,
    }


def store_summary(
    summary_item: dict,
    *,
    conn: psycopg.Connection | None = None,
) -> str:
    """
    Store summary in memory_items table and return its ID.

    Uses INSERT ... ON CONFLICT to handle re-runs safely.
    """
    close_conn = False
    if conn is None:
        conn = psycopg.connect(DB_DSN)
        close_conn = True

    try:
        import json
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO memory_items (
                    id, text, memory_type, scope, entity, property, value,
                    status, confidence, freshness_class, sensitivity,
                    resolution_level, constituent_item_ids,
                    time_range_start, time_range_end,
                    first_seen, last_confirmed
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s,
                    %s, %s, %s, %s,
                    %s, %s::jsonb,
                    %s, %s,
                    %s, %s
                )
                ON CONFLICT (id) DO UPDATE SET
                    text = EXCLUDED.text,
                    constituent_item_ids = EXCLUDED.constituent_item_ids,
                    last_confirmed = EXCLUDED.last_confirmed
                """,
                (
                    summary_item["id"],
                    summary_item["text"],
                    summary_item["memory_type"],
                    summary_item["scope"],
                    summary_item.get("entity", ""),
                    summary_item.get("property", ""),
                    summary_item.get("value", ""),
                    summary_item["status"],
                    summary_item["confidence"],
                    summary_item.get("freshness_class", "archival"),
                    summary_item.get("sensitivity", "public"),
                    summary_item["resolution_level"],
                    json.dumps(summary_item["constituent_item_ids"]),
                    summary_item.get("time_range_start"),
                    summary_item.get("time_range_end"),
                    summary_item["first_seen"],
                    summary_item["last_confirmed"],
                ),
            )
        conn.commit()
        return summary_item["id"]

    finally:
        if close_conn:
            conn.close()


def archive_constituents(
    constituent_ids: list[str],
    *,
    conn: psycopg.Connection | None = None,
) -> int:
    """
    Mark constituent items as 'consolidated' status.

    Returns count of items updated.
    """
    if not constituent_ids:
        return 0

    close_conn = False
    if conn is None:
        conn = psycopg.connect(DB_DSN)
        close_conn = True

    try:
        placeholders = ",".join(["%s"] * len(constituent_ids))
        with conn.cursor() as cur:
            cur.execute(
                f"""
                UPDATE memory_items
                SET status = 'consolidated'
                WHERE id IN ({placeholders})
                  AND status = 'active'
                """,
                tuple(constituent_ids),
            )
            count = cur.rowcount
        conn.commit()
        return count

    finally:
        if close_conn:
            conn.close()


def embed_summary(
    summary_item: dict,
) -> bool:
    """
    Embed summary text into Qdrant memory_items collection.

    Returns True on success, False on failure.
    """
    try:
        from sentence_transformers import SentenceTransformer
        from qdrant_client import QdrantClient
        from qdrant_client.http.models import PointStruct

        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        embedding = model.encode(summary_item["text"], normalize_embeddings=True).tolist()

        client = QdrantClient(
            host=os.environ.get("OPENCLAW_QDRANT_HOST", "localhost"),
            port=int(os.environ.get("OPENCLAW_QDRANT_PORT", "6333")),
        )

        point = PointStruct(
            id=abs(hash(summary_item["id"])) % (2**63),
            vector=embedding,
            payload={
                "id": summary_item["id"],
                "text": summary_item["text"],
                "memory_type": "summary",
                "scope": summary_item.get("scope", "global"),
                "resolution_level": summary_item.get("resolution_level"),
                "confidence": summary_item.get("confidence", 0.7),
            },
        )

        client.upsert(collection_name="memory_items", points=[point])
        return True

    except Exception:
        LOGGER.exception("Failed to embed summary %s", summary_item.get("id"))
        return False


def consolidate_group(
    group: dict,
    *,
    dry_run: bool = False,
    conn: psycopg.Connection | None = None,
) -> dict:
    """
    Full consolidation pipeline for a single group:
    1. Select representative items
    2. Generate summary text
    3. Store summary in DB
    4. Embed in Qdrant
    5. Archive constituents

    Returns a result dict with summary_id, items_archived, etc.
    """
    reps = _select_representative_items(group["items"])
    summary_text = generate_summary_text(group, reps)
    summary_item = create_summary_item(group, summary_text)

    if dry_run:
        return {
            "summary_id": summary_item["id"],
            "summary_text": summary_text,
            "constituent_count": len(summary_item["constituent_item_ids"]),
            "dry_run": True,
        }

    summary_id = store_summary(summary_item, conn=conn)
    embedded = embed_summary(summary_item)
    archived = archive_constituents(summary_item["constituent_item_ids"], conn=conn)

    return {
        "summary_id": summary_id,
        "resolution": group["resolution"],
        "scope": group.get("scope", "global"),
        "constituent_count": len(summary_item["constituent_item_ids"]),
        "items_archived": archived,
        "embedded": embedded,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    import argparse
    import json
    from consolidation_grouper import fetch_consolidation_candidates, group_for_consolidation

    parser = argparse.ArgumentParser(description="Summary generator")
    parser.add_argument("--dry-run", action="store_true", help="Preview without writing")
    parser.add_argument("--limit", type=int, default=None, help="Max groups to process")
    parser.add_argument("--resolution", choices=["daily", "weekly", "monthly"], help="Filter by resolution")
    args = parser.parse_args()

    items = fetch_consolidation_candidates()
    groups = group_for_consolidation(items)

    if args.resolution:
        groups = [g for g in groups if g["resolution"] == args.resolution]

    if args.limit:
        groups = groups[:args.limit]

    print(f"Processing {len(groups)} groups...")
    results = []
    for g in groups:
        result = consolidate_group(g, dry_run=args.dry_run)
        results.append(result)
        if not args.dry_run:
            LOGGER.info("Consolidated %s: %d items → %s", g["group_key"], len(g["items"]), result["summary_id"])

    total_archived = sum(r.get("items_archived", 0) for r in results)
    print(f"Done. {len(results)} summaries created, {total_archived} items archived.")
    if args.dry_run:
        for r in results[:5]:
            print(json.dumps(r, indent=2, default=str))


if __name__ == "__main__":
    main()
