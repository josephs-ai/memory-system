"""
Compact and deduplicate memory items — groups by scope/entity/property
and merges redundant items to reduce storage bloat.
"""
from __future__ import annotations

import hashlib
import sys
from datetime import datetime, timezone
from collections import defaultdict
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from memory_db import fetch_memory_items, upsert_memory_items, close_pool

def compact_memories():
    items = fetch_memory_items(["active"])

    # Filter for durable/reinforced that are eligible for compaction
    candidates = [
        i for i in items
        if i.get("lifecycle_state") in ("durable", "reinforced")
        and i.get("retrieval_eligibility") in ("normal", "always")
        and i.get("memory_type") in ("fact", "observation", "rule", "architecture_rule", "decision", "lesson_learned")
    ]

    # Group by scope (project_id), memory_type, entity, property, and domain tags
    groups = defaultdict(list)
    for item in candidates:
        scope = item.get("scope") or "global"
        mtype = item.get("memory_type") or "general"
        entity = item.get("entity") or ""
        prop = item.get("property") or ""
        tags = tuple(sorted(item.get("tags") or []))

        # Only group if we have meaningful structural fields to group on
        if entity and prop:
            groups[(scope, mtype, entity, prop, tags)].append(item)

    updates = []
    compacted_count = 0

    for key, group_items in groups.items():
        # Only compact if there are multiple granular items in the exact same structural bucket
        if len(group_items) > 1:
            scope, mtype, entity, prop, tags = key

            # Sort by ID for deterministic combination
            group_items.sort(key=lambda x: x.get("id", ""))

            combined_text = f"Compacted {mtype} regarding {entity} {prop}:\n"
            for i in group_items:
                combined_text += f"- {i.get('text', '')}\n"

            new_id = f"mem-comp-{hashlib.sha1(combined_text.encode('utf-8')).hexdigest()[:12]}"

            datetime.now(timezone.utc).isoformat()

            new_item = {
                "id": new_id,
                "text": combined_text.strip(),
                "memory_type": mtype,
                "scope": scope,
                "entity": entity,
                "property": prop,
                "value": "COMPACTED_SUMMARY",
                "status": "active",
                "confidence": min(1.0, max(float(i.get("confidence", 0.0) or 0.0) for i in group_items) + 0.05),
                "retention_priority": min(1.0, max(float(i.get("retention_priority", 0.0) or 0.0) for i in group_items) + 0.05),
                "lifecycle_state": "durable",
                "retrieval_eligibility": "normal",
                "tags": list(tags),
                "source_agent": "system_compaction",
                "source_session": "compaction_cycle",
            }

            updates.append(new_item)
            compacted_count += 1

            for old in group_items:
                old["lifecycle_state"] = "archived"
                old["retrieval_eligibility"] = "suppressed"
                old["supersedes"] = new_id
                updates.append(old)

    if updates:
        upsert_memory_items(updates)
        print(f"Compacted {compacted_count} groups. Total DB upserts (new items + archived originals): {len(updates)}")
    else:
        print("No compaction needed.")

if __name__ == "__main__":
    try:
        compact_memories()
    finally:
        close_pool()
