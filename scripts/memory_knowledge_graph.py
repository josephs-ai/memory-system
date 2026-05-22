"""
memory_knowledge_graph.py — Semantic relationships between memory items.

Part of P4: Memory Knowledge Graph.

Creates typed, traversable links between memory items in Neo4j:
- SUPERSEDES: Decision X replaced Decision Y
- CAUSED_BY: Lesson A was caused by Incident B
- CONTRADICTS: Preference P conflicts with Preference Q
- RELATES_TO: General semantic relationship
- SUPPORTS: Evidence/fact that supports a decision
- DEPENDS_ON: Item requires another item as context

Relationships are detected via deterministic heuristic matching:
1. Explicit supersedes references (existing supersedes field)
2. Entity+property co-occurrence (same entity, related properties)
3. Temporal proximity + semantic overlap
4. Memory type inference (decisions relate to lessons, facts support rules)

No LLM calls — purely structural + heuristic analysis.
"""

from __future__ import annotations

import logging
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

LOGGER = logging.getLogger("openclaw.memory_knowledge_graph")

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

DB_DSN = os.environ.get("OPENCLAW_MEMORY_DSN", "dbname=openclaw_memory")

# Relationship types
REL_SUPERSEDES = "SUPERSEDES"
REL_CAUSED_BY = "CAUSED_BY"
REL_CONTRADICTS = "CONTRADICTS"
REL_RELATES_TO = "RELATES_TO"
REL_SUPPORTS = "SUPPORTS"
REL_DEPENDS_ON = "DEPENDS_ON"

ALL_RELATIONSHIP_TYPES = [
    REL_SUPERSEDES, REL_CAUSED_BY, REL_CONTRADICTS,
    REL_RELATES_TO, REL_SUPPORTS, REL_DEPENDS_ON,
]

# Type-based inference rules
TYPE_RELATIONSHIPS = {
    ("decision", "lesson_learned"): REL_CAUSED_BY,
    ("lesson_learned", "decision"): REL_RELATES_TO,
    ("fact", "rule"): REL_SUPPORTS,
    ("fact", "architecture_rule"): REL_SUPPORTS,
    ("observation", "decision"): REL_RELATES_TO,
    ("decision", "decision"): REL_SUPERSEDES,  # Only if same entity+property
    ("rule", "rule"): REL_SUPERSEDES,
}

# Temporal proximity threshold for relationship inference
TEMPORAL_PROXIMITY_HOURS = 48


def detect_supersedes_links(items: list[dict]) -> list[dict]:
    """
    Detect SUPERSEDES relationships from explicit supersedes fields.
    """
    links = []
    item_map = {i["id"]: i for i in items if i.get("id")}

    for item in items:
        supersedes_id = item.get("supersedes")
        if supersedes_id and supersedes_id in item_map:
            links.append({
                "source_id": item["id"],
                "target_id": supersedes_id,
                "relationship": REL_SUPERSEDES,
                "confidence": 1.0,
                "reason": "explicit_supersedes_field",
            })

    return links


def detect_entity_links(items: list[dict]) -> list[dict]:
    """
    Detect relationships based on shared entity+property combinations.

    Items with the same entity and property but different values likely
    represent updates/contradictions.
    """
    links = []
    by_entity_prop: dict[tuple, list[dict]] = defaultdict(list)

    for item in items:
        entity = item.get("entity")
        prop = item.get("property")
        if entity and prop:
            by_entity_prop[(entity, prop)].append(item)

    for (entity, prop), group in by_entity_prop.items():
        if len(group) < 2:
            continue

        # Sort by first_seen (newest first)
        sorted_items = sorted(
            group,
            key=lambda x: x.get("first_seen") or datetime.min.replace(tzinfo=timezone.utc),
            reverse=True,
        )

        for i in range(len(sorted_items) - 1):
            newer = sorted_items[i]
            older = sorted_items[i + 1]

            # Same entity+property with different values → supersedes
            newer_val = (newer.get("value") or "").strip().lower()
            older_val = (older.get("value") or "").strip().lower()

            if newer_val and older_val and newer_val != older_val:
                rel = REL_SUPERSEDES
                confidence = 0.85
            else:
                rel = REL_RELATES_TO
                confidence = 0.6

            links.append({
                "source_id": newer["id"],
                "target_id": older["id"],
                "relationship": rel,
                "confidence": confidence,
                "reason": f"shared_entity_property:{entity}.{prop}",
            })

    return links


def detect_type_based_links(items: list[dict]) -> list[dict]:
    """
    Detect relationships based on memory type + temporal proximity.

    E.g., a lesson_learned created shortly after a decision → CAUSED_BY.
    """
    links = []

    # Group by scope for efficiency
    by_scope: dict[str, list[dict]] = defaultdict(list)
    for item in items:
        scope = item.get("scope") or "global"
        by_scope[scope].append(item)

    for scope, scope_items in by_scope.items():
        for i, item_a in enumerate(scope_items):
            type_a = item_a.get("memory_type") or ""
            ts_a = item_a.get("first_seen")
            if not ts_a:
                continue
            if ts_a.tzinfo is None:
                ts_a = ts_a.replace(tzinfo=timezone.utc)

            for j, item_b in enumerate(scope_items):
                if i >= j:
                    continue
                type_b = item_b.get("memory_type") or ""
                ts_b = item_b.get("first_seen")
                if not ts_b:
                    continue
                if ts_b.tzinfo is None:
                    ts_b = ts_b.replace(tzinfo=timezone.utc)

                # Check temporal proximity
                time_diff = abs((ts_a - ts_b).total_seconds()) / 3600.0
                if time_diff > TEMPORAL_PROXIMITY_HOURS:
                    continue

                # Check type-based relationship
                rel = TYPE_RELATIONSHIPS.get((type_a, type_b))
                if not rel:
                    rel = TYPE_RELATIONSHIPS.get((type_b, type_a))
                    if rel:
                        # Swap direction
                        item_a, item_b = item_b, item_a

                if rel:
                    # For SUPERSEDES between same types, require same entity
                    if rel == REL_SUPERSEDES:
                        if item_a.get("entity") != item_b.get("entity"):
                            continue

                    links.append({
                        "source_id": item_a["id"],
                        "target_id": item_b["id"],
                        "relationship": rel,
                        "confidence": 0.6,
                        "reason": f"type_proximity:{type_a}→{type_b}",
                    })

    return links


def detect_all_links(items: list[dict]) -> list[dict]:
    """
    Run all link detection strategies and deduplicate.
    """
    all_links = []
    all_links.extend(detect_supersedes_links(items))
    all_links.extend(detect_entity_links(items))
    all_links.extend(detect_type_based_links(items))

    # Deduplicate: keep highest confidence for each (source, target, rel) triple
    seen: dict[tuple, dict] = {}
    for link in all_links:
        key = (link["source_id"], link["target_id"], link["relationship"])
        if key not in seen or link["confidence"] > seen[key]["confidence"]:
            seen[key] = link

    return list(seen.values())


def store_links_neo4j(links: list[dict]) -> int:
    """
    Store memory links in Neo4j.

    Creates MemoryItem nodes and typed relationships.
    Returns count of relationships created.
    """
    try:
        from graph_store_neo4j import get_neo4j_driver
    except ImportError:
        LOGGER.error("graph_store_neo4j not available")
        return 0

    driver = get_neo4j_driver()
    if not driver:
        return 0

    count = 0
    with driver.session() as session:
        for link in links:
            try:
                rel_type = link["relationship"]
                session.run(
                    f"""
                    MERGE (a:MemoryItem {{id: $source_id}})
                    MERGE (b:MemoryItem {{id: $target_id}})
                    MERGE (a)-[r:{rel_type}]->(b)
                    SET r.confidence = $confidence,
                        r.reason = $reason,
                        r.detected_at = datetime()
                    """,
                    source_id=link["source_id"],
                    target_id=link["target_id"],
                    confidence=link["confidence"],
                    reason=link["reason"],
                )
                count += 1
            except Exception:
                LOGGER.debug("Failed to store link %s", link, exc_info=True)

    driver.close()
    return count


def query_related_memories(
    item_id: str,
    *,
    relationship_types: list[str] | None = None,
    max_depth: int = 2,
) -> list[dict]:
    """
    Traverse the memory knowledge graph to find related items.

    Returns list of related memory dicts with relationship path info.
    """
    try:
        from graph_store_neo4j import get_neo4j_driver
    except ImportError:
        return []

    driver = get_neo4j_driver()
    if not driver:
        return []

    rel_filter = ""
    if relationship_types:
        types = "|".join(relationship_types)
        rel_filter = f":{types}"

    results = []
    with driver.session() as session:
        records = session.run(
            f"""
            MATCH path = (start:MemoryItem {{id: $item_id}})-[r{rel_filter}*1..{max_depth}]-(related:MemoryItem)
            WITH related, r, length(path) as depth
            RETURN DISTINCT related.id AS id, type(r[0]) AS relationship, depth
            ORDER BY depth, related.id
            LIMIT 20
            """,
            item_id=item_id,
        )
        for record in records:
            results.append({
                "id": record["id"],
                "relationship": record["relationship"],
                "depth": record["depth"],
            })

    driver.close()
    return results


def fetch_items_for_linking(
    *,
    limit: int = 5000,
) -> list[dict]:
    """Fetch active memory items for link detection."""
    import psycopg
    conn = psycopg.connect(DB_DSN)
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, text, memory_type, scope, entity, property, value,
                       first_seen, last_confirmed, supersedes, confidence
                FROM memory_items
                WHERE status = 'active'
                  AND memory_type NOT IN ('summary', 'feature_summary', 'project_summary')
                ORDER BY first_seen DESC NULLS LAST
                LIMIT %s
                """,
                (limit,),
            )
            cols = [d[0] for d in cur.description]
            return [dict(zip(cols, row)) for row in cur.fetchall()]
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Memory knowledge graph")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("detect", help="Detect links from active items")
    sub.add_parser("store", help="Detect and store links in Neo4j")

    query_p = sub.add_parser("query", help="Query related memories")
    query_p.add_argument("item_id")
    query_p.add_argument("--depth", type=int, default=2)

    args = parser.parse_args()

    if args.command == "detect":
        items = fetch_items_for_linking()
        links = detect_all_links(items)
        print(f"Detected {len(links)} links from {len(items)} items")
        for rel_type in ALL_RELATIONSHIP_TYPES:
            count = sum(1 for lnk in links if lnk["relationship"] == rel_type)
            if count:
                print(f"  {rel_type}: {count}")

    elif args.command == "store":
        items = fetch_items_for_linking()
        links = detect_all_links(items)
        stored = store_links_neo4j(links)
        print(f"Stored {stored}/{len(links)} links in Neo4j")

    elif args.command == "query":
        results = query_related_memories(args.item_id, max_depth=args.depth)
        print(json.dumps(results, indent=2))

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
