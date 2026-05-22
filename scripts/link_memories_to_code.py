"""
link_memories_to_code.py — Semantic bridge between memory items and code graph.

L7-S2 of the Level 7 "Autonomous Enterprise" implementation.

Analyzes memory_items and draws Neo4j ABOUT_CODE edges to specific
:Function, :Class, or :CodeFile nodes using strict, deterministic heuristics.

Matching strategy (no LLM, no fuzzy hallucination):
    1. Entity-to-module matching: memory item entity field → module/script names
    2. Exact qualified name matching: text contains exact function/class names
    3. Filename matching: text references specific .py files
    4. Qdrant semantic proximity: memory embedding ↔ code embedding cosine similarity
       (only above a strict threshold to prevent false positives)

All matches require a minimum confidence score. False positives in a dependency
graph are worse than missing edges.

Usage:
    python link_memories_to_code.py [--batch-size 500] [--threshold 0.55]
    python link_memories_to_code.py --dry-run

Features:
    - Batched processing (configurable batch size)
    - Idempotent (MERGE-based Cypher)
    - Multi-strategy matching with confidence scoring
    - Strict threshold to prevent false positive edges
    - Detailed stats and match provenance logging
"""

from __future__ import annotations

import argparse
import logging
import os
import re
from pathlib import Path
from typing import Any

import psycopg

LOGGER = logging.getLogger("openclaw.link_memories_to_code")
logging.basicConfig(
    level=os.environ.get("OPENCLAW_LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)

DB_DSN = os.environ.get("OPENCLAW_MEMORY_DSN", "dbname=openclaw_memory")
NEO4J_URI = os.environ.get("OPENCLAW_NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.environ.get("OPENCLAW_NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.environ.get("OPENCLAW_NEO4J_PASSWORD", "neo4jpassword")

# Matching thresholds
SEMANTIC_THRESHOLD = 0.55       # Minimum cosine similarity for semantic matches
ENTITY_MATCH_BOOST = 0.15      # Confidence boost when entity field matches
TEXT_EXACT_MATCH_BOOST = 0.20   # Confidence boost for exact qualified name in text
FILENAME_MATCH_BOOST = 0.10    # Confidence boost for filename reference
MIN_LINK_CONFIDENCE = 0.12     # Minimum total confidence to create an edge


def get_neo4j_driver():
    from neo4j import GraphDatabase
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))


# ---------------------------------------------------------------------------
# Build code reference lookup tables
# ---------------------------------------------------------------------------

def build_code_lookups(conn) -> dict[str, Any]:
    """
    Build lookup tables from code_chunks for matching.
    Returns dict with:
        qualified_names: set of all qualified names
        module_names: set of all module names
        filenames: dict of filename → file_path
        qname_to_info: dict of qualified_name → {chunk_type, module_name, file_path}
        module_to_file: dict of module_name → file_path
    """
    with conn.cursor() as cur:
        cur.execute("""
            SELECT DISTINCT qualified_name, chunk_type, module_name, file_path
            FROM code_chunks
            WHERE chunk_type IN ('function', 'method', 'class')
        """)
        rows = cur.fetchall()

    lookups = {
        "qualified_names": set(),
        "module_names": set(),
        "filenames": {},
        "qname_to_info": {},
        "module_to_file": {},
        # Short name → list of qualified names (for ambiguity detection)
        "short_name_to_qnames": {},
    }

    for qname, ctype, module_name, file_path in rows:
        lookups["qualified_names"].add(qname)
        lookups["module_names"].add(module_name)
        lookups["filenames"][Path(file_path).name] = file_path
        lookups["qname_to_info"][qname] = {
            "chunk_type": ctype,
            "module_name": module_name,
            "file_path": file_path,
        }
        lookups["module_to_file"][module_name] = file_path

        # Short name (just the function/class name without module prefix)
        short = qname.split(".")[-1]
        lookups["short_name_to_qnames"].setdefault(short, []).append(qname)

    return lookups


# ---------------------------------------------------------------------------
# Matching heuristics
# ---------------------------------------------------------------------------

# Pre-compiled patterns for code references in text
_PY_FILE_RE = re.compile(r'\b([a-zA-Z_]\w*\.py)\b')
_FUNC_DEF_RE = re.compile(r'\bdef\s+([a-zA-Z_]\w*)')
_CLASS_DEF_RE = re.compile(r'\bclass\s+([a-zA-Z_]\w*)')
_QUALIFIED_RE = re.compile(r'\b([a-zA-Z_]\w+\.[a-zA-Z_]\w+(?:\.[a-zA-Z_]\w+)?)\b')
# Module/script name patterns (e.g., "checkpoint_agent", "memory_db")
_MODULE_REF_RE = re.compile(r'\b([a-z][a-z0-9_]+(?:_[a-z0-9]+)+)\b')


def find_code_references(
    memory_item: dict[str, Any],
    lookups: dict[str, Any],
) -> list[dict[str, Any]]:
    """
    Find code references in a memory item using strict deterministic heuristics.

    Returns a list of match dicts:
        {target_type: "Function"|"Class"|"CodeFile",
         target_key: qualified_name or file_path,
         confidence: float,
         match_reasons: list[str]}
    """
    text = str(memory_item.get("text") or "")
    entity = str(memory_item.get("entity") or "")
    matches: dict[str, dict] = {}  # target_key → match info

    # --- Strategy 1: Entity-to-module matching ---
    # Entity values like "checkpoint_pipeline" may match module names
    # Uses two tiers: exact containment (high confidence) and shared prefix (lower confidence)
    if entity:
        entity_lower = entity.lower().replace("-", "_")
        entity_parts = set(entity_lower.split("_"))

        for module_name in lookups["module_names"]:
            boost = 0.0
            reason = ""

            # Tier A: Direct containment (require entity length >= 8 to avoid "worker" matching everything)
            if len(entity_lower) >= 8 and (module_name in entity_lower or entity_lower in module_name):
                boost = ENTITY_MATCH_BOOST * 2
                reason = f"entity '{entity}' contains/matches module '{module_name}'"
            else:
                # Tier B: Shared significant prefix (e.g., "checkpoint_pipeline" ↔ "checkpoint_agent")
                module_parts = set(module_name.split("_"))
                shared = entity_parts & module_parts
                # Require at least one shared word, and it must be a significant word (not "a", "the", etc.)
                significant_shared = {w for w in shared if len(w) >= 4}
                if significant_shared and len(significant_shared) >= 1:
                    # Scale by overlap ratio
                    overlap_ratio = len(significant_shared) / max(len(entity_parts), len(module_parts))
                    if overlap_ratio >= 0.3:  # At least 30% word overlap
                        boost = ENTITY_MATCH_BOOST * overlap_ratio
                        reason = f"entity '{entity}' shares [{','.join(significant_shared)}] with module '{module_name}'"

            if boost > 0:
                file_path = lookups["module_to_file"].get(module_name)
                if file_path:
                    key = f"CodeFile:{file_path}"
                    if key not in matches:
                        matches[key] = {
                            "target_type": "CodeFile",
                            "target_key": file_path,
                            "confidence": 0.0,
                            "match_reasons": [],
                        }
                    matches[key]["confidence"] += boost
                    matches[key]["match_reasons"].append(reason)

    # --- Strategy 2: Exact qualified name in text ---
    qname_matches = _QUALIFIED_RE.findall(text)
    for candidate in qname_matches:
        if candidate in lookups["qualified_names"]:
            info = lookups["qname_to_info"][candidate]
            target_type = "Class" if info["chunk_type"] == "class" else "Function"
            key = f"{target_type}:{candidate}"
            if key not in matches:
                matches[key] = {
                    "target_type": target_type,
                    "target_key": candidate,
                    "confidence": 0.0,
                    "match_reasons": [],
                }
            matches[key]["confidence"] += TEXT_EXACT_MATCH_BOOST
            matches[key]["match_reasons"].append(f"exact qualified name '{candidate}' in text")

    # --- Strategy 3: def/class name in text (unambiguous only) ---
    for pattern, label in [(_FUNC_DEF_RE, "def"), (_CLASS_DEF_RE, "class")]:
        for name in pattern.findall(text):
            candidates = lookups["short_name_to_qnames"].get(name, [])
            # Only match if unambiguous (exactly one code entity with this name)
            if len(candidates) == 1:
                qname = candidates[0]
                info = lookups["qname_to_info"][qname]
                target_type = "Class" if info["chunk_type"] == "class" else "Function"
                key = f"{target_type}:{qname}"
                if key not in matches:
                    matches[key] = {
                        "target_type": target_type,
                        "target_key": qname,
                        "confidence": 0.0,
                        "match_reasons": [],
                    }
                matches[key]["confidence"] += TEXT_EXACT_MATCH_BOOST * 0.75  # Slightly less than full qualified
                matches[key]["match_reasons"].append(f"'{label} {name}' in text → unambiguous '{qname}'")

    # --- Strategy 4: Filename references in text ---
    for filename in _PY_FILE_RE.findall(text):
        if filename in lookups["filenames"]:
            file_path = lookups["filenames"][filename]
            key = f"CodeFile:{file_path}"
            if key not in matches:
                matches[key] = {
                    "target_type": "CodeFile",
                    "target_key": file_path,
                    "confidence": 0.0,
                    "match_reasons": [],
                }
            matches[key]["confidence"] += FILENAME_MATCH_BOOST
            matches[key]["match_reasons"].append(f"filename '{filename}' in text")

    # --- Strategy 5: Module name references in text ---
    for module_ref in _MODULE_REF_RE.findall(text):
        if module_ref in lookups["module_names"]:
            file_path = lookups["module_to_file"].get(module_ref)
            if file_path:
                key = f"CodeFile:{file_path}"
                if key not in matches:
                    matches[key] = {
                        "target_type": "CodeFile",
                        "target_key": file_path,
                        "confidence": 0.0,
                        "match_reasons": [],
                    }
                matches[key]["confidence"] += FILENAME_MATCH_BOOST
                matches[key]["match_reasons"].append(f"module name '{module_ref}' in text")

    # Filter by minimum confidence
    results = [m for m in matches.values() if m["confidence"] >= MIN_LINK_CONFIDENCE]

    return results


# ---------------------------------------------------------------------------
# Semantic matching (Qdrant cross-collection similarity)
# ---------------------------------------------------------------------------

def find_semantic_matches(
    conn,
    memory_items_batch: list[dict],
    threshold: float = SEMANTIC_THRESHOLD,
    top_k: int = 3,
) -> dict[str, list[dict]]:
    """
    Find semantically similar code chunks for a batch of memory items
    using pre-computed embeddings in PostgreSQL.

    Returns: {memory_item_id → [match_dicts]}
    """
    results: dict[str, list[dict]] = {}

    # Get memory item IDs that have embeddings
    mem_ids = [m["id"] for m in memory_items_batch]
    if not mem_ids:
        return results

    with conn.cursor() as cur:
        # For each memory item with an embedding, find nearest code chunks
        # Using PostgreSQL pgvector cosine distance
        placeholders = ",".join(["%s"] * len(mem_ids))
        cur.execute(f"""
            SELECT
                mie.memory_id,
                cc.qualified_name,
                cc.chunk_type,
                cc.file_path,
                1 - (mie.embedding <=> cce.embedding) as similarity
            FROM memory_item_embeddings mie
            CROSS JOIN LATERAL (
                SELECT cce2.chunk_id, cce2.embedding
                FROM code_chunk_embeddings cce2
                ORDER BY mie.embedding <=> cce2.embedding
                LIMIT {top_k}
            ) cce
            JOIN code_chunks cc ON cc.id = cce.chunk_id
            WHERE mie.memory_id IN ({placeholders})
            AND 1 - (mie.embedding <=> cce.embedding) >= %s
        """, (*mem_ids, threshold))

        for row in cur.fetchall():
            mem_id, qname, ctype, fpath, similarity = row
            target_type = "Class" if ctype == "class" else "Function"
            match = {
                "target_type": target_type,
                "target_key": qname,
                "confidence": float(similarity) * 0.5,  # Scale semantic to not dominate heuristic
                "match_reasons": [f"semantic similarity {similarity:.3f} to '{qname}'"],
            }
            results.setdefault(mem_id, []).append(match)

    return results


# ---------------------------------------------------------------------------
# Merge and deduplicate matches
# ---------------------------------------------------------------------------

def merge_matches(
    heuristic: list[dict],
    semantic: list[dict],
) -> list[dict]:
    """
    Merge heuristic and semantic matches, combining confidence for
    items matched by both strategies.
    """
    merged: dict[str, dict] = {}

    for m in heuristic + semantic:
        key = f"{m['target_type']}:{m['target_key']}"
        if key in merged:
            merged[key]["confidence"] += m["confidence"]
            merged[key]["match_reasons"].extend(m["match_reasons"])
        else:
            merged[key] = dict(m)

    # Re-filter by minimum confidence after merging
    return [m for m in merged.values() if m["confidence"] >= MIN_LINK_CONFIDENCE]


# ---------------------------------------------------------------------------
# Neo4j edge creation
# ---------------------------------------------------------------------------

def ensure_memory_nodes(
    neo4j_session,
    edges: list[dict[str, Any]],
    memory_items_lookup: dict[str, dict],
) -> int:
    """
    Ensure MemoryItem nodes exist in Neo4j for all memory items
    referenced in the edges. Uses MERGE for idempotency.
    """
    mem_ids = list({e["memory_id"] for e in edges})
    if not mem_ids:
        return 0

    params = []
    for mid in mem_ids:
        item = memory_items_lookup.get(mid, {})
        params.append({
            "id": mid,
            "memory_type": item.get("memory_type"),
            "entity": item.get("entity"),
            "confidence": item.get("confidence"),
        })

    neo4j_session.run(
        """
        UNWIND $params AS p
        MERGE (mi:MemoryItem {id: p.id})
        SET mi.memory_type = p.memory_type,
            mi.entity = p.entity,
            mi.confidence = p.confidence,
            mi.synced_at = datetime()
        """,
        params=params,
    )
    return len(params)


def create_about_code_edges(
    neo4j_session,
    edges: list[dict[str, Any]],
    stats: dict,
    memory_items_lookup: dict[str, dict] | None = None,
) -> None:
    """
    Create ABOUT_CODE edges in Neo4j.
    Each edge dict has: memory_id, target_type, target_key, confidence, match_reasons
    """
    if not edges:
        return

    # Ensure MemoryItem nodes exist for all referenced items
    if memory_items_lookup:
        ensure_memory_nodes(neo4j_session, edges, memory_items_lookup)

    # Group by target type for efficient batched queries
    func_edges = [e for e in edges if e["target_type"] == "Function"]
    class_edges = [e for e in edges if e["target_type"] == "Class"]
    file_edges = [e for e in edges if e["target_type"] == "CodeFile"]

    if func_edges:
        neo4j_session.run(
            """
            UNWIND $params AS p
            MATCH (mi:MemoryItem {id: p.memory_id})
            MATCH (fn:Function {qualified_name: p.target_key})
            MERGE (mi)-[r:ABOUT_CODE]->(fn)
            SET r.confidence = p.confidence,
                r.match_reasons = p.match_reasons,
                r.updated_at = datetime()
            """,
            params=[{
                "memory_id": e["memory_id"],
                "target_key": e["target_key"],
                "confidence": e["confidence"],
                "match_reasons": e["match_reasons"],
            } for e in func_edges],
        )
        stats["function_edges"] += len(func_edges)

    if class_edges:
        neo4j_session.run(
            """
            UNWIND $params AS p
            MATCH (mi:MemoryItem {id: p.memory_id})
            MATCH (c:Class {qualified_name: p.target_key})
            MERGE (mi)-[r:ABOUT_CODE]->(c)
            SET r.confidence = p.confidence,
                r.match_reasons = p.match_reasons,
                r.updated_at = datetime()
            """,
            params=[{
                "memory_id": e["memory_id"],
                "target_key": e["target_key"],
                "confidence": e["confidence"],
                "match_reasons": e["match_reasons"],
            } for e in class_edges],
        )
        stats["class_edges"] += len(class_edges)

    if file_edges:
        neo4j_session.run(
            """
            UNWIND $params AS p
            MATCH (mi:MemoryItem {id: p.memory_id})
            MATCH (f:CodeFile {path: p.target_key})
            MERGE (mi)-[r:ABOUT_CODE]->(f)
            SET r.confidence = p.confidence,
                r.match_reasons = p.match_reasons,
                r.updated_at = datetime()
            """,
            params=[{
                "memory_id": e["memory_id"],
                "target_key": e["target_key"],
                "confidence": e["confidence"],
                "match_reasons": e["match_reasons"],
            } for e in file_edges],
        )
        stats["file_edges"] += len(file_edges)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def fetch_memory_items_batch(conn, offset: int, limit: int) -> list[dict]:
    """Fetch a batch of active memory items."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT id, text, entity, memory_type, confidence
            FROM memory_items
            WHERE status = 'active'
            ORDER BY id
            OFFSET %s LIMIT %s
        """, (offset, limit))
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]


def main():
    parser = argparse.ArgumentParser(
        description="Link memory items to code graph nodes via ABOUT_CODE edges."
    )
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--threshold", type=float, default=SEMANTIC_THRESHOLD,
                        help=f"Semantic similarity threshold (default: {SEMANTIC_THRESHOLD})")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-semantic", action="store_true",
                        help="Skip semantic matching (heuristic only)")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    conn = psycopg.connect(DB_DSN)
    driver = get_neo4j_driver() if not args.dry_run else None

    stats = {
        "total_items": 0,
        "items_with_matches": 0,
        "function_edges": 0,
        "class_edges": 0,
        "file_edges": 0,
        "heuristic_matches": 0,
        "semantic_matches": 0,
    }

    try:
        # Build lookup tables once
        lookups = build_code_lookups(conn)
        LOGGER.info(
            "Code lookups: %d qualified_names, %d modules, %d files",
            len(lookups["qualified_names"]),
            len(lookups["module_names"]),
            len(lookups["filenames"]),
        )

        offset = 0
        while True:
            batch = fetch_memory_items_batch(conn, offset, args.batch_size)
            if not batch:
                break

            batch_edges = []

            # Phase 1: Heuristic matching
            for item in batch:
                stats["total_items"] += 1
                heuristic = find_code_references(item, lookups)
                stats["heuristic_matches"] += len(heuristic)

                # Phase 2: Semantic matching (if not skipped)
                semantic = []
                if not args.skip_semantic:
                    sem_results = find_semantic_matches(
                        conn, [item], threshold=args.threshold, top_k=3
                    )
                    semantic = sem_results.get(item["id"], [])
                    stats["semantic_matches"] += len(semantic)

                # Merge
                merged = merge_matches(heuristic, semantic)
                if merged:
                    stats["items_with_matches"] += 1
                    for match in merged:
                        edge = {
                            "memory_id": item["id"],
                            **match,
                        }
                        batch_edges.append(edge)

                        if args.verbose:
                            LOGGER.info(
                                "MATCH %s → %s:%s (conf=%.3f, reasons=%s)",
                                item["id"][:12],
                                match["target_type"],
                                match["target_key"],
                                match["confidence"],
                                match["match_reasons"],
                            )

            # Write edges to Neo4j
            if batch_edges and not args.dry_run:
                # Build lookup for memory items in this batch
                mem_lookup = {item["id"]: item for item in batch}
                with driver.session() as session:
                    create_about_code_edges(session, batch_edges, stats, mem_lookup)

            offset += args.batch_size
            LOGGER.info(
                "Processed %d items, %d edges so far",
                stats["total_items"],
                stats["function_edges"] + stats["class_edges"] + stats["file_edges"],
            )

    finally:
        conn.close()
        if driver:
            driver.close()

    total_edges = stats["function_edges"] + stats["class_edges"] + stats["file_edges"]

    print(f"TOTAL_ITEMS={stats['total_items']}")
    print(f"ITEMS_WITH_MATCHES={stats['items_with_matches']}")
    print(f"TOTAL_EDGES={total_edges}")
    print(f"  Function={stats['function_edges']}")
    print(f"  Class={stats['class_edges']}")
    print(f"  CodeFile={stats['file_edges']}")
    print(f"HEURISTIC_MATCHES={stats['heuristic_matches']}")
    print(f"SEMANTIC_MATCHES={stats['semantic_matches']}")


if __name__ == "__main__":
    main()
