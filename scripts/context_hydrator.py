"""
context_hydrator.py — Unified context retrieval fusing code + memory + graph.

Slice 6 (S6) of the Zero-Reread Architecture.

Replaces manual file reads with a single retrieval call that delivers:
    1. Relevant code chunks (from Qdrant code_chunks collection)
    2. Relevant memory items (from Qdrant memory_items collection)
    3. Graph-derived context (dependencies, callers from Neo4j)

Token-optimized output: primary hits get full body, secondary hits get
signature + docstring only, dependency map is compact.

Usage as CLI:
    python context_hydrator.py "fetch memory items for embedding"
    python context_hydrator.py "what would break if I change memory_db" --limit 5

Usage as library:
    from context_hydrator import build_context_pack
    pack = build_context_pack("fix the embedding pipeline", limit=8)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any

import psycopg
from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models

LOGGER = logging.getLogger("openclaw.context_hydrator")
logging.basicConfig(
    level=os.environ.get("OPENCLAW_LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)

DB_DSN = (os.environ.get("OPENCLAW_MEMORY_DSN") or os.environ.get("OPENCLAW_MEMORY_DB_DSN") or "dbname=openclaw_memory")  # honor both canonical DSN env names
QDRANT_HOST = os.environ.get("OPENCLAW_QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.environ.get("OPENCLAW_QDRANT_PORT", "6333"))
NEO4J_URI = os.environ.get("OPENCLAW_NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.environ.get("OPENCLAW_NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.environ.get("OPENCLAW_NEO4J_PASSWORD", "neo4jpassword")

CODE_COLLECTION = "code_chunks"
MEMORY_COLLECTION = "memory_items"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
RERANK_MODEL = os.environ.get("OPENCLAW_RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
RERANK_CAP = int(os.environ.get("OPENCLAW_RERANK_CAP", "20"))
RERANK_TEXT_CHAR_CAP = int(os.environ.get("OPENCLAW_RERANK_TEXT_CHAR_CAP", "1000"))

# Token budget constants (approximate, 1 token ≈ 4 chars)
PRIMARY_BODY_MAX_CHARS = 3000    # Full body for top hits
SECONDARY_BODY_MAX_CHARS = 0     # Signature + docstring only for secondary hits
DEPENDENCY_SIG_MAX_CHARS = 120   # Compact signatures in dependency map
MEMORY_TEXT_MAX_CHARS = 500      # Memory item text cap

# Lazy-loaded globals
_model = None
_reranker = None
_qdrant = None
_neo4j_driver = None


# ---------------------------------------------------------------------------
# Lazy initialization
# ---------------------------------------------------------------------------

def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(MODEL_NAME, device="cpu")
    return _model


def _get_reranker():
    """Lazy-load the cross-encoder reranker."""
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder
        try:
            _reranker = CrossEncoder(RERANK_MODEL, device="cpu")
        except TypeError:
            _reranker = CrossEncoder(RERANK_MODEL)
        LOGGER.info("Loaded reranker: %s", RERANK_MODEL)
    return _reranker


def _get_qdrant() -> QdrantClient:
    global _qdrant
    if _qdrant is None:
        _qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    return _qdrant


def _get_neo4j():
    global _neo4j_driver
    if _neo4j_driver is None:
        from neo4j import GraphDatabase
        _neo4j_driver = GraphDatabase.driver(
            NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD)
        )
    return _neo4j_driver


# ---------------------------------------------------------------------------
# Vector search
# ---------------------------------------------------------------------------

def _search_code_chunks(
    query_vector: list[float],
    *,
    limit: int = 8,
    chunk_type_filter: str | None = None,
) -> list[dict[str, Any]]:
    """Search the code_chunks Qdrant collection."""
    client = _get_qdrant()

    filter_cond = None
    if chunk_type_filter:
        filter_cond = qdrant_models.Filter(
            must=[
                qdrant_models.FieldCondition(
                    key="chunk_type",
                    match=qdrant_models.MatchValue(value=chunk_type_filter),
                )
            ]
        )

    results = client.query_points(
        collection_name=CODE_COLLECTION,
        query=query_vector,
        limit=limit,
        with_payload=True,
        query_filter=filter_cond,
    )

    return [
        {
            "source": "code",
            "score": pt.score,
            "chunk_id": pt.payload.get("chunk_id"),
            "qualified_name": pt.payload.get("qualified_name"),
            "chunk_type": pt.payload.get("chunk_type"),
            "module_name": pt.payload.get("module_name"),
            "file_path": pt.payload.get("file_path"),
            "signature": pt.payload.get("signature"),
            "docstring": pt.payload.get("docstring"),
            "line_start": pt.payload.get("line_start"),
            "line_end": pt.payload.get("line_end"),
            "class_name": pt.payload.get("class_name"),
            "project_id": pt.payload.get("project_id"),
        }
        for pt in results.points
    ]


def _search_memory_items(
    query_vector: list[float],
    *,
    limit: int = 5,
) -> list[dict[str, Any]]:
    """Search the memory_items Qdrant collection."""
    client = _get_qdrant()

    results = client.query_points(
        collection_name=MEMORY_COLLECTION,
        query=query_vector,
        limit=limit,
        with_payload=True,
    )

    return [
        {
            "source": "memory",
            "score": pt.score,
            "id": pt.payload.get("id"),
            "text": pt.payload.get("text"),
            "memory_type": pt.payload.get("memory_type"),
            "scope": pt.payload.get("scope"),
            "entity": pt.payload.get("entity"),
            "confidence": pt.payload.get("confidence"),
        }
        for pt in results.points
    ]


# ---------------------------------------------------------------------------
# Cross-encoder reranking
# ---------------------------------------------------------------------------

def _build_rerank_text(hit: dict[str, Any]) -> str:
    """Build the text representation of a hit for cross-encoder scoring."""
    if hit["source"] == "code":
        parts = []
        if hit.get("qualified_name"):
            parts.append(hit["qualified_name"])
        if hit.get("signature"):
            parts.append(hit["signature"])
        if hit.get("docstring"):
            parts.append(hit["docstring"])
        text = "\n".join(parts)
    else:  # memory
        text = hit.get("text") or ""

    if len(text) > RERANK_TEXT_CHAR_CAP:
        text = text[:RERANK_TEXT_CHAR_CAP]
    return text


def _rerank_hits(
    query: str,
    hits: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Rerank merged code + memory hits using the cross-encoder.

    Caps at RERANK_CAP to bound latency. Items beyond the cap retain
    their original vector scores and are appended after the reranked set.
    """
    if not hits:
        return hits

    reranker = _get_reranker()
    capped = hits[:RERANK_CAP]
    overflow = hits[RERANK_CAP:]

    pairs = [(query, _build_rerank_text(h)) for h in capped]
    scores = reranker.predict(pairs)

    # strict=True: scores are produced from `capped` 1:1; a mismatch would
    # assign rerank scores to the wrong hits and silently corrupt ranking.
    for hit, score in zip(capped, scores, strict=True):
        hit["rerank_score"] = float(score)

    capped.sort(key=lambda x: x.get("rerank_score", -1e9), reverse=True)

    if overflow:
        capped.extend(overflow)

    return capped


# ---------------------------------------------------------------------------
# PostgreSQL: fetch full bodies for code chunks
# ---------------------------------------------------------------------------

def _fetch_chunk_bodies(conn, chunk_ids: list[str]) -> dict[str, str]:
    """Fetch full body text for specific chunk IDs from PostgreSQL."""
    if not chunk_ids:
        return {}
    with conn.cursor() as cur:
        placeholders = ",".join(["%s"] * len(chunk_ids))
        cur.execute(
            f"SELECT id, body FROM code_chunks WHERE id IN ({placeholders})",
            tuple(chunk_ids),
        )
        return {row[0]: row[1] for row in cur.fetchall()}


# ---------------------------------------------------------------------------
# Neo4j: graph traversal for dependencies and callers
# ---------------------------------------------------------------------------

def _get_callers(qualified_name: str) -> list[dict[str, str]]:
    """Find all functions that CALL this function."""
    driver = _get_neo4j()
    with driver.session() as session:
        result = session.run(
            """
            MATCH (caller:Function)-[:CALLS]->(callee:Function {qualified_name: $qname})
            RETURN caller.qualified_name AS name,
                   caller.signature AS signature,
                   caller.file_path AS file_path
            ORDER BY caller.qualified_name
            LIMIT 15
            """,
            qname=qualified_name,
        )
        return [dict(r) for r in result]


def _get_callees(qualified_name: str) -> list[dict[str, str]]:
    """Find all functions that this function CALLS."""
    driver = _get_neo4j()
    with driver.session() as session:
        result = session.run(
            """
            MATCH (caller:Function {qualified_name: $qname})-[:CALLS]->(callee:Function)
            RETURN callee.qualified_name AS name,
                   callee.signature AS signature,
                   callee.file_path AS file_path
            ORDER BY callee.qualified_name
            LIMIT 15
            """,
            qname=qualified_name,
        )
        return [dict(r) for r in result]


def _get_file_imports(file_path: str) -> list[dict[str, Any]]:
    """Get modules imported by this file."""
    driver = _get_neo4j()
    with driver.session() as session:
        result = session.run(
            """
            MATCH (f:CodeFile {path: $path})-[:IMPORTS]->(m:CodeModule)
            RETURN m.name AS module, m.is_local AS is_local
            ORDER BY m.name
            """,
            path=file_path,
        )
        return [dict(r) for r in result]


def _get_dependents(module_name: str) -> list[str]:
    """Get files that import this module (blast radius)."""
    driver = _get_neo4j()
    with driver.session() as session:
        result = session.run(
            """
            MATCH (f:CodeFile)-[:IMPORTS]->(m:CodeModule {name: $name})
            RETURN f.path AS path
            ORDER BY f.path
            LIMIT 20
            """,
            name=module_name,
        )
        return [r["path"] for r in result]


def _get_class_methods(class_qname: str) -> list[dict[str, str]]:
    """Get methods defined on a class."""
    driver = _get_neo4j()
    with driver.session() as session:
        result = session.run(
            """
            MATCH (c:Class {qualified_name: $qname})-[:HAS_METHOD]->(fn:Function)
            RETURN fn.qualified_name AS name, fn.signature AS signature
            ORDER BY fn.qualified_name
            """,
            qname=class_qname,
        )
        return [dict(r) for r in result]


# ---------------------------------------------------------------------------
# Token-optimized formatting
# ---------------------------------------------------------------------------

def _format_code_primary(hit: dict, body: str | None) -> dict[str, Any]:
    """Format a primary code hit with full body (token-budgeted)."""
    result = {
        "type": "code",
        "qualified_name": hit["qualified_name"],
        "chunk_type": hit["chunk_type"],
        "file": _short_path(hit.get("file_path")),
        "lines": f"L{hit.get('line_start', '?')}-{hit.get('line_end', '?')}",
        "score": round(hit["score"], 4),
    }

    if hit.get("signature"):
        result["signature"] = hit["signature"]

    if hit.get("docstring"):
        result["docstring"] = hit["docstring"]

    if body:
        if len(body) > PRIMARY_BODY_MAX_CHARS:
            result["body"] = body[:PRIMARY_BODY_MAX_CHARS] + "\n# ... truncated"
            result["body_truncated"] = True
        else:
            result["body"] = body
    elif hit.get("signature"):
        result["body_note"] = "signature-only (body in PostgreSQL)"

    return result


def _format_code_secondary(hit: dict) -> dict[str, Any]:
    """Format a secondary code hit — signature + docstring only."""
    result = {
        "type": "code_ref",
        "qualified_name": hit["qualified_name"],
        "chunk_type": hit["chunk_type"],
        "file": _short_path(hit.get("file_path")),
        "lines": f"L{hit.get('line_start', '?')}-{hit.get('line_end', '?')}",
        "score": round(hit["score"], 4),
    }
    if hit.get("signature"):
        result["signature"] = hit["signature"]
    if hit.get("docstring"):
        result["docstring"] = hit["docstring"][:200]
    return result


def _format_memory_item(hit: dict) -> dict[str, Any]:
    """Format a memory item hit."""
    text = hit.get("text") or ""
    if len(text) > MEMORY_TEXT_MAX_CHARS:
        text = text[:MEMORY_TEXT_MAX_CHARS] + "..."

    result = {
        "type": "memory",
        "memory_type": hit.get("memory_type"),
        "text": text,
        "entity": hit.get("entity"),
        "confidence": hit.get("confidence"),
        "score": round(hit["score"], 4),
    }

    # Summary drill-down metadata (P2: progressive summarization)
    if hit.get("memory_type") == "summary":
        result["is_summary"] = True
        result["hint"] = f"[Summary of consolidated items. ID: {hit.get('id', '?')}]"

    return result


def _format_dependency_entry(dep: dict) -> str:
    """Compact one-line dependency entry."""
    sig = dep.get("signature") or dep.get("name") or "?"
    if len(sig) > DEPENDENCY_SIG_MAX_CHARS:
        sig = sig[:DEPENDENCY_SIG_MAX_CHARS] + "..."
    return sig


def _short_path(path: str | None) -> str:
    """Shorten absolute paths to just filename."""
    if not path:
        return "?"
    return Path(path).name


# ---------------------------------------------------------------------------
# Core hydration logic
# ---------------------------------------------------------------------------

def build_context_pack(
    query: str,
    *,
    code_limit: int = 8,
    memory_limit: int = 5,
    primary_count: int = 3,
    include_graph: bool = True,
    rerank: bool = True,
) -> dict[str, Any]:
    """
    Build a unified context pack for a query.

    Args:
        query: Natural language or code-related query
        code_limit: Max code chunk results from vector search
        memory_limit: Max memory item results from vector search
        primary_count: Number of top code hits that get full body
        include_graph: Whether to include Neo4j graph traversal
        rerank: Whether to apply cross-encoder reranking (L7-S1)

    Returns:
        A token-optimized context pack dict with:
        - code_context: list of code chunks (primary with body, secondary with sig only)
        - memory_context: list of relevant memory items
        - dependency_map: dict of graph-derived relationships
        - meta: search metadata (includes rerank info when enabled)
    """
    import time
    t0 = time.monotonic()

    model = _get_model()
    query_vector = model.encode(query, normalize_embeddings=True).tolist()

    # --- Phase 1: Dual vector search ---
    code_hits = _search_code_chunks(query_vector, limit=code_limit)
    memory_hits = _search_memory_items(query_vector, limit=memory_limit) if memory_limit > 0 else []

    t_vector = time.monotonic()

    # --- Phase 1.5: Feedback boost (P0: closed-loop retrieval) ---
    feedback_applied = False
    try:
        from feedback_score_engine import feedback_boost_batch, classify_query_type
        query_type = classify_query_type(query)
        # Collect item IDs from memory hits (code chunks don't have retrieval feedback)
        mem_ids = [h.get("id") for h in memory_hits if h.get("id")]
        if mem_ids:
            boosts = feedback_boost_batch(mem_ids, query_type=query_type)
            for h in memory_hits:
                iid = h.get("id")
                if iid and iid in boosts and boosts[iid] != 0.0:
                    h["feedback_boost"] = boosts[iid]
                    # Apply as additive score modifier (conservative 0.15 weight)
                    h["score"] = h.get("score", 0.0) + boosts[iid] * 0.15
            feedback_applied = True
    except Exception:
        LOGGER.debug("Feedback boost in context_hydrator failed, continuing without", exc_info=True)

    # --- Phase 2: Cross-encoder reranking (L7-S1) ---
    rerank_applied = False
    rerank_ms = 0.0
    if rerank and (code_hits or memory_hits):
        merged = code_hits + memory_hits
        t_rerank_start = time.monotonic()
        reranked = _rerank_hits(query, merged)
        t_rerank_end = time.monotonic()
        rerank_ms = round((t_rerank_end - t_rerank_start) * 1000, 1)
        rerank_applied = True

        # Re-split into code and memory, preserving reranked order
        code_hits = [h for h in reranked if h["source"] == "code"][:code_limit]
        memory_hits = [h for h in reranked if h["source"] == "memory"][:memory_limit]

    # --- Phase 3: Fetch full bodies for primary code hits ---
    primary_hits = code_hits[:primary_count]
    secondary_hits = code_hits[primary_count:]

    primary_chunk_ids = [h["chunk_id"] for h in primary_hits if h.get("chunk_id")]

    conn = psycopg.connect(DB_DSN)
    try:
        bodies = _fetch_chunk_bodies(conn, primary_chunk_ids)
    finally:
        conn.close()

    # --- Phase 4: Format code context ---
    code_context = []
    for hit in primary_hits:
        body = bodies.get(hit.get("chunk_id"))
        formatted = _format_code_primary(hit, body)
        if hit.get("rerank_score") is not None:
            formatted["rerank_score"] = round(hit["rerank_score"], 4)
        code_context.append(formatted)
    for hit in secondary_hits:
        formatted = _format_code_secondary(hit)
        if hit.get("rerank_score") is not None:
            formatted["rerank_score"] = round(hit["rerank_score"], 4)
        code_context.append(formatted)

    # --- Phase 5: Format memory context ---
    memory_context = []
    for h in memory_hits:
        formatted = _format_memory_item(h)
        if h.get("rerank_score") is not None:
            formatted["rerank_score"] = round(h["rerank_score"], 4)
        memory_context.append(formatted)

    # --- Phase 6: Graph traversal for dependency context ---
    dependency_map = {}
    if include_graph and primary_hits:
        dependency_map = _build_dependency_map(primary_hits)

    t_total = time.monotonic()

    # --- Assemble pack ---
    meta: dict[str, Any] = {
        "code_hits": len(code_hits),
        "memory_hits": len(memory_hits),
        "primary_with_body": len([c for c in code_context if c.get("body")]),
        "graph_traversals": len(dependency_map),
        "rerank_applied": rerank_applied,
        "feedback_applied": feedback_applied,
        "vector_search_ms": round((t_vector - t0) * 1000, 1),
        "total_ms": round((t_total - t0) * 1000, 1),
    }
    if rerank_applied:
        meta["rerank_model"] = RERANK_MODEL
        meta["rerank_ms"] = rerank_ms

    pack = {
        "query": query,
        "code_context": code_context,
        "memory_context": memory_context,
        "dependency_map": dependency_map,
        "meta": meta,
    }

    return pack


def _build_dependency_map(primary_hits: list[dict]) -> dict[str, Any]:
    """
    Build a compact dependency map for the primary code hits.
    Includes callers, callees, imports, and dependents for each hit.
    """
    dep_map = {}

    seen_modules = set()

    for hit in primary_hits:
        qname = hit.get("qualified_name")
        if not qname:
            continue

        chunk_type = hit.get("chunk_type")
        entry: dict[str, Any] = {}

        if chunk_type in ("function", "method"):
            # Get callers (who calls this?)
            callers = _get_callers(qname)
            if callers:
                entry["called_by"] = [_format_dependency_entry(c) for c in callers]

            # Get callees (what does this call?)
            callees = _get_callees(qname)
            if callees:
                entry["calls"] = [_format_dependency_entry(c) for c in callees]

        if chunk_type == "class":
            # Get methods
            methods = _get_class_methods(qname)
            if methods:
                entry["methods"] = [_format_dependency_entry(m) for m in methods]

        # Get file-level imports and dependents (once per module)
        module_name = hit.get("module_name")
        file_path = hit.get("file_path")
        if module_name and module_name not in seen_modules:
            seen_modules.add(module_name)

            if file_path:
                imports = _get_file_imports(file_path)
                if imports:
                    local_imports = [i["module"] for i in imports if i.get("is_local")]
                    external_imports = [i["module"] for i in imports if not i.get("is_local")]
                    if local_imports:
                        entry["imports_local"] = local_imports
                    if external_imports:
                        entry["imports_external"] = external_imports

            dependents = _get_dependents(module_name)
            if dependents:
                entry["depended_on_by"] = [_short_path(p) for p in dependents]
                entry["dependent_count"] = len(dependents)

        if entry:
            dep_map[qname] = entry

    return dep_map


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Build a unified context pack from code + memory + graph."
    )
    parser.add_argument("query", help="Natural language or code query")
    parser.add_argument("--code-limit", type=int, default=8)
    parser.add_argument("--memory-limit", type=int, default=5)
    parser.add_argument("--primary-count", type=int, default=3)
    parser.add_argument("--no-graph", action="store_true")
    parser.add_argument("--no-rerank", action="store_true",
                        help="Skip cross-encoder reranking")
    parser.add_argument("--compact", action="store_true",
                        help="Omit full bodies, show structure only")
    args = parser.parse_args()

    pack = build_context_pack(
        args.query,
        code_limit=args.code_limit,
        memory_limit=args.memory_limit,
        primary_count=args.primary_count,
        include_graph=not args.no_graph,
        rerank=not args.no_rerank,
    )

    if args.compact:
        # Strip bodies for compact view
        for item in pack.get("code_context", []):
            if "body" in item:
                lines = item["body"].count("\n") + 1
                item["body"] = f"<{lines} lines>"

    print(json.dumps(pack, indent=2, default=str))


if __name__ == "__main__":
    main()
