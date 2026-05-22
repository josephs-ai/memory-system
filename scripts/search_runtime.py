"""
Search and retrieve from runtime.

Key functions: get_embed_model, get_rerank_model, collect_project_sources, simple_project_score
"""
from __future__ import annotations

import math
from pathlib import Path

from sentence_transformers import CrossEncoder, SentenceTransformer

from extract_memory_fields import extract_fields
from graph_store_neo4j import get_neo4j_driver
from memory_db import hybrid_search_memory_items
from vector_store_qdrant import search_memory_vectors

WORKSPACE = Path.home() / ".openclaw" / "workspace"
MEMORY_DIR = WORKSPACE / "memory"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
RERANK_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

_EMBED_MODEL: SentenceTransformer | None = None
_RERANK_MODEL: CrossEncoder | None = None


def get_embed_model() -> SentenceTransformer:
    global _EMBED_MODEL
    if _EMBED_MODEL is None:
        _EMBED_MODEL = SentenceTransformer(MODEL_NAME)
    return _EMBED_MODEL


def get_rerank_model() -> CrossEncoder:
    global _RERANK_MODEL
    if _RERANK_MODEL is None:
        _RERANK_MODEL = CrossEncoder(RERANK_MODEL_NAME)
    return _RERANK_MODEL


def collect_project_sources(project_id: str | None):
    if not project_id:
        return

    project_dir = MEMORY_DIR / "projects" / project_id
    current_file = project_dir / "current.md"
    milestone_files = sorted(project_dir.glob("milestone-*.md"))
    daily_dir = project_dir / "daily"
    daily_files = sorted(daily_dir.glob("*.md")) if daily_dir.exists() else []

    if current_file.exists():
        mtime = current_file.stat().st_mtime
        for line in current_file.read_text(encoding="utf-8", errors="replace").splitlines():
            s = line.strip()
            if s.startswith("- "):
                yield {
                    "source_type": "project_current",
                    "path": str(current_file),
                    "text": s,
                    "mtime": mtime,
                }

    for mf in milestone_files:
        mtime = mf.stat().st_mtime
        for line in mf.read_text(encoding="utf-8", errors="replace").splitlines():
            s = line.strip()
            if s.startswith("- ") or s.startswith("## ") or s.startswith("### "):
                yield {
                    "source_type": "project_milestone",
                    "path": str(mf),
                    "text": s,
                    "mtime": mtime,
                }

    for df in daily_files[-7:]:
        mtime = df.stat().st_mtime
        for line in df.read_text(encoding="utf-8", errors="replace").splitlines():
            s = line.strip()
            if s.startswith("- "):
                yield {
                    "source_type": "project_daily",
                    "path": str(df),
                    "text": s,
                    "mtime": mtime,
                }


def simple_project_score(query: str, text: str, source_type: str) -> float:
    q_terms = [x.lower() for x in query.split() if x.strip()]
    t = (text or "").lower()
    raw_hits = sum(1 for term in q_terms if term in t)

    base = raw_hits * 0.30

    if source_type == "project_current":
        base += 0.12
    elif source_type == "project_milestone":
        base += 0.07
    elif source_type == "project_daily":
        base += 0.03

    return base


def search_graph_memory(entity: str | None, prop: str | None) -> list[dict]:
    if not entity and not prop:
        return []

    driver = get_neo4j_driver()
    out = []

    with driver.session() as session:
        if entity and prop:
            rows = session.run(
                """
                MATCH (e:Entity {name: $entity})-[:HAS_PROPERTY]->(p:Property {key: $property})-[:HAS_VALUE]->(v:Value)
                RETURN e.name AS entity, p.key AS property, v.key AS value
                ORDER BY value
                LIMIT 10
                """,
                entity=entity,
                property=prop,
            )
        elif entity:
            rows = session.run(
                """
                MATCH (e:Entity {name: $entity})-[:HAS_PROPERTY]->(p:Property)-[:HAS_VALUE]->(v:Value)
                RETURN e.name AS entity, p.key AS property, v.key AS value
                ORDER BY property, value
                LIMIT 10
                """,
                entity=entity,
            )
        else:
            rows = session.run(
                """
                MATCH (e:Entity)-[:HAS_PROPERTY]->(p:Property {key: $property})-[:HAS_VALUE]->(v:Value)
                RETURN e.name AS entity, p.key AS property, v.key AS value
                ORDER BY entity, value
                LIMIT 10
                """,
                property=prop,
            )

        for row in rows:
            out.append(
                {
                    "entity": row["entity"],
                    "property": row["property"],
                    "value": row["value"],
                    "text": (
                        f"{str(row['entity']).replace('_', ' ')} "
                        f"{str(row['property']).replace('_', ' ')} is "
                        f"{str(row['value']).replace('_', ' ')}."
                    ),
                }
            )

    driver.close()
    return out


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-float(x)))


def encode_query_embedding(model: SentenceTransformer, query: str):
    try:
        return model.encode(
            query,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
    except ValueError as e:
        if "Prompt name 'True' not found" not in str(e):
            raise
        return model.encode(
            sentences=query,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )


def rerank_rows(query: str, rows: list[dict], top_n: int = 12) -> list[dict]:
    if not rows:
        return rows

    rows.sort(key=lambda x: (-x["score"], -x.get("mtime", 0), x["source_type"], x["path"]))

    head = rows[:top_n]
    tail = rows[top_n:]

    model = get_rerank_model()
    pairs = [[query, row["text"]] for row in head]
    raw_scores = model.predict(pairs)

    for row, raw in zip(head, raw_scores):
        row["rerank_raw"] = float(raw)
        row["rerank_score"] = sigmoid(float(raw))
        row["score"] = (row["score"] * 0.35) + (row["rerank_score"] * 1.25)

    head.sort(key=lambda x: (-x["score"], -x.get("mtime", 0), x["source_type"], x["path"]))
    return head + tail


def run_search(query: str, top_k: int = 8, project_id: str | None = None) -> list[dict]:
    model = get_embed_model()
    query_embedding = encode_query_embedding(model, query)

    scored = []
    seen = set()

    query_fields = extract_fields(query)
    graph_hits = search_graph_memory(
        query_fields.get("entity"),
        query_fields.get("property"),
    )

    canonical_items = hybrid_search_memory_items(
        query_embedding,
        query_text=query,
        status="active",
        allowed_sensitivities=["public", "internal"],
        limit=50,
    )

    for item in canonical_items:
        text = item.get("text", "")
        seen.add(text)
        scored.append(
            {
                "score": float(item.get("final_score", 0) or 0) + 0.10,
                "source_type": "canonical",
                "path": "db:memory_items",
                "text": text,
                "mtime": float("inf"),
            }
        )

    qdrant_hits = search_memory_vectors(query_embedding, limit=20)
    for hit in qdrant_hits:
        payload = hit.payload or {}
        text = payload.get("text") or ""
        if not text or text in seen:
            continue

        seen.add(text)
        scored.append(
            {
                "score": float(hit.score or 0.0) + 0.20,
                "source_type": "canonical_qdrant",
                "path": "qdrant:memory_items",
                "text": text,
                "mtime": float("inf"),
            }
        )

    query_entity = query_fields.get("entity")
    query_property = query_fields.get("property")

    for hit in graph_hits:
        text = hit["text"]
        if not text or text in seen:
            continue

        score = 0.55
        if hit.get("entity") == query_entity:
            score += 0.15
        if hit.get("property") == query_property:
            score += 0.20

        seen.add(text)
        scored.append(
            {
                "score": score,
                "source_type": "canonical_graph",
                "path": "neo4j:memory_graph",
                "text": text,
                "mtime": float("inf"),
            }
        )

    for src in collect_project_sources(project_id):
        score = simple_project_score(query, src["text"], src["source_type"])
        if score > 0:
            row = dict(src)
            row["score"] = float(score)
            scored.append(row)

    scored = rerank_rows(query, scored, top_n=min(12, len(scored)))

    out = []
    for row in scored[:top_k]:
        out.append(
            {
                "score": row["score"],
                "source_type": row["source_type"],
                "path": row["path"],
                "text": row["text"],
            }
        )
    return out
