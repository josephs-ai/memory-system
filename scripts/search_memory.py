from typing import Any
import os
import httpx
import argparse
import json
import logging
import math
import sys
from pathlib import Path

from sentence_transformers import SentenceTransformer, CrossEncoder

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from memory_db import hybrid_search_memory_items, close_pool
from vector_store_qdrant import search_memory_vectors
from extract_memory_fields import extract_fields
from graph_store_neo4j import get_neo4j_driver
from feedback_score_engine import feedback_boost_batch, classify_query_type
from temporal_scoring import compute_temporal_boost

LOGGER = logging.getLogger("openclaw.search_memory")

WORKSPACE = Path.home() / ".openclaw" / "workspace"
MEMORY_DIR = WORKSPACE / "memory"

SEARCH_SERVICE_URL = os.environ.get("OPENCLAW_SEARCH_SERVICE_URL", "http://127.0.0.1:8791")
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
RERANK_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-float(x)))

def search_via_service(query: str, top_k: int = 8) -> list[dict[str, Any]] | None:
    try:
        r = httpx.get(
            f"{SEARCH_SERVICE_URL}/search",
            params={"query": query, "top_k": top_k},
            timeout=30,
        )
        r.raise_for_status()
        data = r.json()
        if not data.get("ok"):
            return None
        return data.get("results", [])
    except Exception:
        return None

def rerank_rows(query: str, rows: list[dict], top_n: int = 12) -> list[dict]:
    if not rows:
        return rows

    rows.sort(key=lambda x: (-x["score"], -x.get("mtime", 0), x["source_type"], x["path"]))

    head = rows[:top_n]
    tail = rows[top_n:]

    model = CrossEncoder(RERANK_MODEL_NAME)
    pairs = [[query, row["text"]] for row in head]
    raw_scores = model.predict(pairs)

    for row, raw in zip(head, raw_scores):
        row["rerank_raw"] = float(raw)
        row["rerank_score"] = sigmoid(float(raw))
        row["score"] = (row["score"] * 0.35) + (row["rerank_score"] * 1.25)

    head.sort(key=lambda x: (-x["score"], -x.get("mtime", 0), x["source_type"], x["path"]))
    return head + tail

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", required=True)
    parser.add_argument("--project-id")
    parser.add_argument("--top-k", type=int, default=8)
    args = parser.parse_args()

    service_rows = search_via_service(args.query, getattr(args, "top_k", 8))
    if service_rows is not None:
        for row in service_rows:
            print(json.dumps(row, ensure_ascii=False))
        return

    model = SentenceTransformer(MODEL_NAME)
    query_embedding = model.encode(
        args.query,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    scored = []
    seen = set()

    query_fields = extract_fields(args.query)
    graph_hits = search_graph_memory(
        query_fields.get("entity"),
        query_fields.get("property"),
    )

    canonical_items = hybrid_search_memory_items(
        query_embedding,
        query_text=args.query,
        status="active",
        allowed_sensitivities=["public", "internal"],
        limit=50,
    )

    for item in canonical_items:
        text = item.get("text", "")
        seen.add(text)
        base_score = float(item.get("final_score", 0) or 0) + 0.10
        is_summary = item.get("memory_type") == "summary"

        # Temporal boost (P3: time-aware scoring)
        temporal_boost = 0.0
        try:
            temporal_boost = compute_temporal_boost(
                item.get("first_seen") or item.get("last_confirmed"),
                args.query,
            )
        except Exception:
            pass

        row = {
            "score": base_score - (0.05 if is_summary else 0) + temporal_boost,
            "source_type": "canonical_summary" if is_summary else "canonical",
            "path": "db:memory_items",
            "text": text,
            "mtime": float("inf"),
            "item_id": item.get("id"),
        }

        if temporal_boost != 0.0:
            row["temporal_boost"] = temporal_boost

        # Summary drill-down metadata
        if is_summary:
            constituent_ids = item.get("constituent_item_ids") or []
            row["is_summary"] = True
            row["resolution_level"] = item.get("resolution_level")
            row["constituent_count"] = len(constituent_ids) if isinstance(constituent_ids, list) else 0

        scored.append(row)

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
                "item_id": payload.get("id"),
            }
        )

    # --- Feedback boost (P0: closed-loop retrieval) ---
    feedback_item_ids = [r["item_id"] for r in scored if r.get("item_id")]
    if feedback_item_ids:
        try:
            query_type = classify_query_type(args.query)
            boosts = feedback_boost_batch(feedback_item_ids, query_type=query_type)
            for row in scored:
                iid = row.get("item_id")
                if iid and iid in boosts:
                    row["feedback_boost"] = boosts[iid]
                    row["score"] += boosts[iid] * 0.15  # Conservative weight
        except Exception:
            LOGGER.debug("Feedback boost failed, continuing without", exc_info=True)

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

    for src in collect_project_sources(args.project_id):
        score = simple_project_score(args.query, src["text"], src["source_type"])
        if score > 0:
            row = dict(src)
            row["score"] = float(score)
            scored.append(row)

    scored = rerank_rows(args.query, scored, top_n=min(12, len(scored)))

    if not scored:
        print("NONE")
        close_pool()
        return

    for row in scored[: args.top_k]:
        print(json.dumps({
            "score": row["score"],
            "source_type": row["source_type"],
            "path": row["path"],
            "text": row["text"],
        }, ensure_ascii=False))

    close_pool()


if __name__ == "__main__":
    main()
