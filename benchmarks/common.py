"""
common.py — Shared utilities for memory system benchmark evaluation.

Provides:
- Timing / latency measurement (p50, p95)
- Text scoring: exact-match F1, token-level F1
- IR metrics: recall@k, MRR
- LLM-as-judge scoring (Claude / OpenAI)
- Memory item ingestion helper (with isolated project namespace)
- Cleanup helper (delete benchmark items by source_agent prefix)
- Results formatting (JSON + markdown table)
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent.parent / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from memory_db import (
    upsert_memory_items,
    upsert_memory_embedding,
    hybrid_search_memory_items,
    POOL,
)
from vector_store_qdrant import upsert_memory_vector, ensure_qdrant_collection

LOGGER = logging.getLogger("openclaw.benchmarks.common")

BENCHMARKS_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BENCHMARKS_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
_embed_model = None

# Remote GPU inference server (optional).
# Set GPU_INFERENCE_URL to offload embedding/reranking to a GPU host.
# Example: GPU_INFERENCE_URL=http://192.168.64.1:9999
# When unset or unreachable, falls back to local CPU inference.
GPU_INFERENCE_URL = os.environ.get("GPU_INFERENCE_URL", "")
_use_remote_gpu = None


def _remote_gpu_available() -> bool:
    global _use_remote_gpu
    if _use_remote_gpu is None:
        if not GPU_INFERENCE_URL:
            _use_remote_gpu = False
        else:
            try:
                import requests
                r = requests.get(f"{GPU_INFERENCE_URL}/health", timeout=2)
                _use_remote_gpu = r.status_code == 200
                if _use_remote_gpu:
                    LOGGER.info("Remote GPU inference active at %s (device=%s)",
                                GPU_INFERENCE_URL, r.json().get("device"))
            except Exception:
                _use_remote_gpu = False
                LOGGER.info("Remote GPU inference not available, using local CPU")
    return _use_remote_gpu


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------


def get_embed_model():
    global _embed_model
    if _embed_model is None:
        from sentence_transformers import SentenceTransformer
        _embed_model = SentenceTransformer(MODEL_NAME)
        LOGGER.info("Loaded embedding model: %s", MODEL_NAME)
    return _embed_model


def embed_texts(texts: list[str]) -> np.ndarray:
    if _remote_gpu_available():
        import requests
        r = requests.post(f"{GPU_INFERENCE_URL}/embed",
                          json={"texts": texts}, timeout=120)
        r.raise_for_status()
        return np.array(r.json()["embeddings"], dtype=np.float32)
    model = get_embed_model()
    return model.encode(texts, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)


def embed_text(text: str) -> np.ndarray:
    return embed_texts([text])[0]


# ---------------------------------------------------------------------------
# Memory item ingestion
# ---------------------------------------------------------------------------


def make_memory_item(
    text: str,
    *,
    source_agent: str,
    source_session: str | None = None,
    entity: str | None = None,
    property: str | None = None,
    value: str | None = None,
    memory_type: str = "fact",
    scope: str = "benchmark",
    tags: list[str] | None = None,
    importance: float = 0.5,
    item_id: str | None = None,
) -> dict:
    """Create a memory item dict suitable for upsert_memory_items()."""
    return {
        "id": item_id or f"bench_{uuid.uuid4().hex}",
        "text": text,
        "memory_type": memory_type,
        "scope": scope,
        "project_id": None,
        "subproject_id": None,
        "workflow_id": None,
        "pipeline_id": None,
        "context_scope_id": None,
        "context_scope_type": None,
        "context_scope_payload": {},
        "inheritance_policy": None,
        "scope_confidence": 1.0,
        "entity": entity,
        "property": property,
        "value": value,
        "status": "active",
        "confidence": 0.9,
        "importance": importance,
        "freshness_class": "stable",
        "source_agent": source_agent,
        "source_session": source_session,
        "source_chunk": None,
        "first_seen": datetime.now(timezone.utc).isoformat(),
        "last_confirmed": datetime.now(timezone.utc).isoformat(),
        "supersedes": None,
        "tags": tags or [],
        "notes": None,
        "candidate_id": None,
        "candidate_score": None,
        "candidate_reasons": [],
        "suggested_route": None,
        "target_file": None,
        "target_section": None,
        "sensitivity": "public",
        "approved_at": datetime.now(timezone.utc).isoformat(),
        "approved_by": "benchmark",
        "approval_source": "benchmark",
        "rejected_at": None,
        "rejected_by": None,
        "rejection_reason": None,
        "rejection_source": None,
        "ranking_bonus": 0.0,
        "ranking_penalty": 0.0,
        "feedback_last_applied_at": None,
    }


def _reconcile_supersedes(new_items: list[dict]):
    """After ingesting new items, check if any existing active items with the
    same entity+property should be superseded. Uses the reconcile system's
    can_auto_supersede() logic."""
    try:
        from memory_db import POOL
        from reconcile_memory_items import can_auto_supersede
    except ImportError:
        return

    # Group new items by (entity, property)
    ep_map = {}
    for item in new_items:
        entity = (item.get("entity") or "").strip()
        prop = (item.get("property") or "").strip()
        if entity and prop:
            key = (entity.lower(), prop.lower())
            # Keep the newest item per key
            existing = ep_map.get(key)
            if existing is None:
                ep_map[key] = item
            else:
                c_ts = str(item.get("last_confirmed") or item.get("first_seen") or "")
                e_ts = str(existing.get("last_confirmed") or existing.get("first_seen") or "")
                if c_ts > e_ts:
                    ep_map[key] = item

    if not ep_map:
        return

    with POOL.connection() as conn:
        with conn.cursor() as cur:
            for (entity_lower, prop_lower), new_item in ep_map.items():
                # Find active items with same entity+property+source_agent that aren't this item.
                # Scoped to same source_agent to avoid cross-scenario interference.
                source_agent = new_item.get("source_agent") or ""
                cur.execute("""
                    SELECT id, entity, property, value, last_confirmed, first_seen,
                           source_agent, status
                    FROM memory_items
                    WHERE LOWER(entity) = %s AND LOWER(property) = %s
                      AND status = 'active'
                      AND id != %s
                      AND source_agent = %s
                """, [entity_lower, prop_lower, new_item["id"], source_agent])

                for row in cur.fetchall():
                    existing = dict(zip(
                        ["id", "entity", "property", "value", "last_confirmed",
                         "first_seen", "source_agent", "status"], row))

                    if can_auto_supersede(new_item, existing):
                        cur.execute("""
                            UPDATE memory_items
                            SET status = 'superseded'
                            WHERE id = %s AND status = 'active'
                        """, [existing["id"]])
                        LOGGER.debug("Superseded %s (entity=%s, prop=%s) by %s",
                                     existing["id"], entity_lower, prop_lower, new_item["id"])

            conn.commit()


def ingest_memory_items(items: list[dict], batch_size: int = 128) -> int:
    """
    Ingest items into memory_items + embeddings + Qdrant.
    Batches Qdrant upserts for speed.
    Returns number of items ingested.
    """
    from vector_store_qdrant import (
        get_qdrant_client,
        QDRANT_COLLECTION,
        qdrant_point_id,
    )
    from qdrant_client.http import models as qdrant_models

    ensure_qdrant_collection()
    client = get_qdrant_client()

    texts = [item["text"] or "" for item in items]
    total = len(items)

    for start in range(0, total, batch_size):
        batch_items = items[start : start + batch_size]
        batch_texts = texts[start : start + batch_size]

        embeddings = embed_texts(batch_texts)

        # 1. Upsert to PostgreSQL (bulk)
        upsert_memory_items(batch_items)

        # 1b. Reconcile: auto-supersede older items with same entity+property.
        _reconcile_supersedes(batch_items)

        # 2. Upsert embeddings to pg (bulk via executemany in upsert_memory_embedding)
        for item, emb in zip(batch_items, embeddings):
            vec = emb.tolist()
            upsert_memory_embedding(item["id"], MODEL_NAME, vec)

        # 3. Batch upsert to Qdrant
        points = []
        for item, emb in zip(batch_items, embeddings):
            points.append(qdrant_models.PointStruct(
                id=qdrant_point_id(item["id"]),
                vector=emb.tolist(),
                payload={
                    "id": item["id"],
                    "text": item.get("text"),
                    "memory_type": item.get("memory_type"),
                    "scope": item.get("scope"),
                    "entity": item.get("entity"),
                    "property": item.get("property"),
                    "value": item.get("value"),
                },
            ))
        client.upsert(collection_name=QDRANT_COLLECTION, points=points)

    return total


def cleanup_benchmark_items(source_agent_prefix: str) -> int:
    """
    Delete all memory items whose source_agent starts with the given prefix.
    Also deletes associated embeddings (CASCADE) and Qdrant vectors.
    Returns number of rows deleted.
    """
    with POOL.connection() as conn:
        with conn.cursor() as cur:
            # Fetch IDs first for Qdrant cleanup
            cur.execute(
                "SELECT id FROM memory_items WHERE source_agent LIKE %s",
                (source_agent_prefix + "%",),
            )
            ids = [row[0] for row in cur.fetchall()]

            if not ids:
                return 0

            cur.execute(
                "DELETE FROM memory_items WHERE source_agent LIKE %s",
                (source_agent_prefix + "%",),
            )
            deleted = cur.rowcount
        conn.commit()

    # Qdrant cleanup
    try:
        from vector_store_qdrant import get_qdrant_client, QDRANT_COLLECTION

        client = get_qdrant_client()
        # Convert memory IDs to Qdrant point UUIDs (same approach as upsert_memory_vector)
        from vector_store_qdrant import qdrant_point_id as _qdrant_id

        # Delete in chunks of 100
        for i in range(0, len(ids), 100):
            chunk = ids[i : i + 100]
            point_ids = [_qdrant_id(mid) for mid in chunk]
            try:
                client.delete(
                    collection_name=QDRANT_COLLECTION,
                    points_selector=point_ids,
                )
            except Exception as e:
                LOGGER.warning("Qdrant delete failed for chunk: %s", e)
    except Exception as e:
        LOGGER.warning("Qdrant cleanup skipped: %s", e)

    return deleted


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------


def retrieve(
    query: str,
    *,
    limit: int = 10,
    source_agent_prefix: str | None = None,
    entity_filter: str | None = None,
    session_filter: str | None = None,
    memory_type_filter: str | None = None,
    use_full_pipeline: bool = True,
) -> tuple[list[dict], float]:
    """
    Run search for query. Returns (results, latency_seconds).

    When use_full_pipeline=True (default), uses the full search pipeline
    including metadata filtering, Qdrant vector search, cross-encoder
    reranking, feedback boost, and temporal scoring.

    Metadata filters (entity_filter, session_filter, memory_type_filter)
    are pushed into SQL for DB queries and applied post-retrieval for
    Qdrant/graph sources — matching production search_memory_service.py.
    """
    t0 = time.perf_counter()
    qvec = embed_text(query)

    if use_full_pipeline:
        try:
            results = _retrieve_full_pipeline(
                query, qvec, limit=limit,
                source_agent_prefix=source_agent_prefix,
                entity_filter=entity_filter,
                session_filter=session_filter,
                memory_type_filter=memory_type_filter,
            )
            latency = time.perf_counter() - t0
            return results, latency
        except Exception as e:
            LOGGER.warning("Full pipeline failed, falling back to basic: %s", e)

    # Basic fallback
    results = hybrid_search_memory_items(
        qvec,
        query_text=query,
        status="active",
        allowed_sensitivities=["public", "internal"],
        limit=limit,
        source_agent_prefix=source_agent_prefix,
    )
    latency = time.perf_counter() - t0
    return results, latency


# Cache the cross-encoder model at module level to avoid reloading per query
_reranker_model = None


def _get_reranker():
    global _reranker_model
    if _reranker_model is None:
        if _remote_gpu_available():
            # Return a lightweight proxy that calls the remote GPU server
            class _RemoteReranker:
                def predict(self, pairs):
                    import requests
                    r = requests.post(f"{GPU_INFERENCE_URL}/rerank",
                                      json={"pairs": pairs}, timeout=120)
                    r.raise_for_status()
                    return r.json()["scores"]
            _reranker_model = _RemoteReranker()
            LOGGER.info("Using remote GPU reranker at %s", GPU_INFERENCE_URL)
        else:
            from sentence_transformers import CrossEncoder
            _reranker_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
            LOGGER.info("Loaded cross-encoder reranker: cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _reranker_model


def _retrieve_full_pipeline(
    query: str,
    qvec: np.ndarray,
    *,
    limit: int = 10,
    source_agent_prefix: str | None = None,
    entity_filter: str | None = None,
    session_filter: str | None = None,
    memory_type_filter: str | None = None,
) -> list[dict]:
    """
    Full retrieval pipeline matching production search_memory_service.py:
    1. PostgreSQL hybrid search with metadata filters pushed into SQL
    2. Qdrant vector search with post-retrieval metadata filtering
    3. Cross-encoder reranking
    4. Temporal scoring + feedback boost
    """
    import math

    def sigmoid(x):
        return 1.0 / (1.0 + math.exp(-float(x)))

    # Build MetadataFilter if any filters are specified
    metadata_filter = None
    has_filters = any(f is not None for f in [entity_filter, session_filter, memory_type_filter])
    if has_filters:
        try:
            from search_memory_service import MetadataFilter, _apply_metadata_filter
            metadata_filter = MetadataFilter(
                entity=entity_filter,
                session_id=session_filter,
                memory_type=memory_type_filter,
            )
        except ImportError:
            LOGGER.warning("search_memory_service not importable, skipping metadata filters")

    # Step 1: PostgreSQL search with metadata filters pushed into SQL
    if metadata_filter is not None:
        try:
            from search_memory_service import search_memory_items_filtered
            # Hybrid search with filters in SQL WHERE clause — no post-filter loss
            canonical_items = search_memory_items_filtered(
                query,
                filters=metadata_filter,
                status="active",
                limit=50,
                query_embedding=qvec.tolist(),
            )
        except Exception as e:
            LOGGER.warning("Filtered hybrid search failed: %s, falling back to post-filter", e)
            canonical_items = hybrid_search_memory_items(
                qvec, query_text=query, status="active",
                allowed_sensitivities=["public", "internal"],
                limit=50, source_agent_prefix=source_agent_prefix,
            )
            try:
                from search_memory_service import _apply_metadata_filter
                canonical_items = _apply_metadata_filter(canonical_items, metadata_filter)
            except ImportError:
                pass
    else:
        # Always try the filtered search path (even without metadata filters)
        # because it includes temporal recency boost in the SQL scoring.
        try:
            from search_memory_service import search_memory_items_filtered
            canonical_items = search_memory_items_filtered(
                query,
                filters=None,
                status="active",
                limit=50,
                query_embedding=qvec.tolist(),
            )
            # Post-filter by source_agent_prefix if set
            if source_agent_prefix:
                canonical_items = [
                    r for r in canonical_items
                    if (r.get("source_agent") or "").startswith(source_agent_prefix)
                ]
        except Exception:
            canonical_items = hybrid_search_memory_items(
                qvec, query_text=query, status="active",
                allowed_sensitivities=["public", "internal"],
                limit=50, source_agent_prefix=source_agent_prefix,
            )

    scored = []
    seen_texts = set()

    for item in canonical_items:
        text = item.get("text", "")
        if text in seen_texts:
            continue
        seen_texts.add(text)
        base_score = float(item.get("final_score", 0) or item.get("fts_rank", 0) or 0) + 0.10

        # Temporal boost
        temporal_boost = 0.0
        try:
            from temporal_scoring import compute_temporal_boost
            temporal_boost = compute_temporal_boost(
                item.get("first_seen") or item.get("last_confirmed"), query,
            )
        except Exception:
            pass

        row = {
            "score": base_score + temporal_boost,
            "text": text,
            "source_type": "canonical",
            "path": "db:memory_items",
            **{k: v for k, v in item.items() if k not in ("final_score", "fts_rank")},
        }
        scored.append(row)

    # Step 2: Qdrant vector search (supplementary) + post-retrieval metadata filter
    try:
        from vector_store_qdrant import search_memory_vectors
        qdrant_hits = search_memory_vectors(qvec, limit=20)
        qdrant_rows = []
        for hit in qdrant_hits:
            payload = hit.payload or {}
            text = payload.get("text") or ""
            if not text or text in seen_texts:
                continue
            seen_texts.add(text)
            qdrant_rows.append({
                "score": float(hit.score or 0.0) + 0.20,
                "text": text,
                "source_type": "canonical_qdrant",
                "path": "qdrant:memory_items",
                **{k: v for k, v in payload.items() if k != "text"},
            })

        # Apply metadata filter post-retrieval for Qdrant results
        if metadata_filter is not None and qdrant_rows:
            try:
                from search_memory_service import _apply_metadata_filter
                qdrant_rows = _apply_metadata_filter(qdrant_rows, metadata_filter)
            except ImportError:
                pass

        # Filter out superseded/archived items from Qdrant (phantom vectors)
        if qdrant_rows:
            try:
                qdrant_ids = [r.get("id") or r.get("item_id") for r in qdrant_rows if r.get("id") or r.get("item_id")]
                if qdrant_ids:
                    with POOL.connection() as conn:
                        with conn.cursor() as cur:
                            cur.execute(
                                "SELECT id FROM memory_items WHERE id = ANY(%s) AND status != 'active'",
                                [qdrant_ids],
                            )
                            non_active_ids = {row[0] for row in cur.fetchall()}
                    if non_active_ids:
                        qdrant_rows = [r for r in qdrant_rows
                                       if (r.get("id") or r.get("item_id")) not in non_active_ids]
            except Exception as e:
                LOGGER.warning("Qdrant phantom filter failed: %s", e)

        scored.extend(qdrant_rows)
    except Exception as e:
        LOGGER.debug("Qdrant supplementary search failed: %s", e)

    # Step 3: Feedback boost
    try:
        from feedback_score_engine import feedback_boost_batch, classify_query_type
        feedback_item_ids = [r.get("id") or r.get("item_id") for r in scored
                            if r.get("id") or r.get("item_id")]
        if feedback_item_ids:
            query_type = classify_query_type(query)
            boosts = feedback_boost_batch(feedback_item_ids, query_type=query_type)
            for row in scored:
                iid = row.get("id") or row.get("item_id")
                if iid and iid in boosts:
                    row["score"] += boosts[iid] * 0.15
    except Exception:
        pass

    # Step 4: Cross-encoder reranking (cached model)
    if scored:
        top_n = min(12, len(scored))
        scored.sort(key=lambda x: -x["score"])
        head = scored[:top_n]
        tail = scored[top_n:]

        try:
            reranker = _get_reranker()
            pairs = [[query, row["text"]] for row in head]
            raw_scores = reranker.predict(pairs)

            for row, raw in zip(head, raw_scores):
                row["rerank_score"] = sigmoid(float(raw))
                row["score"] = (row["score"] * 0.35) + (row["rerank_score"] * 1.25)

            head.sort(key=lambda x: -x["score"])
            scored = head + tail
        except Exception as e:
            LOGGER.warning("Cross-encoder reranking failed: %s", e)

    results = scored[:limit]

    # Post-retrieval: drop any items whose DB status is not 'active'.
    # Qdrant results don't carry status, so re-verify against the DB for items
    # that lack it (covers superseded/deleted items leaking from vector index).
    ids_to_check = [r.get("id") for r in results if not r.get("status") and r.get("id")]
    if ids_to_check:
        try:
            status_map: dict[str, str] = {}
            with __import__("memory_db").POOL.connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT id, status FROM memory_items WHERE id = ANY(%s)",
                        (ids_to_check,),
                    )
                    for row in cur.fetchall():
                        status_map[row[0]] = row[1]
            results = [
                r for r in results
                if r.get("status") == "active"
                or (not r.get("status") and status_map.get(r.get("id"), "active") == "active")
            ]
        except Exception:
            pass

    return results


# ---------------------------------------------------------------------------
# Text scoring
# ---------------------------------------------------------------------------


def normalize_answer(s: str) -> str:
    """Lowercase, remove punctuation/articles, collapse whitespace."""
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = re.sub(r"[^\w\s]", " ", s)
    s = " ".join(s.split())
    return s


def token_f1(pred: str, gold: str) -> float:
    """Token-level F1 between prediction and gold answer."""
    pred_tokens = normalize_answer(pred).split()
    gold_tokens = normalize_answer(gold).split()
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)
    pred_counter: dict[str, int] = {}
    for t in pred_tokens:
        pred_counter[t] = pred_counter.get(t, 0) + 1
    gold_counter: dict[str, int] = {}
    for t in gold_tokens:
        gold_counter[t] = gold_counter.get(t, 0) + 1
    common = sum(min(pred_counter.get(t, 0), cnt) for t, cnt in gold_counter.items())
    if common == 0:
        return 0.0
    precision = common / len(pred_tokens)
    recall = common / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def exact_match(pred: str, gold: str) -> float:
    return float(normalize_answer(pred) == normalize_answer(gold))


# ---------------------------------------------------------------------------
# IR metrics
# ---------------------------------------------------------------------------


def recall_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    if not relevant_ids:
        return 0.0
    hits = sum(1 for rid in retrieved_ids[:k] if rid in relevant_ids)
    return hits / len(relevant_ids)


def precision_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    if k == 0:
        return 0.0
    hits = sum(1 for rid in retrieved_ids[:k] if rid in relevant_ids)
    return hits / k


def mrr(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    for rank, rid in enumerate(retrieved_ids, 1):
        if rid in relevant_ids:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved_ids: list[str], relevance_map: dict[str, float], k: int) -> float:
    dcg = sum(
        relevance_map.get(rid, 0.0) / math.log2(rank + 1)
        for rank, rid in enumerate(retrieved_ids[:k], 1)
    )
    ideal = sorted(relevance_map.values(), reverse=True)[:k]
    idcg = sum(v / math.log2(i + 2) for i, v in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------


def compute_percentiles(latencies: list[float]) -> dict[str, float]:
    if not latencies:
        return {"p50": 0.0, "p95": 0.0, "mean": 0.0}
    arr = sorted(latencies)
    n = len(arr)

    def pct(p):
        idx = max(0, min(n - 1, int(p / 100 * n)))
        return arr[idx]

    return {"p50": pct(50), "p95": pct(95), "mean": sum(arr) / n}


# ---------------------------------------------------------------------------
# Token counting (approximate)
# ---------------------------------------------------------------------------


def count_tokens_approx(text: str) -> int:
    """Rough approximation: 4 chars per token."""
    return max(1, len(text) // 4)


def count_tokens_list(texts: list[str]) -> int:
    return sum(count_tokens_approx(t) for t in texts)


# ---------------------------------------------------------------------------
# LLM-as-judge
# ---------------------------------------------------------------------------


def llm_judge_answer(
    question: str,
    gold_answer: str,
    predicted_answer: str,
    *,
    model: str = "claude-opus-4-6",
) -> float:
    """
    Use an LLM to judge whether predicted_answer is correct given gold_answer.
    Returns score in [0, 1].
    """
    try:
        import anthropic

        client = anthropic.Anthropic()
        prompt = f"""You are evaluating a question-answering system.

Question: {question}
Ground Truth Answer: {gold_answer}
Predicted Answer: {predicted_answer}

Is the predicted answer correct? Consider partial credit for partially correct answers.
Respond with ONLY a JSON object: {{"score": <float 0.0-1.0>, "reason": "<brief reason>"}}"""

        msg = client.messages.create(
            model=model,
            max_tokens=150,
            messages=[{"role": "user", "content": prompt}],
        )
        text = msg.content[0].text.strip()
        data = json.loads(text)
        return float(data.get("score", 0.0))
    except Exception as e:
        LOGGER.warning("LLM judge failed: %s. Falling back to F1.", e)
        return token_f1(predicted_answer, gold_answer)


# ---------------------------------------------------------------------------
# Results output
# ---------------------------------------------------------------------------


def save_results(benchmark_name: str, results: dict) -> Path:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"{benchmark_name}_{ts}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    LOGGER.info("Saved results to %s", out_path)
    return out_path


def format_results_table(all_results: list[dict]) -> str:
    """Format a list of benchmark result dicts into a markdown table."""
    header = "| Benchmark | Score | Tokens | Latency p50 | Latency p95 |\n"
    header += "|-----------|-------|--------|-------------|-------------|\n"
    rows = []
    for r in all_results:
        name = r.get("benchmark", "?")
        score = r.get("score", 0.0)
        tokens = r.get("tokens_avg", 0)
        p50 = r.get("latency_p50", 0.0)
        p95 = r.get("latency_p95", 0.0)
        rows.append(
            f"| {name} | {score:.3f} | {tokens:.0f} | {p50*1000:.1f}ms | {p95*1000:.1f}ms |"
        )
    return header + "\n".join(rows)


def print_results_table(all_results: list[dict]) -> None:
    print("\n" + "=" * 60)
    print("MEMORY SYSTEM BENCHMARK RESULTS")
    print("=" * 60)
    print(format_results_table(all_results))
    print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# LLM reader, intent routing, typed-lane retrieval, and trace capture
#
# These were imported by five runners (fever, hotpotqa, musique, longmemeval,
# temporalqa) but defined nowhere in the repo, so every one of them died at
# import. The signatures below are derived from those call sites, not invented:
# the reader dict shape is exactly what fever/run.py already destructures
# (provider/client/model across openai, anthropic, google), and
# retrieve_typed_lanes returns (results, latency) to match retrieve().
# ---------------------------------------------------------------------------

LLM_READER_MODEL = os.environ.get("LLM_READER_MODEL", "claude-opus-4-6")

_READER_CACHE: dict | None = None
_READER_TRIED = False


def _get_llm_reader() -> dict | None:
    """
    Return {"provider", "client", "model"} for the first usable LLM provider,
    or None when no credentials are configured.

    Returning None rather than raising is deliberate: every call site treats a
    missing reader as "fall back to the non-LLM path" (fever drops to NLI,
    hotpotqa/musique to span overlap). A benchmark run without API keys should
    report weaker scores, not crash.

    Cached because the readers are constructed per question, and building a
    client per call turns a rate-limit into a connection storm.
    """
    global _READER_CACHE, _READER_TRIED
    if _READER_TRIED:
        return _READER_CACHE
    _READER_TRIED = True

    model = LLM_READER_MODEL
    if os.environ.get("ANTHROPIC_API_KEY"):
        try:
            import anthropic

            _READER_CACHE = {"provider": "anthropic", "client": anthropic.Anthropic(), "model": model}
            return _READER_CACHE
        except Exception as e:  # pragma: no cover - depends on local env
            LOGGER.warning("anthropic reader unavailable: %s", e)
    if os.environ.get("OPENAI_API_KEY"):
        try:
            import openai

            _READER_CACHE = {
                "provider": "openai",
                "client": openai.OpenAI(),
                "model": os.environ.get("LLM_READER_MODEL_OPENAI", "gpt-4o-mini"),
            }
            return _READER_CACHE
        except Exception as e:  # pragma: no cover
            LOGGER.warning("openai reader unavailable: %s", e)
    if os.environ.get("GOOGLE_API_KEY"):
        try:
            from google import genai

            _READER_CACHE = {
                "provider": "google",
                "client": genai.Client(),
                "model": os.environ.get("LLM_READER_MODEL_GOOGLE", "gemini-2.5-flash"),
            }
            return _READER_CACHE
        except Exception as e:  # pragma: no cover
            LOGGER.warning("google reader unavailable: %s", e)

    LOGGER.warning("no LLM reader configured; runners will use their non-LLM fallback")
    return None


def _reader_complete(reader: dict, prompt: str, *, max_tokens: int = 128) -> str | None:
    """One completion, normalised across the three provider SDKs."""
    try:
        if reader["provider"] == "anthropic":
            resp = reader["client"].messages.create(
                model=reader["model"], max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
            return resp.content[0].text.strip()
        if reader["provider"] == "openai":
            resp = reader["client"].chat.completions.create(
                model=reader["model"], max_tokens=max_tokens, temperature=0,
                messages=[{"role": "user", "content": prompt}],
            )
            return (resp.choices[0].message.content or "").strip()
        if reader["provider"] == "google":
            resp = reader["client"].models.generate_content(
                model=reader["model"], contents=prompt,
            )
            return (resp.text or "").strip()
    except Exception as e:
        LOGGER.warning("reader completion failed: %s", e)
    return None


def llm_read_answer(question: str, retrieved: list[dict], *, max_items: int = 10) -> str | None:
    """
    Answer `question` from `retrieved` alone. None when no reader is configured
    or the model declines, which callers treat as "use the extractive fallback".

    The prompt forbids outside knowledge on purpose: this measures the
    retrieval, so an answer the model knew independently would inflate the
    score without the memory system having contributed anything.
    """
    reader = _get_llm_reader()
    if reader is None or not retrieved:
        return None

    lines = []
    for i, r in enumerate(retrieved[:max_items], 1):
        text = ((r.get("text") or "") + " " + (r.get("value") or "")).strip()
        if text:
            lines.append(f"{i}. {text[:500]}")
    if not lines:
        return None

    prompt = (
        "Answer the question using ONLY the numbered context below. "
        "Do not use outside knowledge. Reply with the shortest exact answer "
        "(a name, date, number or phrase) and nothing else. "
        "If the context does not contain the answer, reply exactly: UNKNOWN\n\n"
        f"Context:\n" + "\n".join(lines) + f"\n\nQuestion: {question}\nAnswer:"
    )
    answer = _reader_complete(reader, prompt, max_tokens=64)
    if not answer or answer.strip().upper() == "UNKNOWN":
        return None
    return answer


# Intent labels are a closed set; longmemeval/run.py compares against
# "aggregation_count" directly, so the strings are part of the contract.
_AGG_PAT = re.compile(r"\b(how many|how much|count|total|number of|sum of)\b", re.I)
_TEMPORAL_PAT = re.compile(
    r"\b(when|what date|what time|before|after|earlier|later|yesterday|today|"
    r"last (week|month|year|night)|this (week|month|year)|ago|since|until|"
    r"first|last|most recent|latest|previously)\b",
    re.I,
)
_COMPARE_PAT = re.compile(r"\b(compare|difference|versus|vs\.?|more than|less than|between)\b", re.I)


def classify_query_intent(query: str) -> str:
    """
    Route a question to a retrieval strategy.

    Cheap and deterministic on purpose: an LLM call here would be a second
    source of latency and non-determinism inside the thing being measured.
    Order matters -- "how many times did X happen last week" is an aggregation
    that also mentions time, and the wider aggregation lane is the one that
    matters for recall.
    """
    q = query or ""
    if _AGG_PAT.search(q):
        return "aggregation_count"
    if _TEMPORAL_PAT.search(q):
        return "temporal"
    if _COMPARE_PAT.search(q):
        return "comparison"
    return "fact_lookup"


def retrieve_typed_lanes(
    query: str,
    *,
    limit: int = 20,
    source_agent_prefix: str | None = None,
    intent: str | None = None,
    expand_parents: bool = True,
) -> tuple[list[dict], float]:
    """
    Retrieve for `query`, widening or narrowing by intent. Returns
    (results, latency_seconds) to match retrieve().

    "Typed lanes" means the intent decides how much is pulled and how it is
    ordered, not that separate indexes are queried:

      aggregation_count  needs recall over precision -- a count is wrong if one
                         instance is missed -- so it pulls wider.
      temporal           orders by recorded time where it is known, because the
                         top semantic hit for "what did we do yesterday" is
                         routinely a month old. Items with no timestamp keep
                         their relevance order behind the dated ones rather
                         than being dropped: 35% of the store has no usable
                         time, and discarding a third of memory to sort the
                         rest is a worse answer.
      otherwise          plain relevance.
    """
    intent = intent or classify_query_intent(query)
    effective_limit = limit
    if intent == "aggregation_count":
        effective_limit = max(limit, 50)
    elif intent == "comparison":
        effective_limit = max(limit, 30)

    results, latency = retrieve(
        query,
        limit=effective_limit,
        source_agent_prefix=source_agent_prefix,
        use_full_pipeline=expand_parents,
    )

    if intent == "temporal" and results:
        def _when(item: dict):
            for key in ("last_confirmed", "first_seen", "created_at"):
                v = item.get(key)
                if v:
                    return v
            return None

        dated = [r for r in results if _when(r) is not None]
        undated = [r for r in results if _when(r) is None]
        dated.sort(key=lambda r: str(_when(r)), reverse=True)
        results = dated + undated

    return results[:limit], latency


class BenchmarkTraceWriter:
    """
    Per-question trace for a benchmark run, so a score can be explained rather
    than only reported.

    A benchmark that emits one aggregate number tells you it regressed but not
    which questions or why. Each row keeps the retrieved ids alongside the
    score, which is what makes a drop diagnosable after the fact.
    """

    def __init__(self, benchmark: str, *, run_label: str = "", trace_dir: Path | None = None):
        self.benchmark = benchmark
        self.run_label = run_label
        self.rows: list[dict] = []
        self.trace_dir = Path(trace_dir) if trace_dir else RESULTS_DIR / "traces"
        self.started_at = datetime.now(timezone.utc)

    def record(self, **fields: Any) -> dict:
        """Store one question's trace and return it, so callers can embed it."""
        retrieved = fields.pop("retrieved_items", None) or []
        row = dict(fields)
        row["retrieved_ids"] = [r.get("id") for r in retrieved][:20]
        row["retrieved_count"] = len(retrieved)
        row["recorded_at"] = datetime.now(timezone.utc).isoformat()
        self.rows.append(row)
        return row

    def summary(self) -> dict:
        """Aggregate the trace and write it out. Safe to call more than once."""
        scored = [r.get("f1") for r in self.rows if isinstance(r.get("f1"), (int, float))]
        latencies = [r.get("latency") for r in self.rows if isinstance(r.get("latency"), (int, float))]
        reader_used = sum(1 for r in self.rows if r.get("reader_used"))
        out: dict[str, Any] = {
            "benchmark": self.benchmark,
            "run_label": self.run_label,
            "questions": len(self.rows),
            "reader_used": reader_used,
            "f1_mean": (sum(scored) / len(scored)) if scored else 0.0,
            "latency_mean": (sum(latencies) / len(latencies)) if latencies else 0.0,
            "started_at": self.started_at.isoformat(),
        }
        try:
            self.trace_dir.mkdir(parents=True, exist_ok=True)
            stamp = self.started_at.strftime("%Y%m%d_%H%M%S")
            path = self.trace_dir / f"{self.benchmark}_{self.run_label or 'run'}_{stamp}.json"
            with open(path, "w") as f:
                json.dump({"summary": out, "rows": self.rows}, f, indent=2, default=str)
            out["trace_path"] = str(path)
        except Exception as e:  # a trace that cannot be written must not fail the run
            LOGGER.warning("could not write trace: %s", e)
        return out
