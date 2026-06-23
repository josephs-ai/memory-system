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
    upsert_memory_embeddings_batch,
    hybrid_search_memory_items,
    POOL,
)
from vector_store_qdrant import upsert_memory_vector, ensure_qdrant_collection

LOGGER = logging.getLogger("openclaw.benchmarks.common")

BENCHMARKS_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BENCHMARKS_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)
TRACE_DIR = RESULTS_DIR / "traces"
TRACE_DIR.mkdir(exist_ok=True)

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
_embed_model = None


# ---------------------------------------------------------------------------
# Benchmark QA tracing / failure attribution
# ---------------------------------------------------------------------------
# This layer is intentionally benchmark-agnostic.  It records enough evidence to
# distinguish retrieval misses from extraction/rerank/reader/formatting misses
# without baking in LoCoMo/LongMemEval-specific labels or answer hacks.

_QUERY_INTENT_PATTERNS: list[tuple[str, str]] = [
    ("aggregation_count", r"\b(how many|how much|total|combined|in total|altogether|count)\b"),
    ("temporal_when", r"\bwhen\s+(?:did|was|were|is|are|do|does|has|have)\b"),
    ("temporal_latest", r"\b(latest|most recent|current|currently|now|newest|last updated)\b"),
    ("temporal_earliest", r"\b(first|earliest|oldest|original|initial)\b"),
    ("temporal_range", r"\b(when|before|after|during|between|date|time|year|month|day)\b"),
    ("preference", r"\b(prefer|preference|favorite|favourite|like|dislike|hate|love)\b"),
    ("contradiction_or_update", r"\b(actually|instead|correction|corrected|wrong|changed|updated|superseded)\b"),
    ("multi_hop", r"\b(why|how did|because|relation|related|connection|compare)\b"),
    ("direct_fact", r"\b(who|what|where|which|whose)\b"),
]


def classify_query_intent(question: str) -> str:
    """Return a coarse, general-purpose query intent for routing/trace analysis."""
    q = (question or "").strip().lower()
    for intent, pattern in _QUERY_INTENT_PATTERNS:
        if re.search(pattern, q):
            return intent
    return "open_ended_summary" if len(q.split()) > 10 else "unknown"


def _answer_token_overlap(text: str, answer: str) -> float:
    """Recall-like token overlap: how much of answer appears in text."""
    answer_tokens = set(normalize_answer(answer).split())
    if not answer_tokens:
        return 0.0
    text_tokens = set(normalize_answer(text).split())
    return len(answer_tokens & text_tokens) / len(answer_tokens)


def evidence_contains_answer(
    retrieved_items: list[dict],
    gold_answer: str,
    *,
    min_overlap: float = 0.6,
) -> bool:
    """Heuristic evidence-hit check for benchmark traces.

    This is not scoring logic; it is diagnostic instrumentation.  It answers:
    did the retrieved evidence probably contain the gold answer somewhere?
    """
    if not gold_answer or not retrieved_items:
        return False
    gold_norm = normalize_answer(gold_answer)
    for item in retrieved_items:
        text = " ".join(
            str(item.get(k) or "")
            for k in ("value", "text", "entity", "property")
        )
        text_norm = normalize_answer(text)
        if gold_norm and gold_norm in text_norm:
            return True
        if _answer_token_overlap(text, gold_answer) >= min_overlap:
            return True
    return False


def classify_failure_bucket(
    *,
    predicted_answer: str,
    gold_answer: str,
    retrieved_items: list[dict],
    reader_used: bool,
    f1: float | None = None,
    exact_match_score: float | None = None,
) -> str:
    """Classify benchmark QA outcome into a general failure/success bucket."""
    if exact_match_score is None:
        exact_match_score = exact_match(predicted_answer, gold_answer)
    if f1 is None:
        f1 = token_f1(predicted_answer, gold_answer)

    if exact_match_score >= 1.0:
        return "success_exact"
    if f1 >= 0.8:
        return "success_partial"

    evidence_hit = evidence_contains_answer(retrieved_items, gold_answer)
    if not retrieved_items:
        return "retrieval_empty"
    if not evidence_hit:
        return "retrieval_miss"
    if reader_used:
        return "reader_miss"

    pred_len = len(normalize_answer(predicted_answer).split())
    gold_len = len(normalize_answer(gold_answer).split())
    if gold_len and pred_len > max(gold_len * 3, gold_len + 8):
        return "formatting_too_verbose"
    return "extraction_or_formatting_miss"


def summarize_retrieved_items(retrieved_items: list[dict], *, max_items: int = 10) -> list[dict]:
    """Compact retrieved evidence for trace JSONL without dumping giant context."""
    summary: list[dict] = []
    for rank, item in enumerate(retrieved_items[:max_items], start=1):
        text = str(item.get("text") or item.get("value") or "")
        summary.append({
            "rank": rank,
            "id": item.get("id") or item.get("item_id"),
            "score": item.get("score"),
            "rerank_score": item.get("rerank_score"),
            "source_type": item.get("source_type"),
            "memory_type": item.get("memory_type"),
            "entity": item.get("entity"),
            "property": item.get("property"),
            "status": item.get("status"),
            "source_agent": item.get("source_agent"),
            "source_session": item.get("source_session"),
            "text_preview": text[:240],
        })
    return summary


class BenchmarkTraceWriter:
    """Append-only JSONL trace writer for benchmark QA diagnostics."""

    def __init__(self, benchmark_name: str, run_label: str | None = None):
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", benchmark_name).strip("_") or "benchmark"
        safe_label = re.sub(r"[^A-Za-z0-9_.-]+", "_", run_label or ts).strip("_")
        self.path = TRACE_DIR / f"{safe_name}_{safe_label}_{ts}.jsonl"
        self.counts: dict[str, int] = {}
        self.count = 0

    def record(
        self,
        *,
        question_id: str | None,
        question: str,
        gold_answer: str,
        predicted_answer: str,
        retrieved_items: list[dict],
        f1: float,
        exact_match_score: float,
        latency: float,
        reader_used: bool,
        metadata: dict[str, Any] | None = None,
    ) -> dict:
        intent = classify_query_intent(question)
        evidence_hit = evidence_contains_answer(retrieved_items, gold_answer)
        bucket = classify_failure_bucket(
            predicted_answer=predicted_answer,
            gold_answer=gold_answer,
            retrieved_items=retrieved_items,
            reader_used=reader_used,
            f1=f1,
            exact_match_score=exact_match_score,
        )
        row = {
            "benchmark": self.path.name.split("_", 1)[0],
            "question_id": question_id,
            "question": question,
            "query_intent": intent,
            "gold_answer": gold_answer,
            "predicted_answer": predicted_answer,
            "f1": f1,
            "exact_match": exact_match_score,
            "latency": latency,
            "reader_used": reader_used,
            "retrieved_count": len(retrieved_items),
            "evidence_hit": evidence_hit,
            "failure_bucket": bucket,
            "retrieved": summarize_retrieved_items(retrieved_items),
            "metadata": metadata or {},
            "recorded_at": datetime.now(timezone.utc).isoformat(),
        }
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, default=str) + "\n")
        self.count += 1
        self.counts[bucket] = self.counts.get(bucket, 0) + 1
        return row

    def summary(self) -> dict:
        return {
            "trace_path": str(self.path),
            "num_traces": self.count,
            "failure_buckets": dict(sorted(self.counts.items())),
        }


# ---------------------------------------------------------------------------
# LLM Reader (optional "performance mode")
# Set LLM_READER_MODEL to enable LLM-based answer generation from retrieved context.
# When unset, falls back to heuristic extraction (zero LLM cost).
# Examples: "openai/gpt-4o-mini", "anthropic/claude-sonnet-4-20250514"
# ---------------------------------------------------------------------------
LLM_READER_MODEL = os.environ.get("LLM_READER_MODEL", "")
_llm_reader_client = None

LLM_READER_PROMPT = """You are an expert at answering questions based on the provided memory context.

Given the question and the retrieved memory items below, provide a concise, direct answer.

Rules:
- Answer ONLY based on the provided context. Do not use external knowledge.
- Be concise — give the shortest correct answer (e.g. "Sweden", not "Caroline moved from Sweden").
- If the answer is a date, give the date. If a name, give the name. If a number, give the number.
- If the context does not contain enough information to answer, say "unknown".
- Do NOT explain your reasoning. Just give the answer.

Question: {question}

Retrieved context:
{context}

Answer:"""


def _get_llm_reader():
    """Lazily initialize the LLM reader client."""
    global _llm_reader_client
    if _llm_reader_client is not None:
        return _llm_reader_client

    if not LLM_READER_MODEL:
        return None

    model = LLM_READER_MODEL

    if model.startswith("openai/") or model.startswith("gpt"):
        try:
            from openai import OpenAI
            _llm_reader_client = {
                "provider": "openai",
                "client": OpenAI(),
                "model": model.replace("openai/", ""),
            }
            LOGGER.info("LLM reader initialized: OpenAI %s", _llm_reader_client["model"])
            return _llm_reader_client
        except Exception as e:
            LOGGER.warning("Failed to init OpenAI reader: %s", e)
            return None

    elif model.startswith("anthropic/") or model.startswith("claude"):
        try:
            import anthropic
            _llm_reader_client = {
                "provider": "anthropic",
                "client": anthropic.Anthropic(),
                "model": model.replace("anthropic/", ""),
            }
            LOGGER.info("LLM reader initialized: Anthropic %s", _llm_reader_client["model"])
            return _llm_reader_client
        except Exception as e:
            LOGGER.warning("Failed to init Anthropic reader: %s", e)
            return None

    elif model.startswith("google/") or model.startswith("gemini"):
        try:
            from google import genai
            client = genai.Client()
            _llm_reader_client = {
                "provider": "google",
                "client": client,
                "model": model.replace("google/", ""),
            }
            LOGGER.info("LLM reader initialized: Google %s", _llm_reader_client["model"])
            return _llm_reader_client
        except Exception as e:
            LOGGER.warning("Failed to init Google reader: %s", e)
            return None

    elif model.startswith("ollama/"):
        try:
            from openai import OpenAI
            ollama_base = os.environ.get("OLLAMA_BASE_URL", "http://192.168.64.1:11434")
            _llm_reader_client = {
                "provider": "openai",  # ollama is OpenAI-compatible
                "client": OpenAI(base_url=f"{ollama_base}/v1", api_key="ollama"),
                "model": model.replace("ollama/", ""),
            }
            LOGGER.info("LLM reader initialized: Ollama %s at %s", _llm_reader_client["model"], ollama_base)
            return _llm_reader_client
        except Exception as e:
            LOGGER.warning("Failed to init Ollama reader: %s", e)
            return None

    else:
        LOGGER.warning("Unknown LLM reader provider in model: %s", model)
        return None


def llm_read_answer(question: str, retrieved_items: list[dict]) -> str | None:
    """Use an LLM to generate an answer from retrieved context.
    Returns the answer string, or None if LLM reader is not configured.
    """
    reader = _get_llm_reader()
    if reader is None:
        return None

    # Build context from retrieved items
    context_parts = []
    for i, item in enumerate(retrieved_items[:20]):
        text = item.get("text", "")
        val = item.get("value", "")
        entity = item.get("entity", "")
        ts = item.get("observed_at", "")
        
        parts = []
        if entity:
            parts.append(f"[{entity}]")
        if ts:
            parts.append(f"({ts})")
        if val:
            parts.append(val)
        elif text:
            parts.append(text[:500])
        
        if parts:
            context_parts.append(f"{i+1}. {' '.join(parts)}")

    if not context_parts:
        return None

    context = "\n".join(context_parts)
    prompt = LLM_READER_PROMPT.format(question=question, context=context)

    try:
        if reader["provider"] == "openai":
            resp = reader["client"].chat.completions.create(
                model=reader["model"],
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=0,
            )
            return resp.choices[0].message.content.strip()

        elif reader["provider"] == "anthropic":
            resp = reader["client"].messages.create(
                model=reader["model"],
                max_tokens=200,
                temperature=0,
                messages=[{"role": "user", "content": prompt}],
            )
            return resp.content[0].text.strip()

        elif reader["provider"] == "google":
            from google.genai import types
            resp = reader["client"].models.generate_content(
                model=reader["model"],
                contents=prompt,
                config=types.GenerateContentConfig(
                    max_output_tokens=200,
                    temperature=0,
                ),
            )
            return resp.text.strip()

    except Exception as e:
        LOGGER.warning("LLM reader error: %s", e)
        return None


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
        entity = str(item.get("entity") or "").strip()
        prop = str(item.get("property") or "").strip()
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

        # 2. Upsert embeddings to pg (single batch, single commit)
        emb_rows = [
            (item["id"], MODEL_NAME, emb.tolist())
            for item, emb in zip(batch_items, embeddings)
        ]
        upsert_memory_embeddings_batch(emb_rows)

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
    # Normalize the ranking key: hybrid_search_memory_items returns rows keyed by
    # SQL columns (final_score / vector_score) with NO "score" key. Every
    # downstream consumer (_merge_ranked_items, _expand_parent_evidence, temporal
    # reorder) reads row.get("score") and falls back to 0.0 when absent — which
    # silently flattens ALL scores to ~0 and destroys ranking. Backfill "score"
    # from the best available signal so the fallback path ranks correctly too.
    for _r in results:
        if _r.get("score") is None:
            _r["score"] = float(
                _r.get("rerank_score")
                or _r.get("final_score")
                or _r.get("vector_score")
                or 0.0
            )
    latency = time.perf_counter() - t0
    return results, latency


def _fetch_memory_items_by_ids(item_ids: list[str]) -> dict[str, dict]:
    """Fetch memory_items rows by id for benchmark evidence expansion."""
    ids = [str(x) for x in item_ids if x]
    if not ids:
        return {}
    try:
        from memory_db import POOL, MEMORY_ITEM_COLUMNS
        with POOL.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"SELECT {', '.join(MEMORY_ITEM_COLUMNS)} FROM memory_items WHERE id = ANY(%s)",
                    (ids,),
                )
                rows = cur.fetchall()
        return {str(row[0]): dict(zip(MEMORY_ITEM_COLUMNS, row)) for row in rows}
    except Exception as e:
        LOGGER.warning("Parent evidence fetch failed: %s", e)
        return {}


def _merge_ranked_items(lane_items: list[tuple[str, float, list[dict]]], limit: int) -> list[dict]:
    """Merge lane outputs by weighted score while deduping by id/text."""
    merged: list[dict] = []
    seen: set[str] = set()
    for lane_name, lane_weight, items in lane_items:
        for rank, item in enumerate(items, start=1):
            item_id = str(item.get("id") or item.get("item_id") or "")
            text = str(item.get("text") or "")
            key = item_id or text
            if not key or key in seen:
                continue
            seen.add(key)
            row = dict(item)
            base_score = float(row.get("score") or row.get("final_score") or 0.0)
            # Small rank discount prevents low-ranked lane tail from outranking better evidence.
            row["score"] = (base_score * lane_weight) - (rank * 0.001)
            row["retrieval_lane"] = lane_name
            row["lane_weight"] = lane_weight
            merged.append(row)
    merged.sort(key=lambda r: float(r.get("score") or 0.0), reverse=True)
    return merged[:limit]


def _expand_parent_evidence(items: list[dict], *, max_parents: int = 20) -> list[dict]:
    """Add parent utterance rows for fact hits that cite source_chunk."""
    parent_ids = []
    existing = {str(x.get("id") or x.get("item_id") or "") for x in items}
    for item in items:
        if item.get("memory_type") == "fact":
            parent_id = str(item.get("source_chunk") or "")
            if parent_id and parent_id not in existing:
                parent_ids.append(parent_id)
    parents = _fetch_memory_items_by_ids(parent_ids[:max_parents])
    if not parents:
        return items

    expanded = list(items)
    seen = {str(x.get("id") or x.get("item_id") or "") for x in expanded}
    child_score_by_parent = {}
    for item in items:
        parent_id = str(item.get("source_chunk") or "")
        if parent_id:
            child_score_by_parent[parent_id] = max(
                float(item.get("score") or 0.0),
                child_score_by_parent.get(parent_id, 0.0),
            )
    for parent_id in parent_ids:
        parent = parents.get(parent_id)
        if not parent or parent_id in seen:
            continue
        row = dict(parent)
        row["score"] = max(0.0, child_score_by_parent.get(parent_id, 0.0) - 0.01)
        row["retrieval_lane"] = "parent_evidence"
        row["source_type"] = "parent_expansion"
        row["expanded_from_fact"] = True
        expanded.append(row)
        seen.add(parent_id)
    expanded.sort(key=lambda r: float(r.get("score") or 0.0), reverse=True)
    return expanded


def _parse_item_time(item: dict) -> float | None:
    """Parse item timestamp for relative temporal ordering."""
    raw = item.get("last_confirmed") or item.get("first_seen") or item.get("observed_at")
    if not raw:
        return None
    try:
        if hasattr(raw, "timestamp"):
            return float(raw.timestamp())
        text = str(raw).strip()
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        return float(datetime.fromisoformat(text).timestamp())
    except Exception:
        return None


def _apply_relative_temporal_order(items: list[dict], intent: str) -> list[dict]:
    """Apply relative temporal ordering inside the retrieved candidate set.

    This deliberately avoids absolute recency-to-now.  For latest/current/update
    queries, newer candidates receive a relative boost. For earliest/original
    queries, older candidates receive the boost. Retrieval scores still matter;
    this is a rerank nudge over the already-matched candidate set.
    """
    if intent not in {"temporal_latest", "temporal_earliest", "contradiction_or_update"}:
        return items

    timed = [(idx, _parse_item_time(item)) for idx, item in enumerate(items)]
    values = [ts for _, ts in timed if ts is not None]
    if len(values) < 2:
        return items

    lo, hi = min(values), max(values)
    span = max(hi - lo, 1.0)
    prefer_new = intent != "temporal_earliest"
    reranked = []
    for idx, item in enumerate(items):
        ts = dict(timed).get(idx)
        row = dict(item)
        base = float(row.get("score") or row.get("final_score") or 0.0)
        if ts is None:
            rel = 0.0
        else:
            newest = (ts - lo) / span
            rel = newest if prefer_new else 1.0 - newest
        # Strong enough to resolve old/new conflicts within matched results,
        # weak enough not to replace semantic retrieval.
        row["relative_temporal_score"] = round(rel, 6)
        row["score"] = base + (0.25 * rel)
        row["temporal_order_applied"] = intent
        reranked.append(row)

    reranked.sort(key=lambda r: float(r.get("score") or 0.0), reverse=True)
    return reranked


def retrieve_typed_lanes(
    query: str,
    *,
    limit: int = 10,
    source_agent_prefix: str | None = None,
    intent: str | None = None,
    use_full_pipeline: bool = True,
    expand_parents: bool = True,
) -> tuple[list[dict], float]:
    """Benchmark QA retrieval with typed lanes and optional parent expansion.

    This is intentionally separate from production `retrieve()`. It prevents
    derived facts and raw utterances from competing in one undifferentiated
    pool, then adds parent utterances for source-backed fact hits.
    """
    try:
        q_intent = intent or classify_query_intent(query)
    except Exception:
        q_intent = intent or "unknown"

    t0 = time.perf_counter()

    # Lane plan: high-precision fact lane first for direct/detail queries;
    # conversation lane remains present for context/narrative evidence.
    if q_intent in {"direct_fact", "preference", "temporal_when", "temporal_latest", "temporal_earliest", "temporal_range", "contradiction_or_update", "aggregation_count"}:
        fact_limit = max(limit * 3, 20)
        conv_limit = max(limit * 2, 12)
        fact_weight = 1.15
        conv_weight = 1.00
    elif q_intent in {"multi_hop", "open_ended_summary"}:
        fact_limit = max(limit * 2, 12)
        conv_limit = max(limit * 3, 20)
        fact_weight = 1.00
        conv_weight = 1.10
    else:
        fact_limit = max(limit * 2, 12)
        conv_limit = max(limit * 2, 12)
        fact_weight = 1.05
        conv_weight = 1.00

    lane_outputs: list[tuple[str, float, list[dict]]] = []
    for lane_name, memory_type, lane_limit, lane_weight in [
        ("fact", "fact", fact_limit, fact_weight),
        ("conversation", "conversation", conv_limit, conv_weight),
    ]:
        try:
            rows, _ = retrieve(
                query,
                limit=lane_limit,
                source_agent_prefix=source_agent_prefix,
                memory_type_filter=memory_type,
                use_full_pipeline=use_full_pipeline,
            )
            # Safety post-filter: ensures prefix is respected even if SQL path didn't filter.
            if source_agent_prefix:
                rows = [r for r in rows if str(r.get("source_agent") or "").startswith(source_agent_prefix)]
            lane_outputs.append((lane_name, lane_weight, rows))
        except Exception as e:
            LOGGER.warning("Typed retrieval lane %s failed: %s", lane_name, e)

    merged = _merge_ranked_items(lane_outputs, limit=max(limit * 3, limit + 10))
    if expand_parents:
        merged = _expand_parent_evidence(merged)
    merged = _apply_relative_temporal_order(merged, q_intent)
    merged = merged[:limit]
    return merged, time.perf_counter() - t0


# Connective words that signal a question spans MULTIPLE entities/events and thus
# needs more than one retrieval probe (e.g. "what do X and Y have in common",
# "which city did both A and B visit"). A single dense query for the *joint*
# phrasing tends to retrieve neither side's specific evidence well.
_MULTIHOP_TRIGGERS = re.compile(
    r"\b(both|in common|each other|compare|comparison|difference between|"
    r"versus|vs\.?|as well as|along with)\b",
    re.IGNORECASE,
)
# Proper-noun-ish capitalized tokens (entity candidates). Deliberately simple:
# LoCoMo/LongMemEval speakers are first names. We avoid an NER dependency.
_CAPWORD = re.compile(r"\b([A-Z][a-z]{2,})\b")
_STOP_CAPS = {
    "What", "When", "Where", "Which", "Who", "Why", "How", "Did", "Does", "Do",
    "Is", "Are", "Was", "Were", "The", "Their", "They", "Both", "And",
}


def _extract_entity_subqueries(question: str) -> list[str]:
    """Decompose a multi-entity question into per-entity sub-queries.

    "Which city have both Jean and John visited?" ->
        ["Which city have Jean visited?", "Which city have John visited?"]
    Falls back to [] when fewer than 2 distinct entity candidates are found.
    """
    entities = []
    for m in _CAPWORD.finditer(question or ""):
        tok = m.group(1)
        if tok in _STOP_CAPS or tok in entities:
            continue
        entities.append(tok)
    if len(entities) < 2:
        return []
    # Build one probe per entity by removing the OTHER entity names + connectives.
    subs = []
    for keep in entities:
        probe = question
        for other in entities:
            if other == keep:
                continue
            # drop "and Other", "both Other", or bare "Other"
            probe = re.sub(rf"\b(and|both)\s+{re.escape(other)}\b", "", probe)
            probe = re.sub(rf"\b{re.escape(other)}\b", "", probe)
        probe = _MULTIHOP_TRIGGERS.sub("", probe)
        # Clean up connective/preposition debris left by entity removal so the
        # sub-query reads naturally (e.g. "What do and Gina have" -> "What do Gina
        # have"; "When was in Paris" -> "When was in Paris" stays, but a dangling
        # leading/trailing "and"/"in"/"with" is dropped).
        probe = re.sub(r"\b(and|both|as well as|along with)\b\s+", " ", probe)
        probe = re.sub(r"\s+(and|both|with|as well as|along with)\b", " ", probe)
        probe = re.sub(r"\s+([?.!,;:])", r"\1", probe)
        probe = re.sub(r"\s{2,}", " ", probe).strip()
        if probe and probe not in subs:
            subs.append(probe)
    return subs if len(subs) >= 2 else []


def retrieve_multihop(
    query: str,
    *,
    limit: int = 10,
    source_agent_prefix: str | None = None,
    intent: str | None = None,
    use_full_pipeline: bool = True,
    expand_parents: bool = True,
) -> tuple[list[dict], float, dict]:
    """Recall-oriented retrieval for multi-hop / aggregation / multi-entity QA.

    Strategy (purely additive over retrieve_typed_lanes):
      1. Always run the base typed-lane retrieval for the original query.
      2. If the question triggers a multi-entity pattern ("both A and B",
         "in common", ...), decompose into per-entity sub-queries and retrieve
         EACH separately, so both sides' evidence reaches the reader instead of
         being crowded out by the joint-phrasing query.
      3. Merge all candidate pools (deduped by id/text), preserving the best
         score per item, then take the top `limit`.

    Returns (results, latency_seconds, diag) where diag reports what fired, so
    the benchmark trace can attribute gains to the multi-hop path honestly.
    """
    t0 = time.perf_counter()
    q_intent = intent or classify_query_intent(query)

    base, _ = retrieve_typed_lanes(
        query, limit=limit, source_agent_prefix=source_agent_prefix,
        intent=q_intent, use_full_pipeline=use_full_pipeline,
        expand_parents=expand_parents,
    )

    subqueries: list[str] = []
    is_multientity = bool(_MULTIHOP_TRIGGERS.search(query or ""))
    if is_multientity or q_intent in {"multi_hop", "aggregation_count", "open_ended_summary"}:
        subqueries = _extract_entity_subqueries(query)

    if not subqueries:
        # No decomposition applicable — base result is the answer. Keep the same
        # 3-tuple contract so callers don't branch.
        return base[:limit], time.perf_counter() - t0, {
            "multihop_fired": False,
            "subqueries": [],
            "base_n": len(base),
        }

    # Per-entity sub-retrievals. Pull a few extra per sub-query so the merge has
    # real material to work with, then dedup down to `limit`.
    sub_limit = max(limit // max(len(subqueries), 1), 4)
    pools: list[tuple[str, float, list[dict]]] = [("base", 1.05, base)]
    for sq in subqueries:
        try:
            rows, _ = retrieve_typed_lanes(
                sq, limit=sub_limit, source_agent_prefix=source_agent_prefix,
                use_full_pipeline=use_full_pipeline, expand_parents=expand_parents,
            )
            pools.append((f"hop:{sq[:24]}", 1.0, rows))
        except Exception as e:
            LOGGER.warning("Multi-hop sub-query failed (%s): %s", sq, e)

    merged = _merge_ranked_items(pools, limit=max(limit * 3, limit + 10))
    merged = _apply_relative_temporal_order(merged, q_intent)
    merged = merged[:limit]
    return merged, time.perf_counter() - t0, {
        "multihop_fired": True,
        "subqueries": subqueries,
        "base_n": len(base),
        "merged_n": len(merged),
    }


def retrieve_and_answer(
    query: str,
    *,
    limit: int = 10,
    source_agent_prefix: str | None = None,
    entity_filter: str | None = None,
    session_filter: str | None = None,
    memory_type_filter: str | None = None,
    use_full_pipeline: bool = True,
) -> tuple[str, list[dict], float]:
    """
    Retrieve + optionally generate answer via LLM reader.
    Returns (answer_text, results, latency_seconds).

    If LLM_READER_MODEL is set, uses the LLM to generate an answer from context.
    Otherwise returns the top result's text (same as before).
    """
    results, latency = retrieve(
        query, limit=limit,
        source_agent_prefix=source_agent_prefix,
        entity_filter=entity_filter,
        session_filter=session_filter,
        memory_type_filter=memory_type_filter,
        use_full_pipeline=use_full_pipeline,
    )

    # Try LLM reader first
    llm_answer = llm_read_answer(query, results)
    if llm_answer is not None:
        return llm_answer, results, latency

    # Fallback: return top result text
    if results:
        val = results[0].get("value", "")
        if val and len(val) > 3:
            return val, results, latency
        return results[0].get("text", "")[:300], results, latency

    return "", results, latency


# Cache the cross-encoder model at module level to avoid reloading per query
_reranker_model = None
_RERANKER_DISABLED = object()


def _benchmark_reranker_disabled() -> bool:
    """Return True when benchmark reranking should be skipped.

    Smoke tests must be able to run even when the optional cross-encoder model
    is not cached.  Without this guard, `CrossEncoder(...)` can block on a slow
    HuggingFace download and prevent end-to-end validation of the rest of the
    memory pipeline.
    """
    return os.environ.get("OPENCLAW_BENCHMARK_DISABLE_RERANKER", "").lower() in {
        "1", "true", "yes", "on"
    }


def _cross_encoder_cache_incomplete() -> bool:
    """Detect a partial local HuggingFace cross-encoder cache."""
    cache_root = Path.home() / ".cache" / "huggingface" / "hub" / "models--cross-encoder--ms-marco-MiniLM-L-6-v2"
    return cache_root.exists() and any(cache_root.rglob("*.incomplete"))


def _get_reranker():
    global _reranker_model
    if _reranker_model is _RERANKER_DISABLED:
        return None
    if _benchmark_reranker_disabled():
        _reranker_model = _RERANKER_DISABLED
        LOGGER.info("Benchmark cross-encoder reranker disabled by OPENCLAW_BENCHMARK_DISABLE_RERANKER")
        return None
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
            if _cross_encoder_cache_incomplete():
                _reranker_model = _RERANKER_DISABLED
                LOGGER.warning("Skipping benchmark cross-encoder reranker: incomplete local HuggingFace cache")
                return None
            from sentence_transformers import CrossEncoder
            _reranker_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", local_files_only=True)
            LOGGER.info("Loaded cross-encoder reranker from local cache: cross-encoder/ms-marco-MiniLM-L-6-v2")
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
            if reranker is not None:
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
    s = str(s).lower()
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


# Sentinel gold value marking an unanswerable/abstention question.
NO_ANSWER_SENTINEL = "__NO_ANSWER__"


class JudgeError(Exception):
    """Raised by llm_judge_answer when strict_judge=True and the judge call fails.

    In the bake-off path we must NOT silently substitute token_f1 on a judge
    outage (that fabricates scores); the caller marks the item judge_error and
    excludes it from accuracy instead."""


def llm_judge_answer(
    question: str,
    gold_answer: str,
    predicted_answer: str,
    *,
    model: str = "claude-opus-4-6",
    strict_judge: bool = False,
) -> float:
    """
    Use an LLM to judge whether predicted_answer is correct given gold_answer.
    Supports Anthropic by default and OpenAI-compatible Ollama via
    model="ollama/<model>" (for example: ollama/qwen3:32b).
    Returns score in [0, 1].

    strict_judge=True: on any judge failure, raise JudgeError instead of the
    legacy silent token_f1 fallback (used by the bake-off so an outage can't fake
    scores). Default False preserves all existing callers' behavior.

    Abstention: when gold_answer == NO_ANSWER_SENTINEL the question is treated as
    unanswerable and an abstention-specific rubric is used. For all other gold
    values the prompt is byte-for-byte the legacy answerable prompt, so answerable
    scoring is unchanged.
    """
    import re

    is_abstention = str(gold_answer).strip() == NO_ANSWER_SENTINEL

    if is_abstention:
        prompt = f"""You are evaluating a question-answering system.

Question: {question}
Ground Truth: This question is UNANSWERABLE from the available information.
Predicted Answer: {predicted_answer}

A correct prediction abstains or says the answer cannot be determined / is not
stated. A confident specific answer is INCORRECT. Score 1.0 for a correct
abstention, 0.0 for a confident specific (non-abstaining) answer.
Respond with ONLY a JSON object: {{"score": <float 0.0-1.0>, "reason": "<brief reason>"}}
Do not include chain-of-thought, markdown, or any text outside the JSON object."""
    else:
        prompt = f"""You are evaluating a question-answering system.

Question: {question}
Ground Truth Answer: {gold_answer}
Predicted Answer: {predicted_answer}

Is the predicted answer correct? Consider partial credit for partially correct answers.
Respond with ONLY a JSON object: {{"score": <float 0.0-1.0>, "reason": "<brief reason>"}}
Do not include chain-of-thought, markdown, or any text outside the JSON object."""

    def _score_from_text(text: str) -> float:
        text = text.strip()
        # Qwen-style models can still emit a thinking block; strip it if present.
        text = re.sub(r"(?is)<think>.*?</think>", "", text).strip()
        match = re.search(r"\{.*\}", text, re.S)
        data = json.loads(match.group(0) if match else text)
        return max(0.0, min(1.0, float(data.get("score", 0.0))))

    try:
        if model.startswith("ollama/"):
            from openai import OpenAI

            ollama_base = os.environ.get("OLLAMA_BASE_URL", "http://192.168.64.1:11435")
            client = OpenAI(base_url=f"{ollama_base}/v1", api_key="ollama")
            resp = client.chat.completions.create(
                model=model.removeprefix("ollama/"),
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=200,
            )
            text = resp.choices[0].message.content or ""
            return _score_from_text(text)

        if model.startswith("openai/") or model.startswith("gpt"):
            from openai import OpenAI

            client = OpenAI()
            resp = client.chat.completions.create(
                model=model.removeprefix("openai/"),
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=200,
            )
            text = resp.choices[0].message.content or ""
            return _score_from_text(text)

        import anthropic

        client = anthropic.Anthropic()
        msg = client.messages.create(
            model=model.removeprefix("anthropic/"),
            max_tokens=150,
            messages=[{"role": "user", "content": prompt}],
        )
        text = msg.content[0].text.strip()
        return _score_from_text(text)
    except Exception as e:
        if strict_judge:
            raise JudgeError(str(e)) from e
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
