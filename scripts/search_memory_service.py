"""
FastAPI-based HTTP search service exposing the memory retrieval pipeline.
Provides /search, /context, and /health endpoints for external consumers.
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Annotated

import httpx
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
from sentence_transformers import CrossEncoder, SentenceTransformer

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from checkpoint_db import close_pool
from memory_db import search_memory_items_by_terms
from graph_search_adapter import search_graph_text
from memory_retrieval_contract import build_memory_pack
from memory_routing_models import WorkItemMemoryMetadata
from normal_agent_api import NormalAgentRequest, build_normal_agent_packet_response
from orchestrator_context_core import fetch_orchestrator_context
from qdrant_search_adapter import search_qdrant_text

LOGGER = logging.getLogger("openclaw.search_service")
logging.basicConfig(
    level=os.environ.get("OPENCLAW_SEARCH_LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)

MODEL_NAME = os.environ.get("OPENCLAW_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
RERANK_MODEL = os.environ.get("OPENCLAW_RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
TOP_K = int(os.environ.get("OPENCLAW_SEARCH_TOPK", "8"))
RERANK_CAP = int(os.environ.get("OPENCLAW_RERANK_CAP", "20"))
RERANK_TEXT_CHAR_CAP = int(os.environ.get("OPENCLAW_RERANK_TEXT_CHAR_CAP", "1000"))
CIRCUIT_BREAKER_SECONDS = int(os.environ.get("OPENCLAW_SEARCH_CB_SECONDS", "60"))
DOCS_ENABLED = os.environ.get("OPENCLAW_SEARCH_ENABLE_DOCS", "0") == "1"

PREFERRED_DEVICE = os.environ.get("OPENCLAW_SEARCH_DEVICE", "auto").lower()
EMBED_BACKEND = os.environ.get("OPENCLAW_EMBED_BACKEND", "default").lower()

# HF model-load resilience. The embed + rerank checkpoints are vendored in the
# local HF cache, but by default sentence-transformers/transformers still make
# an (unauthenticated) HF Hub network round-trip on every cold load to check
# the cache. That makes cold starts depend on network reachability + HF rate
# limits even though everything needed is already on disk — a flaky network or
# HF outage can stall/fail a restart for no good reason.
#
# OPENCLAW_HF_LOCAL_ONLY controls the policy:
#   "auto" (default) -> try cache-only first; only if the cache is genuinely
#                       missing do we fall back to a network-allowed load
#                       (covers first-ever run / un-warmed cache).
#   "1"/"true"/"yes" -> strict: cache-only, never touch the network (fail loud
#                       if a model isn't cached). Recommended for prod once the
#                       cache is warmed.
#   "0"/"false"/"no" -> legacy: always allow network on load.
HF_LOCAL_ONLY = os.environ.get("OPENCLAW_HF_LOCAL_ONLY", "auto").strip().lower()

# T2: absolute recency-decay ranking knobs. Defaults mirror temporal.py so the
# SQL ranking term and the Python-surfaced recency_bonus agree. Setting weight=0
# reproduces legacy ordering exactly (regression-safe kill switch).
try:
    from temporal import DEFAULT_RECENCY_WEIGHT as _RW, DEFAULT_HALF_LIFE_DAYS as _HL
except Exception:  # pragma: no cover - temporal.py is always present
    _RW, _HL = 0.6, 14.0
RECENCY_WEIGHT = float(os.environ.get("OPENCLAW_SEARCH_RECENCY_WEIGHT", str(_RW)))
RECENCY_HALF_LIFE_DAYS = float(os.environ.get("OPENCLAW_SEARCH_RECENCY_HALF_LIFE_DAYS", str(_HL)))


class MetadataFilter(BaseModel):
    """Optional metadata filters to scope search results."""
    entity: str | None = Field(default=None, description="Filter by entity name (exact match)")
    property: str | None = Field(default=None, description="Filter by property name (exact match)")
    session_id: str | None = Field(default=None, description="Filter by source_session (exact match)")
    source_agent: str | None = Field(default=None, description="Filter by source_agent (exact match)")
    memory_type: str | None = Field(default=None, description="Filter by memory_type (exact match)")
    scope: str | None = Field(default=None, description="Filter by scope (exact match)")
    project_id: str | None = Field(default=None, description="Filter by project_id (exact match)")
    min_confidence: float | None = Field(default=None, ge=0.0, le=1.0, description="Minimum confidence threshold")
    # T3: temporal-window filters. All operate on the EFFECTIVE timestamp
    # COALESCE(last_confirmed, first_seen, created_at) so they agree with
    # temporal.py's resolution. Accept ISO-8601 strings or date (YYYY-MM-DD).
    since: str | None = Field(
        default=None,
        description="Only memories with effective timestamp >= this (ISO-8601 or YYYY-MM-DD).",
    )
    until: str | None = Field(
        default=None,
        description="Only memories with effective timestamp <= this (ISO-8601 or YYYY-MM-DD).",
    )
    as_of: str | None = Field(
        default=None,
        description="Point-in-time view: exclude memories whose effective timestamp is "
        "after this instant (ISO-8601 or YYYY-MM-DD). Enables deterministic replay.",
    )


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=TOP_K, ge=1, le=50)
    filters: MetadataFilter | None = Field(default=None, description="Metadata filters to scope results")
    rerank: bool = Field(default=True, description="Whether to apply cross-encoder reranking")


class SearchResult(BaseModel):
    text: str
    path: str = ""
    source_type: str = ""
    score: float | None = None
    rerank_score: float | None = None
    # T1: temporal metadata surfaced to the agent (additive, all optional).
    effective_ts: str | None = None
    age_days: float | None = None
    temporal_bucket: str | None = None
    temporal_label: str | None = None
    temporal_confidence: str | None = None
    recency_bonus: float | None = None


class SearchResponse(BaseModel):
    ok: bool
    query: str
    results: list[SearchResult]
    meta: dict[str, Any]


class OrchestratorContextRequest(BaseModel):
    project_id: str | None = None
    subproject_id: str | None = None
    role: str
    mode: str = "normal"
    work_id: str
    kind: str = "task"
    tags: list[str] = Field(default_factory=list)
    limit: int = Field(default=6, ge=1, le=50)


class SourceCircuit:
    def __init__(self, name: str, cooldown_seconds: int) -> None:
        self.name = name
        self.cooldown_seconds = cooldown_seconds
        self.open_until = 0.0
        self.last_error: str | None = None

    def is_open(self) -> bool:
        return time.time() < self.open_until

    def trip(self, err: Exception) -> None:
        self.open_until = time.time() + self.cooldown_seconds
        self.last_error = repr(err)

    def snapshot(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "open": self.is_open(),
            "open_until": self.open_until,
            "last_error": self.last_error,
        }


def detect_device() -> str:
    if PREFERRED_DEVICE != "auto":
        return PREFERRED_DEVICE
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def _load_cache_first(label: str, loader):
    """Load a model cache-first per the OPENCLAW_HF_LOCAL_ONLY policy.

    `loader` is a callable taking a single bool `local_files_only` and returning
    the loaded model. This centralizes the network-resilience policy for both
    the embed and rerank models:

      - strict ("1"/"true"/"yes"): load cache-only; any failure propagates.
      - legacy ("0"/"false"/"no"): load with network allowed.
      - auto (default): try cache-only first; on failure (e.g. genuine cache
        miss on first run) fall back ONCE to a network-allowed load, logging
        the transition so the network dependency is visible.

    Resolving at load time keeps cold starts fast and network-independent when
    the cache is warm (the normal production case) without breaking first-run
    downloads.
    """
    strict = HF_LOCAL_ONLY in {"1", "true", "yes", "on"}
    legacy_online = HF_LOCAL_ONLY in {"0", "false", "no", "off"}

    if legacy_online:
        return loader(False)

    try:
        return loader(True)
    except Exception as e:
        if strict:
            # Explicit strict-offline: do NOT silently reach for the network.
            LOGGER.error(
                "%s cache-only load failed under strict OPENCLAW_HF_LOCAL_ONLY; "
                "model not present in local HF cache: %r",
                label,
                e,
            )
            raise
        LOGGER.warning(
            "%s not in local HF cache (%r); falling back to network-allowed "
            "load. Warm the cache and set OPENCLAW_HF_LOCAL_ONLY=1 for "
            "deterministic offline starts.",
            label,
            e,
        )
        return loader(False)


def load_embed_model(device: str) -> SentenceTransformer:
    if EMBED_BACKEND == "onnx" and device == "cpu":
        try:
            return _load_cache_first(
                "embed_model(onnx)",
                lambda lfo: SentenceTransformer(
                    MODEL_NAME, device=device, backend="onnx", local_files_only=lfo
                ),
            )
        except Exception as e:
            LOGGER.warning("embed_model_onnx_fallback error=%r", e)
    return _load_cache_first(
        "embed_model",
        lambda lfo: SentenceTransformer(MODEL_NAME, device=device, local_files_only=lfo),
    )


def load_reranker(device: str) -> CrossEncoder:
    try:
        return _load_cache_first(
            "reranker",
            lambda lfo: CrossEncoder(RERANK_MODEL, device=device, local_files_only=lfo),
        )
    except TypeError:
        # Older CrossEncoder signatures may not accept device=; preserve the
        # original fallback, still cache-first for network resilience.
        return _load_cache_first(
            "reranker(no-device)",
            lambda lfo: CrossEncoder(RERANK_MODEL, local_files_only=lfo),
        )


def result_fingerprint(row: dict[str, Any]) -> str:
    text = str(row.get("text", ""))
    path = str(row.get("path", ""))
    raw = f"{path}\n{text}".encode("utf-8", errors="ignore")
    return hashlib.md5(raw).hexdigest()


def is_infra_error(err: Exception) -> bool:
    if isinstance(err, (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout)):
        return True

    msg = repr(err).lower()
    infra_markers = [
        "connection refused",
        "timed out",
        "timeout",
        "temporarily unavailable",
        "service unavailable",
        "connection error",
        "connection reset",
        "refused",
        "5xx",
    ]
    return any(marker in msg for marker in infra_markers)


def truncate_for_rerank(text: str) -> str:
    if len(text) <= RERANK_TEXT_CHAR_CAP:
        return text
    return text[:RERANK_TEXT_CHAR_CAP]


async def run_in_executor(executor: ThreadPoolExecutor, fn, *args, **kwargs):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(executor, lambda: fn(*args, **kwargs))


def _apply_metadata_filter(rows: list[dict[str, Any]], filters: MetadataFilter | None) -> list[dict[str, Any]]:
    """Apply metadata filters to a list of result rows (post-retrieval filtering)."""
    if filters is None:
        return rows

    filtered = []
    for row in rows:
        if filters.entity is not None and str(row.get("entity", "") or "") != filters.entity:
            continue
        if filters.property is not None and str(row.get("property", "") or "") != filters.property:
            continue
        if filters.session_id is not None and str(row.get("source_session", "") or "") != filters.session_id:
            continue
        if filters.source_agent is not None and str(row.get("source_agent", "") or "") != filters.source_agent:
            continue
        if filters.memory_type is not None and str(row.get("memory_type", "") or "") != filters.memory_type:
            continue
        if filters.scope is not None and str(row.get("scope", "") or "") != filters.scope:
            continue
        if filters.project_id is not None and str(row.get("project_id", "") or "") != filters.project_id:
            continue
        if filters.min_confidence is not None:
            conf = float(row.get("confidence", 0) or 0)
            if conf < filters.min_confidence:
                continue
        # T3: temporal-window filtering for non-SQL sources (qdrant/graph).
        # Mirrors the SQL push-down semantics EXACTLY: filter on the effective
        # ts COALESCE(last_confirmed, first_seen, created_at) parsed via
        # temporal.parse_ts; rows with NO usable timestamp are KEPT (not
        # dropped), since we can't place them on the timeline. NOTE: the Qdrant
        # payload currently carries no timestamps, so those rows are effectively
        # undated here and pass through — the SQL source enforces the window.
        if filters.since is not None or filters.until is not None or filters.as_of is not None:
            from temporal import parse_ts as _parse_ts, resolve_effective_ts as _eff
            eff_dt, _conf = _eff(row)
            if eff_dt is not None:
                if filters.since is not None:
                    _d = _parse_ts(filters.since)
                    if _d is not None and eff_dt < _d:
                        continue
                if filters.until is not None:
                    _d = _parse_ts(filters.until)
                    if _d is not None and eff_dt > _d:
                        continue
                if filters.as_of is not None:
                    _d = _parse_ts(filters.as_of)
                    if _d is not None and eff_dt > _d:
                        continue
        filtered.append(row)

    return filtered


def _build_metadata_sql(filters: MetadataFilter | None) -> tuple[str, list[Any]]:
    """Build additional SQL WHERE clauses from metadata filters.

    Returns (sql_fragment, params) to append to a query.
    The sql_fragment starts with ' AND ...' if non-empty.
    """
    if filters is None:
        return "", []

    clauses = []
    params: list[Any] = []

    if filters.entity is not None:
        clauses.append("entity = %s")
        params.append(filters.entity)
    if filters.property is not None:
        clauses.append("property = %s")
        params.append(filters.property)
    if filters.session_id is not None:
        clauses.append("source_session = %s")
        params.append(filters.session_id)
    if filters.source_agent is not None:
        clauses.append("source_agent = %s")
        params.append(filters.source_agent)
    if filters.memory_type is not None:
        clauses.append("memory_type = %s")
        params.append(filters.memory_type)
    if filters.scope is not None:
        clauses.append("scope = %s")
        params.append(filters.scope)
    if filters.project_id is not None:
        clauses.append("project_id = %s")
        params.append(filters.project_id)
    if filters.min_confidence is not None:
        clauses.append("COALESCE(confidence, 0) >= %s")
        params.append(filters.min_confidence)

    # T3: temporal-window filters on the EFFECTIVE timestamp
    # COALESCE(last_confirmed, first_seen, created_at). Rows with a NULL
    # effective ts (fully undated) are NOT excluded by these windows — we can't
    # place them on the timeline, so we keep them findable rather than silently
    # dropping them. Callers wanting strict windows can additionally require a
    # timestamp via other means. Values are parsed by temporal.parse_ts so the
    # SQL and Qdrant push-downs agree byte-for-byte.
    _EFF = "COALESCE(last_confirmed, first_seen, created_at)"
    from temporal import parse_ts as _parse_ts

    if filters.since is not None:
        _dt = _parse_ts(filters.since)
        if _dt is not None:
            clauses.append(f"({_EFF} IS NULL OR {_EFF} >= %s)")
            params.append(_dt)
    if filters.until is not None:
        _dt = _parse_ts(filters.until)
        if _dt is not None:
            clauses.append(f"({_EFF} IS NULL OR {_EFF} <= %s)")
            params.append(_dt)
    if filters.as_of is not None:
        _dt = _parse_ts(filters.as_of)
        if _dt is not None:
            # Point-in-time: exclude memories that became known AFTER as_of.
            clauses.append(f"({_EFF} IS NULL OR {_EFF} <= %s)")
            params.append(_dt)

    if not clauses:
        return "", []

    return " AND " + " AND ".join(clauses), params


def _deduplicate_entity_property(rows: list[dict]) -> list[dict]:
    """When multiple results share the same entity+property, keep only the newest.

    This is the core conflict resolution mechanism: if "user.ide_theme = dark mode"
    (from May) and "user.ide_theme = light mode" (from June) both appear in results,
    only the June one survives. Generic properties like 'utterance' or 'fact' are
    excluded since they don't represent replaceable facts.
    """
    from datetime import datetime

    GENERIC_PROPS = frozenset({"utterance", "fact", "observation", "episodic", ""})

    def _parse_ts(item):
        for key in ("last_confirmed", "first_seen", "created_at"):
            ts = item.get(key)
            if ts is None:
                continue
            if isinstance(ts, datetime):
                return ts
            try:
                s = str(ts).replace("T", " ").split("+")[0].split("Z")[0].strip()
                for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
                    try:
                        return datetime.strptime(s[:26], fmt)
                    except ValueError:
                        continue
            except Exception:
                pass
        return datetime.min

    seen = {}       # (entity, property) -> (index, timestamp)
    remove = set()

    for i, item in enumerate(rows):
        entity = (item.get("entity") or "").strip().lower()
        prop = (item.get("property") or "").strip().lower()

        if not entity or prop in GENERIC_PROPS:
            continue

        key = (entity, prop)
        ts = _parse_ts(item)

        if key in seen:
            prev_idx, prev_ts = seen[key]
            if ts > prev_ts:
                remove.add(prev_idx)
                seen[key] = (i, ts)
            else:
                remove.add(i)
        else:
            seen[key] = (i, ts)

    if remove:
        rows = [r for i, r in enumerate(rows) if i not in remove]

    return rows


def search_memory_items_filtered(
    query: str,
    filters: MetadataFilter | None = None,
    status: str = "active",
    limit: int = 50,
    query_embedding: list[float] | None = None,
) -> list[dict[str, Any]]:
    """Hybrid search (FTS + vector + structured) with metadata filters in SQL.

    This is the full hybrid scoring query from hybrid_search_memory_items()
    but with metadata WHERE clauses injected so filtering happens at the DB
    level rather than post-retrieval.  If query_embedding is not provided,
    it is computed automatically.
    """
    from memory_db import POOL

    query = (query or "").strip()
    if not query:
        return []

    # Compute embedding if not provided
    if query_embedding is None:
        try:
            # Cache-first load for the same network-resilience reason as the
            # primary loaders (this ad-hoc fallback path constructs its own
            # model rather than reusing app.state.embed_model).
            _st = _load_cache_first(
                "embed_model(filtered-fallback)",
                lambda lfo: SentenceTransformer(MODEL_NAME, local_files_only=lfo),
            )
            query_embedding = _st.encode(query, normalize_embeddings=True, convert_to_numpy=True).tolist()
        except Exception:
            # Fall back to FTS-only if embedding fails
            return _search_memory_items_filtered_fts_only(query, filters, status, limit)

    extra_sql, extra_params = _build_metadata_sql(filters)
    vec = "[" + ",".join(str(float(x)) for x in query_embedding) + "]"

    # All positional %s params to avoid psycopg mixed-placeholder errors
    sql = f"""
        WITH q AS (
            SELECT
                ARRAY(
                    SELECT qt
                    FROM unnest(
                        regexp_split_to_array(
                            regexp_replace(LOWER(%s), '[^a-z0-9_ ]', ' ', 'g'),
                            '\\s+'
                        )
                    ) AS qt
                    WHERE qt <> ''
                      AND length(qt) >= 4
                      AND qt NOT IN (
                          'this', 'that', 'with', 'from', 'into', 'they', 'them',
                          'then', 'than', 'have', 'will', 'would', 'could', 'should',
                          'there', 'their', 'about', 'because', 'system',
                          'person', 'does', 'what', 'where', 'love', 'enjoy'
                      )
                ) AS qterms,
                websearch_to_tsquery('english', %s) AS tsq
        ),
        scored AS (
            SELECT
                m.id, m.text, m.memory_type, m.scope, m.entity, m.property, m.value, m.status,
                m.confidence, 0.8 as importance, m.freshness_class,
                m.source_agent, m.source_session, m.source_chunk,
                m.first_seen, m.last_confirmed, m.supersedes,
                m.tags, m.notes, m.candidate_id, m.candidate_score, m.candidate_reasons,
                m.suggested_route, m.target_file, m.target_section,
                m.sensitivity, m.approved_at, m.approved_by, m.approval_source,
                m.rejected_at, m.rejected_by, m.rejection_reason, m.rejection_source,
                m.ranking_bonus, m.ranking_penalty, m.feedback_last_applied_at,

                ts_rank_cd(
                    to_tsvector(
                        'english',
                        COALESCE(m.text, '') || ' ' ||
                        COALESCE(m.entity, '') || ' ' ||
                        COALESCE(m.property, '') || ' ' ||
                        COALESCE(m.value, '') || ' ' ||
                        COALESCE(m.memory_type, '') || ' ' ||
                        COALESCE(m.scope, '')
                    ),
                    q.tsq
                ) AS fts_rank,

                1 - (e.embedding <=> %s::vector) AS vector_score,

                (
                    0.18 * (
                        SELECT COUNT(*)
                        FROM unnest(q.qterms) qt2
                        WHERE qt2 <> ''
                          AND qt2 = ANY(
                              regexp_split_to_array(
                                  regexp_replace(LOWER(COALESCE(m.text, '')), '[^a-z0-9_ ]', ' ', 'g'),
                                  '\\s+'
                              )
                          )
                    )
                    +
                    0.20 * (
                        SELECT COUNT(*)
                        FROM unnest(q.qterms) qt2
                        WHERE qt2 <> ''
                          AND qt2 = ANY(
                              regexp_split_to_array(
                                  regexp_replace(LOWER(COALESCE(m.entity, '')), '[^a-z0-9_ ]', ' ', 'g'),
                                  '\\s+'
                              )
                          )
                    )
                    +
                    0.45 * (
                        SELECT COUNT(*)
                        FROM unnest(q.qterms) qt2
                        WHERE qt2 <> ''
                          AND qt2 = ANY(
                              regexp_split_to_array(
                                  regexp_replace(LOWER(COALESCE(m.property, '')), '[^a-z0-9_ ]', ' ', 'g'),
                                  '\\s+'
                              )
                          )
                    )
                    +
                    0.55 * (
                        SELECT COUNT(*)
                        FROM unnest(q.qterms) qt2
                        WHERE qt2 <> ''
                          AND qt2 = ANY(
                              regexp_split_to_array(
                                  regexp_replace(LOWER(COALESCE(m.value, '')), '[^a-z0-9_ ]', ' ', 'g'),
                                  '\\s+'
                              )
                          )
                    )
                ) AS structured_bonus,

                COALESCE(0.8, 0.75) * 0.25 AS importance_bonus

            FROM memory_items m
            JOIN memory_item_embeddings e ON e.memory_id = m.id
            CROSS JOIN q
            WHERE m.status = %s
              AND m.sensitivity = ANY(%s)
              {extra_sql}
        ),
        -- Relative temporal ranking: within items sharing the same
        -- entity+property, newer items get a bonus over older ones.
        -- This ensures "latest" queries prefer the most recent version.
        with_relative_recency AS (
            SELECT
                s.*,
                CASE
                    WHEN s.entity IS NOT NULL AND s.property IS NOT NULL THEN
                        -- rank 1 = newest within group
                        0.8 * (1.0 - (
                            COALESCE(
                                (ROW_NUMBER() OVER (
                                    PARTITION BY s.entity, s.property
                                    ORDER BY COALESCE(s.last_confirmed, s.first_seen, '1970-01-01'::timestamptz) DESC
                                ) - 1)::float
                                / NULLIF(COUNT(*) OVER (PARTITION BY s.entity, s.property) - 1, 0),
                                0.0
                            )
                        ))
                    ELSE 0.0
                END AS temporal_recency_bonus,
                -- T2: ABSOLUTE recency decay (independent of entity/property groups).
                -- bonus = weight * 0.5 ^ (age_days / half_life). High-confidence
                -- only: uses COALESCE(last_confirmed, first_seen) and is NULL/zero
                -- when only created_at exists (matches temporal.py confidence gate).
                -- Negative age (future ts / clock skew) clamped to 0 via GREATEST.
                CASE
                    WHEN %s > 0.0 AND COALESCE(s.last_confirmed, s.first_seen) IS NOT NULL THEN
                        %s * power(
                            0.5,
                            GREATEST(
                                0.0,
                                EXTRACT(EPOCH FROM (now() - COALESCE(s.last_confirmed, s.first_seen))) / 86400.0
                            ) / %s
                        )
                    ELSE 0.0
                END AS abs_recency_bonus
            FROM scored s
        ),
        ranked AS (
            SELECT
                *,
                (
                    (fts_rank * 3.0)
                    + (vector_score * 1.1)
                    + structured_bonus
                    + importance_bonus
                    + temporal_recency_bonus
                    + abs_recency_bonus
                ) AS final_score
            FROM with_relative_recency
        )
        SELECT *
        FROM ranked
        ORDER BY
            final_score DESC,
            confidence DESC NULLS LAST,
            importance DESC NULLS LAST,
            last_confirmed DESC NULLS LAST,
            id
        LIMIT %s
    """

    # All positional params in order:
    #   q CTE:        query, query
    #   scored CTE:   vec, status, sensitivities, ...extra_filter_params...
    #   recency CTE:  recency_weight, recency_weight, half_life   (T2)
    #   final:        limit
    params = (
        [query, query, vec, status, ["public", "internal"]]
        + extra_params
        + [RECENCY_WEIGHT, RECENCY_WEIGHT, RECENCY_HALF_LIFE_DAYS]
        + [limit]
    )

    with POOL.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            cols = [d[0] for d in cur.description]
            rows = []
            for rec in cur.fetchall():
                row = dict(zip(cols, rec))
                row["tags"] = row.get("tags") or []
                row["candidate_reasons"] = row.get("candidate_reasons") or []
                rows.append(row)

            # Entity+property conflict resolution: when multiple items
            # share the same entity+property, keep only the most recent.
            rows = _deduplicate_entity_property(rows)

            # T1: surface agent-facing temporal metadata (additive enrichment).
            from temporal import enrich_rows
            enrich_rows(rows)

            return rows


def _search_memory_items_filtered_fts_only(
    query: str,
    filters: MetadataFilter | None = None,
    status: str = "active",
    limit: int = 50,
) -> list[dict[str, Any]]:
    """FTS-only fallback when embedding is unavailable."""
    from memory_db import POOL

    extra_sql, extra_params = _build_metadata_sql(filters)

    sql = f"""
        SELECT
            id, text, memory_type, scope, entity, property, value, status,
            confidence, importance, freshness_class,
            source_agent, source_session, source_chunk,
            first_seen, last_confirmed, supersedes,
            tags, notes,
            sensitivity,
            ts_rank_cd(
                to_tsvector(
                    'english',
                    COALESCE(text, '') || ' ' ||
                    COALESCE(entity, '') || ' ' ||
                    COALESCE(value, '')
                ),
                websearch_to_tsquery('english', %s)
            ) AS fts_rank
        FROM memory_items
        WHERE status = %s
          AND to_tsvector(
                'english',
                COALESCE(text, '') || ' ' ||
                COALESCE(entity, '') || ' ' ||
                COALESCE(value, '')
              ) @@ websearch_to_tsquery('english', %s)
          {extra_sql}
        ORDER BY fts_rank DESC, last_confirmed DESC NULLS LAST
        LIMIT %s
    """

    params = [query, status, query] + extra_params + [limit]

    with POOL.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            cols = [d[0] for d in cur.description]
            rows = []
            for rec in cur.fetchall():
                row = dict(zip(cols, rec))
                row["tags"] = row.get("tags") or []
                rows.append(row)
            # T1: surface temporal metadata on the FTS-only fallback path too.
            from temporal import enrich_rows
            enrich_rows(rows)
            return rows


async def query_db(app: FastAPI, query: str, query_vec: list[float], limit: int, filters: MetadataFilter | None = None) -> list[dict[str, Any]]:
    if filters is not None:
        return await run_in_executor(
            app.state.io_executor,
            search_memory_items_filtered,
            query=query,
            filters=filters,
            status="active",
            limit=limit,
        )
    return await run_in_executor(
        app.state.io_executor,
        search_memory_items_by_terms,
        query=query,
        status="active",
        limit=limit,
    )

def _filter_epoch_bounds(filters: "MetadataFilter | None") -> tuple[int | None, int | None]:
    """Derive (since_epoch, until_epoch) from a filter's since/until/as_of.

    as_of is an upper bound (point-in-time), folded into until via min(). Uses
    temporal.parse_ts so the Qdrant push-down agrees with the SQL window. Returns
    (None, None) when no temporal filter is set.
    """
    if filters is None:
        return None, None
    from temporal import parse_ts

    def _ep(v):
        if v is None:
            return None
        dt = parse_ts(v)
        return int(dt.timestamp()) if dt is not None else None

    since = _ep(getattr(filters, "since", None))
    until = _ep(getattr(filters, "until", None))
    as_of = _ep(getattr(filters, "as_of", None))
    # as_of caps the upper bound.
    uppers = [u for u in (until, as_of) if u is not None]
    until_bound = min(uppers) if uppers else None
    return since, until_bound


async def query_qdrant(
    app: FastAPI, query: str, limit: int, filters: "MetadataFilter | None" = None
) -> list[dict[str, Any]]:
    since_epoch, until_epoch = _filter_epoch_bounds(filters)
    return await run_in_executor(
        app.state.io_executor,
        search_qdrant_text,
        query=query,
        limit=limit,
        since_epoch=since_epoch,
        until_epoch=until_epoch,
    )


async def query_graph(app: FastAPI, query: str, limit: int) -> list[dict[str, Any]]:
    return await run_in_executor(app.state.io_executor, search_graph_text, query=query, limit=limit)


def rerank_results(reranker: CrossEncoder, query: str, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        return rows

    capped = rows[:RERANK_CAP]
    pairs = [(query, truncate_for_rerank(str(r.get("text", "")))) for r in capped]
    scores = reranker.predict(pairs)

    # Guard against silent score↔row misalignment. In normal operation the
    # reranker returns exactly one score per pair; a count mismatch would
    # otherwise (with a plain zip) silently truncate/mis-pair scores and
    # corrupt ranking. This is the live retrieval hot path, so we DEGRADE
    # safely instead of raising: log loudly and return the rows in their
    # pre-rerank order rather than crashing the query or returning corrupt
    # rankings. `scores` may not be sized directly (e.g. numpy array), so
    # normalize to a list first.
    scores = list(scores)
    if len(scores) != len(capped):
        LOGGER.error(
            "rerank score/row count mismatch (rows=%d scores=%d); "
            "skipping rerank and returning pre-rerank order",
            len(capped),
            len(scores),
        )
        return rows

    rescored = []
    for row, score in zip(capped, scores, strict=True):
        item = dict(row)
        item["rerank_score"] = float(score)
        rescored.append(item)

    rescored.sort(key=lambda x: x.get("rerank_score", -1e9), reverse=True)

    if len(rows) > RERANK_CAP:
        rescored.extend(rows[RERANK_CAP:])

    return rescored


async def _load_models_background(app: FastAPI, device: str) -> None:
    """Load the embed + rerank models off the event loop AFTER the server has
    bound its port. Populates app.state.embed_model / reranker and flips
    app.state.models_ready when both are present. On failure the service stays
    up and /health reports models_loaded=False so callers can degrade cleanly
    instead of hanging on a never-bound port.
    """
    try:
        embed_model = await run_in_executor(app.state.model_executor, load_embed_model, device)
        reranker = await run_in_executor(app.state.model_executor, load_reranker, device)
        app.state.embed_model = embed_model
        app.state.reranker = reranker
        app.state.models_ready = True
        app.state.model_load_error = None
        LOGGER.info("search_service_models_ready device=%s", device)
    except Exception as e:  # pragma: no cover - exercised operationally
        app.state.models_ready = False
        app.state.model_load_error = repr(e)
        LOGGER.exception("search_service_model_load_failed device=%s", device)


@asynccontextmanager
async def lifespan(app: FastAPI):
    device = detect_device()

    # Separate pools:
    # - model_executor caps heavy CPU/GPU inference concurrency
    # - io_executor handles lighter blocking DB/search calls
    app.state.model_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="search-model")
    app.state.io_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="search-io")

    LOGGER.info(
        "search_service_starting embed_model=%s rerank_model=%s device=%s backend=%s",
        MODEL_NAME,
        RERANK_MODEL,
        device,
        EMBED_BACKEND,
    )

    app.state.device = device
    # Initialize model handles to a known-empty state BEFORE yielding so the
    # ASGI server binds its port immediately and /health is answerable while
    # the heavy models load in the background. Previously these were awaited
    # before yield, so uvicorn never bound until model load finished (and a
    # stall during load left the port unbound / service effectively down).
    app.state.embed_model = None
    app.state.reranker = None
    app.state.models_ready = False
    app.state.model_load_error = None
    app.state.circuits = {
        "qdrant": SourceCircuit("qdrant", CIRCUIT_BREAKER_SECONDS),
        "graph": SourceCircuit("graph", CIRCUIT_BREAKER_SECONDS),
    }

    load_task = asyncio.create_task(_load_models_background(app, device))

    try:
        yield
    finally:
        try:
            load_task.cancel()
            # Await the cancelled task so the loop can settle it cleanly and we
            # avoid a "Task was destroyed but it is pending" warning on a race.
            await asyncio.gather(load_task, return_exceptions=True)
        except Exception as e:
            LOGGER.warning("search_service_model_load_cancel_warning error=%r", e)

        try:
            app.state.model_executor.shutdown(wait=False, cancel_futures=True)
        except Exception as e:
            LOGGER.warning("search_service_model_executor_shutdown_warning error=%r", e)

        try:
            app.state.io_executor.shutdown(wait=False, cancel_futures=True)
        except Exception as e:
            LOGGER.warning("search_service_io_executor_shutdown_warning error=%r", e)

        try:
            close_pool()
        except Exception as e:
            LOGGER.warning("search_service_pool_close_warning error=%r", e)


app = FastAPI(
    title="OpenClaw Search Service",
    lifespan=lifespan,
    docs_url="/docs" if DOCS_ENABLED else None,
    redoc_url="/redoc" if DOCS_ENABLED else None,
    openapi_url="/openapi.json" if DOCS_ENABLED else None,
)


def build_meta(top_k: int, project_id: str | None) -> dict[str, Any]:
    return {
        "runtime": "search_runtime",
        "top_k": top_k,
        "project_id": project_id,
    }


def build_search_work_item_metadata(project_id: str | None, subproject_id: str | None = None) -> WorkItemMemoryMetadata:
    return WorkItemMemoryMetadata(
        project_id=project_id or "global",
        subproject_id=subproject_id,
        inherited_memory_refs=[],
        promoted_memory_refs=[],
        local_memory_refs=[],
    )


async def run_hot_search(
    app: FastAPI,
    query: str,
    top_k: int,
    filters: MetadataFilter | None = None,
    rerank: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    embed_model: SentenceTransformer = app.state.embed_model
    reranker: CrossEncoder = app.state.reranker
    circuits: dict[str, SourceCircuit] = app.state.circuits

    # Models load in the background after the port binds. If a query arrives
    # before they are ready, fail fast with a retriable 503 instead of raising
    # AttributeError on a None model (which would surface as an opaque 500).
    if embed_model is None or reranker is None:
        raise HTTPException(
            status_code=503,
            detail="search_service_models_loading",
        )

    started = time.perf_counter()

    embed_started = time.perf_counter()
    query_vec = await run_in_executor(
        app.state.model_executor,
        embed_model.encode,
        query,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    if hasattr(query_vec, "tolist"):
        query_vec = query_vec.tolist()
    embed_ms = round((time.perf_counter() - embed_started) * 1000, 2)

    db_limit = top_k * 3
    source_meta: dict[str, Any] = {}

    db_task = query_db(app, query, query_vec, db_limit, filters=filters)

    if circuits["qdrant"].is_open():
        qdrant_task = None
        source_meta["qdrant"] = {"skipped": True, "reason": "circuit_open", **circuits["qdrant"].snapshot()}
    else:
        qdrant_task = query_qdrant(app, query, top_k * 2, filters=filters)

    if circuits["graph"].is_open():
        graph_task = None
        source_meta["graph"] = {"skipped": True, "reason": "circuit_open", **circuits["graph"].snapshot()}
    else:
        graph_task = query_graph(app, query, top_k * 2)

    tasks = [db_task]
    task_names = ["db"]
    if qdrant_task is not None:
        tasks.append(qdrant_task)
        task_names.append("qdrant")
    if graph_task is not None:
        tasks.append(graph_task)
        task_names.append("graph")

    source_started = time.perf_counter()
    results = await asyncio.gather(*tasks, return_exceptions=True)
    source_ms = round((time.perf_counter() - source_started) * 1000, 2)

    merged: list[dict[str, Any]] = []
    seen: set[str] = set()

    for name, res in zip(task_names, results):
        if isinstance(res, Exception):
            LOGGER.warning("search_source_failed source=%s query=%r error=%r", name, query, res)
            if name in circuits and is_infra_error(res):
                circuits[name].trip(res)
                source_meta[name] = {"ok": False, "tripped": True, **circuits[name].snapshot()}
            else:
                source_meta[name] = {"ok": False, "tripped": False, "error": repr(res)}
            continue

        source_rows = res
        # Apply metadata filters to Qdrant/graph results (DB filters are pushed into SQL)
        if filters is not None and name in ("qdrant", "graph"):
            source_rows = _apply_metadata_filter(source_rows, filters)

        source_meta[name] = {"ok": True, "count": len(source_rows), "pre_filter": len(res)}

        for row in source_rows:
            item = dict(row)
            item.setdefault("source_type", name)
            fp = result_fingerprint(item)
            if fp in seen:
                continue
            seen.add(fp)
            merged.append(item)

    # T1+T2: enrich ALL merged results (db + qdrant + graph) with temporal
    # metadata at the single choke point, before rerank/return. Centralizing
    # here guarantees uniform coverage regardless of source path. Pass the same
    # recency knobs the SQL ranking uses so surfaced recency_bonus agrees with
    # the score contribution.
    from temporal import enrich_rows
    enrich_rows(merged, weight=RECENCY_WEIGHT, half_life_days=RECENCY_HALF_LIFE_DAYS)

    rerank_ms = 0.0
    if rerank and merged:
        rerank_started = time.perf_counter()
        reranked = await run_in_executor(app.state.model_executor, rerank_results, reranker, query, merged)
        rerank_ms = round((time.perf_counter() - rerank_started) * 1000, 2)
    else:
        reranked = merged

    total_ms = round((time.perf_counter() - started) * 1000, 2)
    top_rows = reranked[:top_k]

    filter_summary = None
    if filters is not None:
        filter_summary = {k: v for k, v in filters.model_dump().items() if v is not None}

    # T4: temporal cohort view over the RETURNED results (additive, read-only —
    # does not reorder top_rows). Gives the agent a temporal map of the result
    # set: how many are today/this_week/this_month/older/undated, with
    # newest/oldest labels and indexes into `results`.
    # T5: temporal_signals = compact observability profile (age/recency/confidence)
    # for the result set, surfaced in meta AND logged. Both are best-effort and
    # exception-isolated so observability can never break search.
    cohorts = None
    tsignals = None
    try:
        from temporal import temporal_cohorts, temporal_signals
        cohorts = temporal_cohorts(top_rows)
        tsignals = temporal_signals(top_rows)
    except Exception:
        LOGGER.exception("temporal_observability_failed")

    LOGGER.info(
        "search_completed query=%r total_ms=%s embed_ms=%s source_ms=%s rerank_ms=%s merged=%s results=%s filters=%s rerank=%s source_meta=%s temporal_signals=%s cohorts=%s",
        query,
        total_ms,
        embed_ms,
        source_ms,
        rerank_ms,
        len(merged),
        len(top_rows),
        filter_summary,
        rerank,
        source_meta,
        tsignals,
        (cohorts or {}).get("counts") if cohorts else None,
    )

    meta = {
        "total_ms": total_ms,
        "embed_ms": embed_ms,
        "source_ms": source_ms,
        "rerank_ms": rerank_ms,
        "rerank_applied": rerank and bool(merged),
        "device": app.state.device,
        "source_meta": source_meta,
        "rerank_cap": RERANK_CAP,
        "rerank_text_char_cap": RERANK_TEXT_CHAR_CAP,
        "filters_applied": filter_summary,
        "temporal_cohorts": cohorts,
        "temporal_signals": tsignals,
    }
    return top_rows, meta


@app.get("/health")
async def health() -> dict[str, Any]:
    models_loaded = bool(
        getattr(app.state, "embed_model", None) is not None
        and getattr(app.state, "reranker", None) is not None
    )
    return {
        # ok=True means the process is up and the port is bound. Use
        # models_loaded to gate readiness for /search traffic.
        "ok": True,
        "ready": models_loaded,
        "embed_model": MODEL_NAME,
        "rerank_model": RERANK_MODEL,
        "device": app.state.device,
        "models_loaded": models_loaded,
        "model_load_error": getattr(app.state, "model_load_error", None),
        "circuits": {
            name: circuit.snapshot()
            for name, circuit in app.state.circuits.items()
        },
    }


@app.get("/search", response_model=SearchResponse)
async def search_get(
    query: Annotated[str, Query(min_length=1)],
    top_k: Annotated[int, Query(ge=1, le=50)] = TOP_K,
    entity: Annotated[str | None, Query(description="Filter by entity")] = None,
    property: Annotated[str | None, Query(description="Filter by property", alias="property")] = None,
    session_id: Annotated[str | None, Query(description="Filter by source_session")] = None,
    source_agent: Annotated[str | None, Query(description="Filter by source_agent")] = None,
    memory_type: Annotated[str | None, Query(description="Filter by memory_type")] = None,
    scope: Annotated[str | None, Query(description="Filter by scope")] = None,
    project_id: Annotated[str | None, Query(description="Filter by project_id")] = None,
    min_confidence: Annotated[float | None, Query(ge=0.0, le=1.0, description="Minimum confidence")] = None,
    since: Annotated[str | None, Query(description="Only effective ts >= this (ISO-8601 or YYYY-MM-DD)")] = None,
    until: Annotated[str | None, Query(description="Only effective ts <= this (ISO-8601 or YYYY-MM-DD)")] = None,
    as_of: Annotated[str | None, Query(description="Point-in-time view: exclude memories newer than this")] = None,
    rerank: Annotated[bool, Query(description="Apply cross-encoder reranking")] = True,
):
    filters = None
    filter_kwargs = {
        "entity": entity, "property": property, "session_id": session_id,
        "source_agent": source_agent, "memory_type": memory_type, "scope": scope,
        "project_id": project_id, "min_confidence": min_confidence,
        "since": since, "until": until, "as_of": as_of,
    }
    if any(v is not None for v in filter_kwargs.values()):
        filters = MetadataFilter(**filter_kwargs)

    try:
        rows, meta = await run_hot_search(app, query, top_k, filters=filters, rerank=rerank)
        return SearchResponse(ok=True, query=query, results=[SearchResult(**r) for r in rows], meta=meta)
    except HTTPException:
        # Preserve intentional HTTP status (e.g. 503 models_loading) — do not
        # collapse a deliberate, retriable signal into an opaque 500.
        raise
    except Exception:
        LOGGER.exception("search_request_failed query=%r", query)
        raise HTTPException(status_code=500, detail="search_service_internal_error")


@app.post("/search", response_model=SearchResponse)
async def search_post(req: SearchRequest):
    try:
        rows, meta = await run_hot_search(app, req.query, req.top_k, filters=req.filters, rerank=req.rerank)
        return SearchResponse(ok=True, query=req.query, results=[SearchResult(**r) for r in rows], meta=meta)
    except HTTPException:
        raise
    except Exception:
        LOGGER.exception("search_request_failed query=%r", req.query)
        raise HTTPException(status_code=500, detail="search_service_internal_error")


@app.post("/orchestrator/context")
async def orchestrator_context(req: OrchestratorContextRequest):
    try:
        project_id = req.project_id or "global"
        raw = fetch_orchestrator_context(
            project_id=project_id,
            subproject_id=req.subproject_id,
            role=req.role,
            mode=req.mode,
            work_id=req.work_id,
            kind=req.kind,
            tags=req.tags,
            limit=req.limit,
        )

        work_item = build_search_work_item_metadata(project_id, req.subproject_id)
        selected_refs = []
        for ref in raw.get("memory_refs", []):
            ref_dict = dict(ref)
            ref_dict["category"] = ref_dict.get("memory_type")
            ref_dict["content"] = ref_dict.get("content")
            for tag in ref_dict.get("tags", []) or []:
                if isinstance(tag, str) and tag.startswith("target_role:"):
                    ref_dict["target_role"] = tag.split(":", 1)[1]
                    break
            selected_refs.append(ref_dict)

        pack = build_memory_pack(
            role=req.role,
            work_item=work_item,
            parent_refs=selected_refs,
            subproject_refs=[],
            project_refs=[],
        )

        return {
            "ok": True,
            "context": raw,
            "memory_pack": pack.model_dump(),
            "meta": build_meta(req.limit, project_id),
        }
    except HTTPException:
        raise
    except Exception:
        LOGGER.exception("orchestrator_context_failed work_id=%r", req.work_id)
        raise HTTPException(status_code=500, detail="orchestrator_context_internal_error")


@app.post("/normal-agent/packet")
async def normal_agent_packet(req: NormalAgentRequest):
    try:
        rows, meta = await run_hot_search(app, req.query, req.max_context_items)
        response = build_normal_agent_packet_response(req, rows, retrieval_meta=meta)
        return response.model_dump()
    except HTTPException:
        # Pass through 503 models_loading (and any deliberate status) so callers
        # get a retriable signal rather than an opaque 500.
        raise
    except Exception:
        LOGGER.exception("normal_agent_packet_failed project_id=%r query=%r", req.project_id, req.query)
        raise HTTPException(status_code=500, detail="normal_agent_packet_internal_error")
