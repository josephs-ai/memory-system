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


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=TOP_K, ge=1, le=50)


class SearchResult(BaseModel):
    text: str
    path: str = ""
    source_type: str = ""
    score: float | None = None
    rerank_score: float | None = None


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


def load_embed_model(device: str) -> SentenceTransformer:
    if EMBED_BACKEND == "onnx" and device == "cpu":
        try:
            return SentenceTransformer(MODEL_NAME, device=device, backend="onnx")
        except Exception as e:
            LOGGER.warning("embed_model_onnx_fallback error=%r", e)
    return SentenceTransformer(MODEL_NAME, device=device)


def load_reranker(device: str) -> CrossEncoder:
    try:
        return CrossEncoder(RERANK_MODEL, device=device)
    except TypeError:
        return CrossEncoder(RERANK_MODEL)


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


async def query_db(app: FastAPI, query: str, query_vec: list[float], limit: int) -> list[dict[str, Any]]:
    return await run_in_executor(
        app.state.io_executor,
        search_memory_items_by_terms,
        query=query,
        status="active",
        limit=limit,
    )

async def query_qdrant(app: FastAPI, query: str, limit: int) -> list[dict[str, Any]]:
    return await run_in_executor(app.state.io_executor, search_qdrant_text, query=query, limit=limit)


async def query_graph(app: FastAPI, query: str, limit: int) -> list[dict[str, Any]]:
    return await run_in_executor(app.state.io_executor, search_graph_text, query=query, limit=limit)


def rerank_results(reranker: CrossEncoder, query: str, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        return rows

    capped = rows[:RERANK_CAP]
    pairs = [(query, truncate_for_rerank(str(r.get("text", "")))) for r in capped]
    scores = reranker.predict(pairs)

    rescored = []
    for row, score in zip(capped, scores):
        item = dict(row)
        item["rerank_score"] = float(score)
        rescored.append(item)

    rescored.sort(key=lambda x: x.get("rerank_score", -1e9), reverse=True)

    if len(rows) > RERANK_CAP:
        rescored.extend(rows[RERANK_CAP:])

    return rescored


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
    app.state.embed_model = await run_in_executor(app.state.model_executor, load_embed_model, device)
    app.state.reranker = await run_in_executor(app.state.model_executor, load_reranker, device)
    app.state.circuits = {
        "qdrant": SourceCircuit("qdrant", CIRCUIT_BREAKER_SECONDS),
        "graph": SourceCircuit("graph", CIRCUIT_BREAKER_SECONDS),
    }

    try:
        yield
    finally:
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


async def run_hot_search(app: FastAPI, query: str, top_k: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    embed_model: SentenceTransformer = app.state.embed_model
    reranker: CrossEncoder = app.state.reranker
    circuits: dict[str, SourceCircuit] = app.state.circuits

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

    db_task = query_db(app, query, query_vec, db_limit)

    if circuits["qdrant"].is_open():
        qdrant_task = None
        source_meta["qdrant"] = {"skipped": True, "reason": "circuit_open", **circuits["qdrant"].snapshot()}
    else:
        qdrant_task = query_qdrant(app, query, top_k * 2)

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

        source_meta[name] = {"ok": True, "count": len(res)}

        for row in res:
            item = dict(row)
            item.setdefault("source_type", name)
            fp = result_fingerprint(item)
            if fp in seen:
                continue
            seen.add(fp)
            merged.append(item)

    rerank_started = time.perf_counter()
    reranked = await run_in_executor(app.state.model_executor, rerank_results, reranker, query, merged)
    rerank_ms = round((time.perf_counter() - rerank_started) * 1000, 2)

    total_ms = round((time.perf_counter() - started) * 1000, 2)
    top_rows = reranked[:top_k]

    LOGGER.info(
        "search_completed query=%r total_ms=%s embed_ms=%s source_ms=%s rerank_ms=%s merged=%s results=%s source_meta=%s",
        query,
        total_ms,
        embed_ms,
        source_ms,
        rerank_ms,
        len(merged),
        len(top_rows),
        source_meta,
    )

    meta = {
        "total_ms": total_ms,
        "embed_ms": embed_ms,
        "source_ms": source_ms,
        "rerank_ms": rerank_ms,
        "device": app.state.device,
        "source_meta": source_meta,
        "rerank_cap": RERANK_CAP,
        "rerank_text_char_cap": RERANK_TEXT_CHAR_CAP,
    }
    return top_rows, meta


@app.get("/health")
async def health() -> dict[str, Any]:
    return {
        "ok": True,
        "embed_model": MODEL_NAME,
        "rerank_model": RERANK_MODEL,
        "device": app.state.device,
        "models_loaded": bool(app.state.embed_model and app.state.reranker),
        "circuits": {
            name: circuit.snapshot()
            for name, circuit in app.state.circuits.items()
        },
    }


@app.get("/search", response_model=SearchResponse)
async def search_get(
    query: Annotated[str, Query(min_length=1)],
    top_k: Annotated[int, Query(ge=1, le=50)] = TOP_K,
):
    try:
        rows, meta = await run_hot_search(app, query, top_k)
        return SearchResponse(ok=True, query=query, results=[SearchResult(**r) for r in rows], meta=meta)
    except Exception:
        LOGGER.exception("search_request_failed query=%r", query)
        raise HTTPException(status_code=500, detail="search_service_internal_error")


@app.post("/search", response_model=SearchResponse)
async def search_post(req: SearchRequest):
    try:
        rows, meta = await run_hot_search(app, req.query, req.top_k)
        return SearchResponse(ok=True, query=req.query, results=[SearchResult(**r) for r in rows], meta=meta)
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
    except Exception:
        LOGGER.exception("orchestrator_context_failed work_id=%r", req.work_id)
        raise HTTPException(status_code=500, detail="orchestrator_context_internal_error")


@app.post("/normal-agent/packet")
async def normal_agent_packet(req: NormalAgentRequest):
    try:
        rows, meta = await run_hot_search(app, req.query, req.max_context_items)
        response = build_normal_agent_packet_response(req, rows, retrieval_meta=meta)
        return response.model_dump()
    except Exception:
        LOGGER.exception("normal_agent_packet_failed project_id=%r query=%r", req.project_id, req.query)
        raise HTTPException(status_code=500, detail="normal_agent_packet_internal_error")
