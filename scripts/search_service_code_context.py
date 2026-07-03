"""
search_service_code_context.py — FastAPI integration for the Context Hydrator.

Slice 7 (S7) of the Zero-Reread Architecture.

Provides a `/context` endpoint that returns unified code + memory + graph
context packs. Designed as a FastAPI router that can be mounted into the
existing search_memory_service.py application.

This is an additive module — it does NOT modify the existing search service
code. It provides a router that the service can optionally mount.

Backward compatibility: all existing endpoints (/search, /orchestrator/context,
/normal-agent/packet, /health) remain completely unchanged.

Usage:
    # In search_memory_service.py or standalone:
    from search_service_code_context import code_context_router
    app.include_router(code_context_router)

Standalone test server:
    python search_service_code_context.py [--port 8100]
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any

from fastapi import APIRouter, FastAPI, HTTPException
from pydantic import BaseModel, Field

LOGGER = logging.getLogger("openclaw.search_service_code_context")

# Ensure script directory is importable
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))


# ---------------------------------------------------------------------------
# Request/Response models
# ---------------------------------------------------------------------------

class ContextRequest(BaseModel):
    """Request model for the /context endpoint."""
    query: str = Field(min_length=1, description="Natural language or code query")
    code_limit: int = Field(default=8, ge=1, le=30, description="Max code chunk results")
    memory_limit: int = Field(default=5, ge=0, le=20, description="Max memory item results")
    primary_count: int = Field(default=3, ge=1, le=10, description="Number of hits with full body")
    include_graph: bool = Field(default=True, description="Include Neo4j dependency traversal")
    rerank: bool = Field(default=True, description="Apply cross-encoder reranking for improved relevance")


class CodeContextItem(BaseModel):
    """A single code context entry in the response."""
    type: str                           # "code" (primary) or "code_ref" (secondary)
    qualified_name: str
    chunk_type: str
    file: str
    lines: str
    score: float
    signature: str | None = None
    docstring: str | None = None
    body: str | None = None
    body_truncated: bool | None = None
    body_note: str | None = None
    rerank_score: float | None = None


class MemoryContextItem(BaseModel):
    """A single memory context entry in the response."""
    type: str = "memory"
    memory_type: str | None = None
    text: str
    entity: str | None = None
    confidence: float | None = None
    score: float
    rerank_score: float | None = None


class ContextResponse(BaseModel):
    """Response model for the /context endpoint."""
    ok: bool
    query: str
    code_context: list[CodeContextItem]
    memory_context: list[MemoryContextItem]
    dependency_map: dict[str, Any]
    meta: dict[str, Any]


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

code_context_router = APIRouter(tags=["code-context"])


@code_context_router.get("/context/health")
async def context_health() -> dict[str, Any]:
    """Health check for the code context subsystem."""
    from context_hydrator import CODE_COLLECTION, MEMORY_COLLECTION

    health: dict[str, Any] = {"ok": True, "subsystem": "code_context"}

    try:
        from qdrant_client import QdrantClient
        client = QdrantClient(
            host=os.environ.get("OPENCLAW_QDRANT_HOST", "localhost"),
            port=int(os.environ.get("OPENCLAW_QDRANT_PORT", "6333")),
        )
        code_info = client.get_collection(CODE_COLLECTION)
        memory_info = client.get_collection(MEMORY_COLLECTION)
        health["qdrant"] = {
            "code_chunks": {
                "status": code_info.status.value,
                "points": code_info.points_count,
            },
            "memory_items": {
                "status": memory_info.status.value,
                "points": memory_info.points_count,
            },
        }
    except Exception as e:
        health["qdrant"] = {"error": str(e)}
        health["ok"] = False

    try:
        import psycopg
        dsn = (os.environ.get("OPENCLAW_MEMORY_DSN") or os.environ.get("OPENCLAW_MEMORY_DB_DSN") or "dbname=openclaw_memory")  # honor both canonical DSN env names
        conn = psycopg.connect(dsn)
        with conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM code_chunks")
            count = cur.fetchone()[0]
        conn.close()
        health["postgresql"] = {"code_chunks": count}
    except Exception as e:
        health["postgresql"] = {"error": str(e)}
        health["ok"] = False

    return health


@code_context_router.post("/context", response_model=ContextResponse)
async def context_search(req: ContextRequest) -> ContextResponse:
    """
    Unified code + memory + graph context retrieval.

    Returns a token-optimized context pack with:
    - code_context: relevant code chunks (primary with full body, secondary with signatures)
    - memory_context: relevant memory items (decisions, patterns, bugs)
    - dependency_map: callers, callees, imports, dependents from Neo4j
    """
    try:
        from context_hydrator import build_context_pack

        pack = build_context_pack(
            req.query,
            code_limit=req.code_limit,
            memory_limit=req.memory_limit,
            primary_count=req.primary_count,
            include_graph=req.include_graph,
            rerank=req.rerank,
        )

        return ContextResponse(
            ok=True,
            query=req.query,
            code_context=[CodeContextItem(**c) for c in pack["code_context"]],
            memory_context=[MemoryContextItem(**m) for m in pack["memory_context"]],
            dependency_map=pack["dependency_map"],
            meta=pack["meta"],
        )

    except Exception:
        LOGGER.exception("context_search_failed query=%r", req.query)
        raise HTTPException(
            status_code=500,
            detail="context_search_internal_error",
        )


@code_context_router.get("/context")
async def context_search_get(
    query: str,
    code_limit: int = 8,
    memory_limit: int = 5,
    primary_count: int = 3,
    include_graph: bool = True,
    rerank: bool = True,
) -> ContextResponse:
    """GET variant for simple browser/curl testing."""
    req = ContextRequest(
        query=query,
        code_limit=code_limit,
        memory_limit=memory_limit,
        primary_count=primary_count,
        include_graph=include_graph,
        rerank=rerank,
    )
    return await context_search(req)


# ---------------------------------------------------------------------------
# Standalone test server
# ---------------------------------------------------------------------------

def create_standalone_app() -> FastAPI:
    """Create a standalone FastAPI app for testing the context endpoint."""
    app = FastAPI(
        title="OpenClaw Code Context Service",
        description="Zero-reread code + memory + graph context retrieval",
        version="1.0.0",
    )
    app.include_router(code_context_router)
    return app


if __name__ == "__main__":
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="Standalone code context server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8100)
    args = parser.parse_args()

    app = create_standalone_app()
    uvicorn.run(app, host=args.host, port=args.port)
