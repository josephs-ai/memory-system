# API Reference

The OpenClaw Memory search service exposes a FastAPI-based REST API on port 8791 (default).

## Start the service

```bash
python scripts/search_memory_service.py --port 8791
```

## Endpoints

### `GET /health`

Health check — returns service status, loaded models, and circuit breaker states.

**Response:**
```json
{
  "ok": true,
  "embed_model": "sentence-transformers/all-MiniLM-L6-v2",
  "rerank_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
  "device": "cpu",
  "models_loaded": true,
  "circuits": {
    "db": "closed",
    "qdrant": "closed",
    "neo4j": "closed"
  }
}
```

---

### `GET /search`

Search memory items by natural language query.

**Parameters:**

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `query` | string (required) | — | Natural language search query |
| `top_k` | int | 10 | Number of results to return (1-50) |

**Example:**
```bash
curl "http://localhost:8791/search?query=how+does+authentication+work&top_k=5"
```

**Response:**
```json
{
  "ok": true,
  "query": "how does authentication work",
  "results": [
    {
      "text": "Authentication uses JWT tokens with 24h expiry...",
      "score": 0.847,
      "source_type": "canonical",
      "path": "db:memory_items",
      "item_id": "mem-abc123"
    }
  ],
  "meta": {
    "db_hits": 15,
    "qdrant_hits": 20,
    "graph_hits": 5,
    "reranked": true
  }
}
```

---

### `POST /search`

Same as GET but accepts JSON body.

**Request body:**
```json
{
  "query": "how does authentication work",
  "top_k": 5
}
```

---

### `POST /orchestrator/context`

Retrieve scoped context for the orchestrator system. Returns memory items filtered by project scope.

**Request body:**
```json
{
  "project_id": "my-project",
  "work_item_id": "WI-001",
  "query": "what design patterns are we using"
}
```

---

### `POST /normal-agent/packet`

Return a formatted memory context packet for standard agent consumption.

**Request body:**
```json
{
  "query": "what was decided about the database schema",
  "max_context_items": 10,
  "agent_id": "memory-coder",
  "scope": "global"
}
```

---

## Retrieval Pipeline

Each search request goes through this pipeline:

```
Query
  │
  ├─► PostgreSQL FTS (ts_rank_cd + keyword matching)
  ├─► Qdrant vector similarity (384-dim sentence-transformers)
  └─► Neo4j graph context (optional, via circuit breaker)
      │
      ▼
  Score fusion (weighted combination)
      │
      ▼
  Feedback boost (from retrieval_feedback table)
      │
      ▼
  Temporal scoring (recency decay / time-reference matching)
      │
      ▼
  Summary penalty (-0.05 for consolidated summaries)
      │
      ▼
  Cross-encoder reranking (ms-marco-MiniLM-L-6-v2)
      │
      ▼
  Top-K results
```

### Scoring Formula

```
base_score = (fts_rank × 3.0) + (vector_score × 1.1) + structured_bonus + importance_bonus
feedback_adjusted = base_score + (feedback_boost × 0.15)
temporal_adjusted = feedback_adjusted + temporal_boost
reranked_score = (pg_score × 0.35) + (cross_encoder_score × 1.25)
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENCLAW_MEMORY_DSN` | `dbname=openclaw_memory` | PostgreSQL DSN |
| `QDRANT_HOST` | `localhost` | Qdrant host |
| `QDRANT_PORT` | `6333` | Qdrant port |
| `NEO4J_URI` | `bolt://localhost:7687` | Neo4j URI |
| `NEO4J_USER` | `neo4j` | Neo4j user |
| `NEO4J_PASSWORD` | `neo4jpassword` | Neo4j password |
| `OPENCLAW_RERANK_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Reranking model |
| `OPENCLAW_RERANK_CAP` | `20` | Max items to rerank |
| `OPENCLAW_SEARCH_PORT` | `8791` | Service port |

## Feedback API

### Mark retrieval as useful

```bash
python scripts/mark_retrieval_useful.py --item-id mem-abc123 --query "auth flow"
```

### Mark retrieval as bad

```bash
python scripts/mark_retrieval_bad.py --item-id mem-xyz789 --query "deployment"
```

### View feedback stats

```bash
python scripts/feedback_score_engine.py stats
```

## Benchmark API

```bash
python scripts/retrieval_benchmark.py seed     # Create golden dataset from live DB
python scripts/retrieval_benchmark.py run      # Run IR benchmark
python scripts/retrieval_benchmark.py compare  # Compare against baseline
python scripts/retrieval_benchmark.py report   # Generate full report
```
