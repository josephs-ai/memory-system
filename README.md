# 🧠 OpenClaw Memory System

**Enterprise-grade long-term memory for AI agents.** Hybrid retrieval, self-improving feedback, progressive summarization, temporal reasoning, and semantic knowledge graphs — in one system.

[![CI](https://github.com/josephs-ai/Memory-System-claw/actions/workflows/ci.yml/badge.svg)](https://github.com/josephs-ai/Memory-System-claw/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

---

## Why This Exists

Most AI memory systems store flat key-value pairs and call it done. This one doesn't.

OpenClaw Memory turns raw agent conversations into **structured, queryable, self-improving long-term memory** — with 12 memory types, lifecycle management, hybrid retrieval (full-text + vector + cross-encoder reranking), and a feedback loop that makes retrieval better over time without any manual tuning.

### What makes it different

| Feature | mem0 | Letta/MemGPT | Zep | **This** |
|---------|------|-------------|-----|----------|
| Hybrid retrieval (FTS + vector + rerank) | ❌ | ❌ | Partial | ✅ |
| Self-improving feedback loop | ❌ | ❌ | ❌ | ✅ |
| Built-in IR benchmarks (precision/recall/MRR/nDCG) | ❌ | ❌ | ❌ | ✅ |
| Progressive time-based summarization | ❌ | ❌ | ❌ | ✅ |
| Temporal reasoning in retrieval | ❌ | ❌ | ❌ | ✅ |
| Memory knowledge graph | ❌ | ❌ | Partial | ✅ |
| Structured memory types (12 types) | ❌ | ❌ | ❌ | ✅ |
| Memory lifecycle management | ❌ | Partial | ❌ | ✅ |
| Code-aware context (AST + graph) | ❌ | ❌ | ❌ | ✅ |

---

## Quickstart

### 1. Start infrastructure

```bash
git clone https://github.com/josephs-ai/Memory-System-claw.git
cd Memory-System-claw
docker compose up -d
```

This starts PostgreSQL, Qdrant, and Neo4j.

### 2. Install

```bash
pip install -e ".[all]"
```

### 3. Configure

```bash
export OPENCLAW_MEMORY_DSN="host=localhost dbname=openclaw_memory user=openclaw password=openclaw"
```

### 4. Initialize the database

```bash
psql "$OPENCLAW_MEMORY_DSN" -f scripts/memory_db_schema.sql
```

### 5. Start the search service

```bash
python scripts/search_memory_service.py --port 8791
```

### 6. Query memory

```bash
curl "http://localhost:8791/search?q=how+does+the+retrieval+pipeline+work&limit=5"
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Agent Sessions                           │
└──────────────┬──────────────────────────────────────────────────┘
               │
               ▼
┌──────────────────────────┐
│   Transcript Dehydration  │  Strip noise, extract signal
└──────────┬───────────────┘
           ▼
┌──────────────────────────┐
│   Chunk + Extract Updates │  Split into candidate memory items
└──────────┬───────────────┘
           ▼
┌──────────────────────────┐
│   Judge + Structure       │  Classify: type, scope, entity, confidence
└──────────┬───────────────┘
           ▼
┌──────────────────────────┐
│   Route + Deduplicate     │  Merge with existing, handle supersedes
└──────────┬───────────────┘
           ▼
┌──────────────────────────────────────────────────────────┐
│                    Memory Store                           │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐               │
│  │PostgreSQL│  │  Qdrant  │  │  Neo4j   │               │
│  │ 18K+     │  │ Vectors  │  │ Graph    │               │
│  │ items    │  │ 384-dim  │  │ Links    │               │
│  └──────────┘  └──────────┘  └──────────┘               │
└──────────────────────┬───────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────┐
│              Hybrid Retrieval Pipeline                    │
│                                                          │
│  FTS (ts_rank) ──┐                                       │
│  Vector search ──┼──► Score fusion ──► Cross-encoder     │
│  Graph context ──┘    + feedback      reranking          │
│                       + temporal       ──► Final results │
└──────────────────────────────────────────────────────────┘
```

---

## Core Capabilities

### 🔄 Self-Improving Retrieval (P0)
Memory gets better the more you use it. Every retrieval action can be marked useful/bad, and feedback scores are blended into future rankings — no retraining needed.

```python
from scripts.mark_retrieval_useful import mark_useful
from scripts.mark_retrieval_bad import mark_bad

mark_useful(item_id="mem-abc123", query="how does auth work")
mark_bad(item_id="mem-xyz789", query="deployment steps")
```

### 📊 Evaluation Benchmarks (P1)
Built-in IR metrics to measure and prevent retrieval regression:

```bash
python scripts/retrieval_benchmark.py seed    # Create golden dataset
python scripts/retrieval_benchmark.py run     # Run benchmark
python scripts/retrieval_benchmark.py report  # Generate report
```

Tracks: Precision@K, Recall@K, MRR, nDCG. Alerts on >5% regression.

### 📝 Progressive Summarization (P2)
Old memories are automatically consolidated — daily summaries after 7 days, weekly after 30, monthly after 90. Originals are archived, not deleted.

### ⏰ Temporal Reasoning (P3)
Retrieval understands time. Recent items get a gentle boost; "what happened last week" queries match items from that time window instead of favoring recency.

### 🕸️ Knowledge Graph (P4)
Memories are linked by typed relationships — SUPERSEDES, CAUSED_BY, CONTRADICTS, SUPPORTS, DEPENDS_ON, RELATES_TO — enabling graph traversal for context expansion.

### ⚡ Streaming Ingestion (P5)
Event-driven architecture for real-time memory creation. Agent actions immediately produce queryable memories via an in-process pub/sub bus.

### 🖼️ Multi-Modal (P6)
Images, diagrams, screenshots, and structured data stored as description-indexed memory items with content hashing and modality metadata.

---

## Memory Types

| Type | Description |
|------|-------------|
| `decision` | Choices made and their rationale |
| `lesson_learned` | Insights from mistakes or successes |
| `fact` | Verified information |
| `rule` | Operational rules and constraints |
| `architecture_rule` | System design principles |
| `preference` | User/agent preferences |
| `observation` | Noted behaviors or patterns |
| `implementation_pattern` | Code patterns worth remembering |
| `learned_fix` | Bug fixes and workarounds |
| `bug_history` | Known issues and their resolution |
| `episodic` | Narrative events |
| `feature_summary` | Feature-level summaries |

---

## Memory Lifecycle

```
candidate → durable → [superseded | consolidated | discarded]
                 ↑
           re-confirmed
```

- **candidate**: Newly extracted, awaiting confirmation
- **durable**: Confirmed, actively retrievable
- **superseded**: Replaced by a newer item (linked via SUPERSEDES)
- **consolidated**: Rolled into a summary (P2)
- **discarded**: Low-quality, removed from active retrieval

---

## Project Structure

```
scripts/
├── search_memory.py              # Core hybrid search
├── search_memory_service.py      # FastAPI search API
├── context_hydrator.py           # Unified retrieval (code + memory + graph)
├── memory_db.py                  # PostgreSQL operations
├── embed_memory_items.py         # Qdrant vector indexing
├── feedback_score_engine.py      # P0: Self-improving retrieval
├── retrieval_benchmark.py        # P1: IR evaluation
├── consolidation_grouper.py      # P2: Time-based grouping
├── summary_generator.py          # P2: Summary creation
├── temporal_scoring.py           # P3: Time-aware scoring
├── memory_knowledge_graph.py     # P4: Semantic links
├── streaming_ingestion.py        # P5: Event-driven ingestion
├── multimodal_memory.py          # P6: Multi-modal support
├── parse_code_ast.py             # AST parser for code context
├── sync_code_graph.py            # Code → Neo4j graph
├── rerank_crossencoder.py        # Cross-encoder reranking
├── dehydrate_transcript.py       # Transcript → clean text
├── generate_memory_candidates.py # Extract candidate memories
├── memory_db_schema.sql          # Database schema
└── test_*.py                     # 466 tests
```

---

## Running Tests

```bash
# All tests (excluding integration tests requiring live DB)
python -m pytest scripts/ --ignore=scripts/test_hybrid_search.py -q

# Just the P0-P6 capability tests (no DB needed)
python -m pytest scripts/test_feedback_score_engine.py \
  scripts/test_consolidation_grouper.py \
  scripts/test_temporal_scoring.py \
  scripts/test_memory_knowledge_graph.py \
  scripts/test_streaming_ingestion.py \
  scripts/test_multimodal_memory.py -v
```

---

## Configuration

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `OPENCLAW_MEMORY_DSN` | `dbname=openclaw_memory` | PostgreSQL connection string |
| `QDRANT_HOST` | `localhost` | Qdrant server host |
| `QDRANT_PORT` | `6333` | Qdrant server port |
| `NEO4J_URI` | `bolt://localhost:7687` | Neo4j connection URI |
| `NEO4J_USER` | `neo4j` | Neo4j username |
| `NEO4J_PASSWORD` | `neo4jpassword` | Neo4j password |
| `OPENCLAW_RERANK_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Cross-encoder model |
| `OPENCLAW_RERANK_CAP` | `20` | Max items to cross-encoder rerank |

---

## License

MIT — see [LICENSE](LICENSE).
