# 🧠 OpenClaw Memory System

**Long-term memory for AI agents that actually works.** Hybrid retrieval, temporal reasoning, contradiction resolution, self-improving feedback — no LLM in the retrieval loop.

[![CI](https://github.com/josephs-ai/Memory-System-claw/actions/workflows/ci.yml/badge.svg)](https://github.com/josephs-ai/Memory-System-claw/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

---

## Benchmarks

Evaluated across **9 retrieval benchmarks** — no LLM in the loop. Just PostgreSQL FTS, Qdrant vectors, and two small cross-encoders (23M + 86M params).

| Benchmark | Score | Trials | What it tests |
|---|---|---|---|
| **🪡 NIAH** | **1.000** Recall@1 | 40 needles / **110K items** | Find specific facts buried in massive stores |
| **💬 MSC** | **1.000** Recall | **300 queries** / 900 items / 100 personas | Multi-session persona memory |
| **🔗 HotpotQA** | **1.000** Support Recall | 10 multi-hop queries | 2-hop reasoning — both supporting docs found every time |
| **🎵 MuSiQue** | **1.000** Support Recall | 8 queries (2-4 hops) | Complex multi-hop: 2-hop, 3-hop, and 4-hop all perfect |
| **📝 CRUD-RAG** | **0.955** Accuracy | **21/22 correct** / 12 scenarios | Create, Read, Update, Delete lifecycle ops |
| **⚔️ Contradiction** | **0.900** Resolution | **9/10 correct** | Detect and resolve conflicting facts via temporal recency |
| **🔥 FEVER** | **1.000** Evidence Recall | 15 claims / ~75 items | Always finds the right evidence for fact verification |
| **⏰ TemporalQA** | **0.733** Accuracy | 15 queries | Time-aware retrieval (earliest=1.0, latest=0.625) |
| **🗣️ LoCoMo** | **0.328** F1 (When Qs) | 37 temporal questions | Date extraction from conversation memory, no LLM |

### The numbers that matter

```
 40/40    needles found at rank 1 across 110,000 items
 1.000    retrieval recall on 5 independent benchmarks
 0.900    contradiction resolution (temporal recency ranking)
 ~200ms   median retrieval latency
 0 LLMs   in the retrieval pipeline
```

### How we compare

| Capability | mem0 | Letta/MemGPT | Zep | **OpenClaw Memory** |
|---|---|---|---|---|
| Hybrid retrieval (FTS + vector + rerank) | ❌ | ❌ | Partial | ✅ |
| Self-improving feedback loop | ❌ | ❌ | ❌ | ✅ |
| Contradiction detection + resolution | ❌ | ❌ | ❌ | ✅ 0.900 |
| Multi-hop retrieval (2-4 hops) | ❌ | ❌ | ❌ | ✅ 1.000 |
| Temporal reasoning in retrieval | ❌ | ❌ | ❌ | ✅ |
| CRUD lifecycle (supersede/delete) | ❌ | Partial | ❌ | ✅ 0.955 |
| 9-benchmark evaluation suite | ❌ | ❌ | ❌ | ✅ |
| Progressive time-based summarization | ❌ | ❌ | ❌ | ✅ |
| Memory knowledge graph | ❌ | ❌ | Partial | ✅ |
| Needle-in-haystack at 100K scale | ❌ | ❌ | ❌ | ✅ 1.000 |

---

## How It Works

### Retrieval Pipeline

```
  Query
    │
    ├──► PostgreSQL FTS (ts_rank)  ──┐
    │                                 │
    ├──► Qdrant Vector Search ───────┤
    │    (384-dim, cosine)           │
    │                                 ▼
    │                          ┌─────────────┐
    │                          │ Score Fusion │
    │                          │              │
    │                          │ FTS × 3.0    │
    │                          │ Vector × 1.1 │
    │                          │ + Structured  │
    │                          │ + Importance  │
    │                          │ + Temporal    │
    │                          └──────┬──────┘
    │                                 │
    │                                 ▼
    │                     ┌───────────────────┐
    │                     │ Cross-Encoder      │
    │                     │ Reranking          │
    │                     │ (ms-marco-MiniLM)  │
    │                     └─────────┬─────────┘
    │                               │
    │                               ▼
    │                     ┌───────────────────┐
    │                     │ DB Status Filter   │
    │                     │ + Feedback Boost   │
    │                     └─────────┬─────────┘
    │                               │
    └───────────────────────────────▼
                              Final Results
```

### Key Innovations

- **Relative Temporal Ranking** — `ROW_NUMBER()` within entity+property partitions. When two facts conflict, the newer one wins. Not based on absolute time from now — works on synthetic and real data equally.
- **Hybrid SQL Push-down** — Metadata filters run inside the SQL `WHERE` clause, not post-filtered on top-50. No session-scoped items get lost.
- **Cached Cross-Encoder** — Module-level singleton eliminates 52× latency penalty from repeated model loading.
- **Post-Retrieval Status Check** — Qdrant vectors don't carry status. DB re-verification catches superseded/deleted items that leaked through vector search.
- **Dual-Level NLI** — Sentence-level + paragraph-level scoring for fact verification. Catches contradictions that paragraph-level alone dilutes.

### Memory Lifecycle

```
candidate ──► durable ──► superseded (replaced by newer fact)
                 │    ──► consolidated (rolled into summary)
                 │    ──► discarded (low quality)
                 │
                 └──── re-confirmed (still true)
```

### Storage Layer

| Store | Role | Scale |
|---|---|---|
| **PostgreSQL** | Memory items, FTS index, metadata, lifecycle state | 18K+ items |
| **Qdrant** | 384-dim vector index (all-MiniLM-L6-v2) | Tested to 100K |
| **Neo4j** | Knowledge graph — SUPERSEDES, CONTRADICTS, CAUSED_BY, etc. | Typed relationships |

---

## Quickstart

```bash
# 1. Start infrastructure
git clone https://github.com/josephs-ai/Memory-System-claw.git
cd Memory-System-claw
docker compose up -d   # PostgreSQL, Qdrant, Neo4j

# 2. Install
pip install -e ".[all]"

# 3. Configure
export OPENCLAW_MEMORY_DSN="host=localhost dbname=openclaw_memory user=openclaw password=openclaw"

# 4. Initialize
psql "$OPENCLAW_MEMORY_DSN" -f scripts/memory_db_schema.sql

# 5. Start search service
python scripts/search_memory_service.py --port 8791

# 6. Query
curl "http://localhost:8791/search?q=how+does+the+retrieval+pipeline+work&limit=5"
```

---

## Models (all small, no LLM)

| Model | Params | Role |
|---|---|---|
| `all-MiniLM-L6-v2` | 23M | Text embeddings (384-dim) |
| `ms-marco-MiniLM-L-6-v2` | 23M | Cross-encoder reranking + answer relevance |
| `nli-deberta-v3-base` | 86M | NLI classification for fact verification |

Total: **132M parameters**. Runs on CPU. No API calls. No tokens burned.

---

## Memory Types

| Type | Description |
|---|---|
| `fact` | Verified information |
| `decision` | Choices made and their rationale |
| `lesson_learned` | Insights from mistakes or successes |
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

## Benchmark Suite

Run the full evaluation suite:

```bash
cd benchmarks

# Individual benchmarks
python -m niah.run --save
python -m msc.run --save
python -m hotpotqa.run --save
python -m musique.run --save
python -m crud_rag.run --save
python -m temporalqa.run --save
python -m contradiction.run --save
python -m fever.run --save
python -m locomo.run --max-convs 3 --save

# All at once
python run_all.py
```

Results saved to `benchmarks/results/` as timestamped JSON. The control panel at `/benchmarks` renders them with interactive charts.

---

## Self-Improving Retrieval

Every retrieval can be marked useful or bad. Feedback scores blend into future rankings — no retraining, no LLM calls:

```python
from scripts.mark_retrieval_useful import mark_useful
from scripts.mark_retrieval_bad import mark_bad

mark_useful(item_id="mem-abc123", query="how does auth work")
mark_bad(item_id="mem-xyz789", query="deployment steps")
```

---

## Project Structure

```
scripts/
├── search_memory.py              # Core hybrid search
├── search_memory_service.py      # FastAPI search API (port 8791)
├── memory_db.py                  # PostgreSQL operations
├── embed_memory_items.py         # Qdrant vector indexing
├── feedback_score_engine.py      # Self-improving retrieval
├── context_hydrator.py           # Unified retrieval (code + memory + graph)
├── temporal_scoring.py           # Time-aware scoring
├── memory_knowledge_graph.py     # Semantic links (Neo4j)
├── consolidation_grouper.py      # Progressive summarization
├── dehydrate_transcript.py       # Transcript → clean text
├── checkpoint_agent.py           # Agent memory checkpointing
└── memory_db_schema.sql          # Database schema

benchmarks/
├── niah/                         # Needle in a Haystack (10K-100K)
├── msc/                          # Multi-Session Chat (100 personas)
├── hotpotqa/                     # 2-hop reasoning
├── musique/                      # 2-4 hop reasoning
├── crud_rag/                     # CRUD lifecycle
├── temporalqa/                   # Temporal retrieval
├── contradiction/                # Conflict resolution
├── fever/                        # Fact verification
├── locomo/                       # Long conversation memory
├── common.py                     # Shared retrieval pipeline
├── run_all.py                    # Run all benchmarks
└── results/                      # Timestamped JSON results

control_panel/                    # Web dashboard (9 pages, live data)
```

---

## Configuration

| Variable | Default | Description |
|---|---|---|
| `OPENCLAW_MEMORY_DSN` | `dbname=openclaw_memory` | PostgreSQL connection |
| `QDRANT_HOST` | `localhost` | Qdrant server |
| `QDRANT_PORT` | `6333` | Qdrant port |
| `NEO4J_URI` | `bolt://localhost:7687` | Neo4j connection |
| `OPENCLAW_RERANK_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Cross-encoder model |
| `OPENCLAW_RERANK_CAP` | `20` | Max items to rerank |

---

## Tests

```bash
python -m pytest scripts/ -q
```

466 tests covering retrieval, lifecycle, feedback, summarization, temporal scoring, and knowledge graph operations.

---

## License

MIT — see [LICENSE](LICENSE).
