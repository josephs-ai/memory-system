# Memory System Benchmark Evaluation Harness

Comprehensive benchmark suite for evaluating the PostgreSQL + Qdrant + Neo4j hybrid memory retrieval system.

## Quick Start

```bash
cd ~/.openclaw/workspace/.memory-index/benchmarks
source ~/.openclaw/workspace/.memory-venv/bin/activate

# Run just NIAH (no external data needed, fastest):
python niah/run.py --sizes 100 1000 --save

# Run all benchmarks in quick mode:
python run_all.py --quick --save

# Run full suite:
python run_all.py --save
```

## Benchmarks

### NIAH — Needle-in-a-Haystack (Synthetic)

**Purpose:** Tests raw retrieval precision — can we find a specific fact buried in a large memory store?

**Method:**
- Generate `N` synthetic "needle" facts (e.g., "The secret passphrase for Project Alpha is ABCD-1234")
- Surround with `K` haystack distractors
- Query for each needle, measure retrieval rank

**Metrics:** recall@1, recall@5, MRR  
**No external data required**

```bash
python niah/run.py --sizes 100 1000 10000
```

---

### LongMemEval (ICLR 2025)

**Purpose:** 500 QA questions over long multi-session chat histories.

**Question types:**
- `information_extraction` — Find specific facts stated in sessions
- `multi_session` — Reasoning across multiple sessions
- `knowledge_update` — Later info supersedes earlier info
- `temporal_reasoning` — Questions about when things happened
- `abstention` — Questions with no answer in context

**Dataset:** `xiaowu0162/longmemeval-cleaned` on HuggingFace (auto-downloaded)

**Metrics:** token F1, exact match

```bash
python longmemeval/run.py --split s --max-entries 50
```

---

### LoCoMo (Snap Research)

**Purpose:** 10 very long conversations (months-long) with 4 QA categories.

**Question categories:** single_hop, multi_hop, temporal, adversarial

**Dataset:** `snap-research/locomo` on GitHub (auto-downloaded)

**Metrics:** token F1, exact match by category

```bash
python locomo/run.py --max-convs 5 --max-qa 20
```

---

### MSC — Multi-Session Chat (Facebook Research)

**Purpose:** Tests recall of user persona facts across conversation sessions.

**Method:** Ingest prior sessions → query for persona facts → measure recall

**Dataset:** `facebook/msc` on HuggingFace (auto-downloaded, with synthetic fallback)

**Metrics:** token F1, persona recall rate

```bash
python msc/run.py --max-examples 50
```

---

## Architecture

```
benchmarks/
  README.md           ← This file
  common.py           ← Shared utilities (scoring, timing, ingestion, cleanup)
  run_all.py          ← Orchestrator: run all + produce table
  datasets/           ← Downloaded raw datasets (gitignored)
    longmemeval/
    locomo/
    msc/
  results/            ← Output JSON + markdown reports
  niah/
    generator.py      ← Synthetic dataset generator
    run.py            ← NIAH runner
  longmemeval/
    adapter.py        ← Dataset loader + ingestion adapter
    run.py            ← LongMemEval runner
  locomo/
    adapter.py
    run.py
  msc/
    adapter.py
    run.py
```

## Design Decisions

### Isolation
Each benchmark run ingests into the live `memory_items` table but uses a unique
`source_agent` prefix (`benchmark_{name}_{run_id}`) for identification. Cleanup
deletes all items matching that prefix. This avoids polluting real memories.

### Scoring
- **Primary score:** token F1 for QA benchmarks, recall@1 for NIAH
- **Secondary:** exact match, MRR, recall@5
- **Latency:** p50 and p95 measured per query
- **Tokens:** approximate token count of retrieved context

### LLM-as-judge
The `common.py::llm_judge_answer()` function calls Claude for semantic evaluation.
Disabled by default (use `--llm-judge` flag) to avoid API costs during development.

## Validated Results (initial run, 2026-06-11)

```
Benchmark     Score    Tokens   Latency p50   Latency p95   Notes
NIAH @100     1.000    1801     3741ms        4493ms        recall@1 on 100-item store
NIAH @500     1.000    1801     3741ms        4493ms        recall@1 on 500-item store
MSC (synth)   0.942    1960     4038ms        5131ms        synthetic personas, F1
LongMemEval   (run when dataset downloaded — 277MB)
LoCoMo        (run when dataset downloaded — 2.7MB)
```

> **Latency note:** p50 ~4s includes pgvector scan over the *entire* `memory_items` table
> (not just benchmark items). This is intentional — it measures real-world noise.
> On a fresh DB or with an index warm-up, expect significantly lower latencies.

## Development Notes

- Python venv: `~/.openclaw/workspace/.memory-venv/`
- DB: `dbname=openclaw_memory` (PostgreSQL)
- Embedding model: `sentence-transformers/all-MiniLM-L6-v2`
- All runners are importable as modules (for orchestration) or runnable as CLI scripts
- Add `--no-cleanup` during debugging to inspect ingested items
