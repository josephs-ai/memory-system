# OpenClaw Memory System

A database-backed long-term memory system for OpenClaw that turns raw session and transcript activity into structured, reviewable, retrievable, durable memory.

## What this system does

The memory system takes conversation and session output, removes junk, extracts durable updates, classifies them into structured memory items, routes them by confidence and scope, stores approved memory in a canonical database, syncs vector and graph retrieval layers, and records retrieval feedback so ranking can improve over time.

In practice, it is built to answer one question well:

> How do you turn temporary agent and session activity into trustworthy long-term memory without filling the system with junk?

---

## High-level pipeline

1. **Transcript / session source**
   - Raw session output is produced by Claw.

2. **Dehydration**
   - Transcript noise and low-value structure are stripped out.

3. **Chunking + update extraction**
   - The cleaned transcript is split into chunks and candidate memory updates are extracted.

4. **Judging / structuring**
   - Candidates are classified into structured memory items with:
     - `memory_type`
     - `scope`
     - `entity`
     - `property`
     - `value`
     - `confidence`
     - `importance`
     - `suggested_route`

5. **Routing**
   - Items are routed to:
     - canonical memory directly
     - inbox
     - pending stable approval
     - discarded

6. **Approval / lifecycle**
   - Higher-risk stable memory can require review before promotion.
   - Items can also be promoted, demoted, restored, or rejected.

7. **Canonical storage**
   - Durable memory is stored in `memory_items` through the DB layer.

8. **Embedding + graph sync**
   - Active canonical items are embedded and written into:
     - the local embedding table
     - Qdrant for vector recall
   - Structured `entity/property/value` memory is also synced into Neo4j for graph recall.

9. **Hybrid retrieval**
   - Retrieval now combines:
     - database lexical / structured search
     - Qdrant vector search
     - Neo4j graph recall
     - cross-encoder reranking

10. **Feedback**
   - Retrieval results can be marked useful or bad.
   - Feedback is stored and can be used to improve ranking behavior.

---

## Core design principles

- **Structured memory over raw blobs**  
  Durable memory should be explicit and atomic, not vague transcript residue.

- **Confidence-gated durability**  
  Not all observations deserve permanent memory.

- **Scope matters**  
  Stable system policy, project-scoped state, and daily observations should not be treated the same way.

- **Hybrid retrieval beats single-mode retrieval**  
  Durable memory should be discoverable through exact phrasing, paraphrase, and graph structure.

- **Feedback closes the loop**  
  Retrieval quality should improve from operator signals, not stay static.

- **Postgres remains the source of truth**  
  Vector and graph layers support retrieval, but canonical lifecycle and approval still live in the main DB.

---

## Current architecture

The system now uses a multi-layer memory architecture:

### 1. Postgres / canonical lifecycle layer
Used for:
- canonical memory rows
- queues
- approvals
- lifecycle changes
- embeddings table
- retrieval feedback

### 2. Qdrant / vector retrieval layer
Used for:
- semantic vector recall over active canonical memory

### 3. Neo4j / graph memory layer
Used for:
- entity / property / value graph relationships
- graph-assisted recall
- conflict and relationship inspection

### 4. Cross-encoder reranker
Used for:
- final ranking of merged candidates from DB, Qdrant, and Neo4j

---

## Main folders

### `.memory-index/scripts/`
Core pipeline scripts, routing logic, retrieval, diagnostics, maintenance, vector helpers, and graph helpers.

### `.memory-index/control_panel/`
Split web UI for:
- agent management
- heartbeat configuration
- project tracker
- command runner
- architecture explorer
- hybrid memory search
- conflict controls
- live queue / heartbeat monitoring

### `memory/`
Human-readable memory and project files plus review-oriented artifacts.

---

## Most important scripts

## Database / canonical memory
- `memory_db.py`  
  Main DB access layer for memory items, queues, embeddings, feedback, and search helpers.

- `approve_pending_memory_item.py`  
  Approves stable memory from pending review into canonical storage.

- `sync_registry_to_markdown.py`  
  Syncs canonical DB state into human-readable markdown mirrors.

## Ingest / judgment
- `dehydrate_transcript.py`  
  Cleans raw transcript and session output.

- `chunk_by_topic.py`  
  Splits cleaned transcript into usable chunks.

- `extract_chunk_updates.py`  
  Extracts candidate memory updates from chunks.

- `extract_memory_fields.py`  
  Extracts entity / property / value structure and query hints.

- `judge_memory_candidates.py`  
  Converts candidates into structured memory records and assigns confidence, importance, and route hints.

## Routing / policy
- `memory_routing_rules.py`  
  Shared routing-policy helper logic, including safe stable auto-promotion rules.

- `route_memory_items_batch.py`  
  Main batch router for judged items.

- `route_memory_item.py`  
  Single-item routing path.

## Retrieval / embeddings / graph
- `embed_memory_items.py`  
  Encodes active canonical memory into embeddings and dual-writes vectors.

- `search_memory.py`  
  Main hybrid retrieval path: DB + Qdrant + Neo4j + reranking.

- `search_memory_rbac.py`  
  Hybrid retrieval with sensitivity / access filtering.

- `search_qdrant.py`  
  Direct Qdrant vector retrieval sanity check.

- `vector_store_qdrant.py`  
  Qdrant helper layer for collection creation, upsert, and search.

- `graph_store_neo4j.py`  
  Neo4j helper layer for graph constraints and graph upserts.

- `sync_memory_to_neo4j.py`  
  Syncs canonical memory rows into the Neo4j graph.

- `query_neo4j.py`  
  Direct graph query sanity check.

- `rerank_crossencoder.py`  
  Test script for final reranking with a cross-encoder.

## Feedback / diagnostics
- `mark_retrieval_useful.py`
- `mark_retrieval_bad.py`
- `show_review_queue.py`
- `report_memory_maintenance.py`
- `test_hybrid_search.py`

---

## Memory states / queues

### `memory_items`
Canonical durable memory table.

### `memory_inbox`
Review queue for uncertain, project-scoped, or underspecified items.

### `memory_pending_stable`
Approval queue for strong stable memories that are not yet safe enough to auto-promote.

### `memory_discarded`
Discard cache and rejection history.

### `memory_item_embeddings`
Local embedding table used alongside vector retrieval.

### `retrieval_feedback`
Feedback table for retrieval quality signals.

---

## Current routing behavior

The system is mostly automatic, but not blindly so.

- Low-confidence or junk-like items are discarded.
- Project-scoped items usually go to inbox / review flow.
- Strong stable items can:
  - auto-promote if they meet the stricter `stable_safe_auto` rule
  - otherwise go to `pending_stable`

The shared source of truth for this policy should live in:

- `memory_routing_rules.py`

---

## Stable safe auto-promotion

A stable memory can auto-promote only when it is safe enough.

Typical requirements:
- `scope == stable`
- high confidence
- high importance
- full structure (`entity`, `property`, `value`)
- atomic / short enough to be durable
- allowed memory type

Anything below that threshold should still go through review.

---

## Retrieval stack

### Embedding model
- `sentence-transformers/all-MiniLM-L6-v2`

### Vector store
- Qdrant

### Graph store
- Neo4j

### Reranker
- `cross-encoder/ms-marco-MiniLM-L-6-v2`

### Final hybrid retrieval path
The main retrieval path in `search_memory.py` now combines:
- database hybrid lexical / structured search
- Qdrant semantic recall
- Neo4j graph recall
- cross-encoder reranking on the merged top candidates

This means memory can now be retrieved by:
- exact matching
- paraphrase similarity
- entity / property / value graph structure
- final semantic reranking

---

## Control panel

The active UI is the split control panel under:

- `.memory-index/control_panel/`

It is used to inspect and manage:
- agents
- heartbeat rules
- tracked paths
- projects
- commands
- memory architecture
- live queue state
- conflict controls
- hybrid search

The control panel now includes:
- **Memory flow monitor**
- **Conflict resolution panel**
- **Heartbeat monitor**
- **Hybrid memory search page**

Legacy or backup single-file control-panel code should be treated as non-authoritative.

---

## Running the control panel

```bash
cd ~/.openclaw/workspace/.memory-index/control_panel
OPENCLAW_CONTROL_PANEL_PORT=8788 ~/.openclaw/venvs/memory-db/bin/python -m uvicorn app:app --host 127.0.0.1 --port 8788 --reload
