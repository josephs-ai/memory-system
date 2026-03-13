# OpenClaw Memory System

A DB-backed long-term memory system for OpenClaw that turns raw session/transcript activity into structured, retrievable, durable memory.

## What this system does

The memory system takes conversation/session output, removes junk, extracts durable updates, routes them by confidence and scope, stores approved memory in a canonical database, generates embeddings for hybrid retrieval, and records retrieval feedback so ranking can improve over time.

In practice, it is built to answer one question well:

> How do you turn temporary agent/session activity into trustworthy long-term memory without filling the system with junk?

---

## High-level pipeline

1. **Transcript/session source**
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

8. **Embedding / retrieval**
   - Active canonical items are embedded with Sentence-Transformers.
   - Retrieval uses hybrid ranking:
     - full-text search
     - vector similarity
     - structured bonuses
     - importance weighting
     - optional RBAC filtering

9. **Feedback**
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

- **Hybrid retrieval beats simple text search**  
  Durable memory must be discoverable through both exact phrasing and paraphrase.

- **Feedback closes the loop**  
  Retrieval quality should improve from operator signals, not stay static.

---

## Main folders

### `.memory-index/scripts/`
Core pipeline scripts, routing logic, retrieval, diagnostics, maintenance, and helpers.

### `.memory-index/control_panel/`
Split web UI for:
- agent management
- heartbeat configuration
- project tracker
- command runner
- architecture explorer

### `memory/`
Human-readable memory/project files and review-oriented artifacts.

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
  Cleans raw transcript/session output.

- `chunk_by_topic.py`  
  Splits cleaned transcript into usable chunks.

- `extract_chunk_updates.py`  
  Extracts candidate memory updates from chunks.

- `extract_memory_fields.py`  
  Extracts entity/property/value-style structure.

- `judge_memory_candidates.py`  
  Converts candidates into structured memory records and assigns confidence, importance, and route hints.

## Routing / policy
- `memory_routing_rules.py`  
  Shared routing-policy helper logic, including safe stable auto-promotion rules.

- `route_memory_items_batch.py`  
  Main batch router for judged items.

- `route_memory_item.py`  
  Single-item routing path.

## Retrieval / embeddings
- `embed_memory_items.py`  
  Encodes active canonical memory into embeddings.

- `search_memory.py`  
  Main hybrid memory search.

- `search_memory_rbac.py`  
  Hybrid retrieval with sensitivity / access filtering.

- `test_hybrid_search.py`  
  Debugging and validation helper for ranking behavior.

## Feedback / diagnostics
- `mark_retrieval_useful.py`
- `mark_retrieval_bad.py`
- `show_review_queue.py`
- `report_memory_maintenance.py`

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
Embedding table for hybrid retrieval.

### `retrieval_feedback`
Feedback table for retrieval quality signals.

---

## Current routing behavior

The system is **mostly automatic**, but not blindly so.

- Low-confidence or junk-like items are discarded.
- Project-scoped items usually go to inbox/project review flow.
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

## Retrieval model

Current embedding path uses:

- `sentence-transformers/all-MiniLM-L6-v2`

Embeddings are normalized so vector retrieval works efficiently and hybrid search can combine:
- semantic similarity
- full-text relevance
- structured term overlap
- importance weighting

---

## Control panel

The active UI is the **split control panel** under:

- `.memory-index/control_panel/`

It is used to inspect and manage:
- agents
- heartbeat rules
- tracked paths
- projects
- commands
- memory architecture

Legacy or backup single-file control-panel code should be treated as non-authoritative.

---

## Running the control panel

```bash
cd ~/.openclaw/workspace/.memory-index/control_panel
OPENCLAW_CONTROL_PANEL_PORT=8788 ~/.openclaw/venvs/memory-db/bin/python -m uvicorn app:app --host 127.0.0.1 --port 8788 --reload
