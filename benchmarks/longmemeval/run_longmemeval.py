#!/usr/bin/env python3
"""
LongMemEval benchmark adapter for the OpenClaw Memory System.

Ingests LongMemEval chat sessions into our memory pipeline (PostgreSQL + embeddings),
then queries our hybrid search for each question and evaluates answer quality.

Pipeline:
  1. Ingest: Each session's turns → memory items via our DB layer
  2. Retrieve: Question → hybrid search → top-k evidence
  3. Answer: Evidence + question → LLM generates answer
  4. Evaluate: Compare generated answer to gold answer

Usage:
    python run_longmemeval.py --dataset longmemeval_s.json --limit 50
    python run_longmemeval.py --dataset longmemeval_oracle.json --limit 500
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Add scripts to path for memory_db access
SCRIPT_DIR = Path(__file__).resolve().parent.parent.parent / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

BENCHMARK_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BENCHMARK_DIR / "results"


# ---------------------------------------------------------------------------
# Memory system integration
# ---------------------------------------------------------------------------

def get_db_dsn():
    return os.environ.get("OPENCLAW_MEMORY_DB_DSN", "dbname=openclaw_memory")


def setup_benchmark_tables():
    """Create isolated benchmark tables so we don't pollute the real memory DB."""
    import psycopg
    with psycopg.connect(get_db_dsn()) as conn:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS lme_memory_items (
                    id TEXT PRIMARY KEY,
                    text TEXT NOT NULL,
                    session_id TEXT,
                    question_id TEXT,
                    role TEXT,
                    turn_index INT,
                    session_date TEXT,
                    created_at TIMESTAMPTZ DEFAULT now()
                )
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS lme_embeddings (
                    item_id TEXT PRIMARY KEY REFERENCES lme_memory_items(id),
                    embedding vector(384),
                    created_at TIMESTAMPTZ DEFAULT now()
                )
            """)
            # FTS index
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_lme_text_fts
                ON lme_memory_items USING gin(to_tsvector('english', text))
            """)
            # Vector index
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_lme_emb_cosine
                ON lme_embeddings USING ivfflat (embedding vector_cosine_ops) WITH (lists = 50)
            """)
        conn.commit()
    print("  [setup] Benchmark tables ready")


def clear_benchmark_tables(question_id: str | None = None):
    """Clear benchmark data, optionally scoped to a question."""
    import psycopg
    with psycopg.connect(get_db_dsn()) as conn:
        with conn.cursor() as cur:
            if question_id:
                cur.execute("DELETE FROM lme_embeddings WHERE item_id IN (SELECT id FROM lme_memory_items WHERE question_id = %s)", (question_id,))
                cur.execute("DELETE FROM lme_memory_items WHERE question_id = %s", (question_id,))
            else:
                cur.execute("DELETE FROM lme_embeddings")
                cur.execute("DELETE FROM lme_memory_items")
        conn.commit()


def ingest_sessions(question_id: str, sessions: list[list[dict]], session_ids: list[str],
                    haystack_dates: list[str] | None = None):
    """Ingest all sessions for a question into benchmark tables."""
    import psycopg
    items = []
    for s_idx, (session, s_id) in enumerate(zip(sessions, session_ids)):
        date = haystack_dates[s_idx] if haystack_dates and s_idx < len(haystack_dates) else ""
        for t_idx, turn in enumerate(session):
            role = turn.get("role", "user")
            content = turn.get("content", "").strip()
            if not content:
                continue
            item_id = f"lme-{question_id}-{s_id}-{t_idx}"
            items.append({
                "id": item_id,
                "text": content,
                "session_id": s_id,
                "question_id": question_id,
                "role": role,
                "turn_index": t_idx,
                "session_date": date,
            })

    if not items:
        return 0

    with psycopg.connect(get_db_dsn()) as conn:
        with conn.cursor() as cur:
            for item in items:
                cur.execute("""
                    INSERT INTO lme_memory_items (id, text, session_id, question_id, role, turn_index, session_date)
                    VALUES (%(id)s, %(text)s, %(session_id)s, %(question_id)s, %(role)s, %(turn_index)s, %(session_date)s)
                    ON CONFLICT (id) DO NOTHING
                """, item)
        conn.commit()

    return len(items)


_ST_MODEL = None

def _get_st_model():
    global _ST_MODEL
    if _ST_MODEL is None:
        from sentence_transformers import SentenceTransformer
        _ST_MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _ST_MODEL


def embed_items(question_id: str):
    """Embed all items for a question using MiniLM."""
    import psycopg

    model = _get_st_model()

    with psycopg.connect(get_db_dsn()) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, text FROM lme_memory_items WHERE question_id = %s AND id NOT IN (SELECT item_id FROM lme_embeddings)",
                (question_id,),
            )
            rows = cur.fetchall()

        if not rows:
            return 0

        ids = [r[0] for r in rows]
        texts = [r[1] for r in rows]

        # Batch encode
        embeddings = model.encode(texts, normalize_embeddings=True, batch_size=128, show_progress_bar=False)
        print(f"    embedded {len(texts)} items", flush=True)

        with conn.cursor() as cur:
            for item_id, emb in zip(ids, embeddings):
                cur.execute(
                    "INSERT INTO lme_embeddings (item_id, embedding) VALUES (%s, %s) ON CONFLICT (item_id) DO NOTHING",
                    (item_id, emb.tolist()),
                )
        conn.commit()

    return len(rows)


def retrieve_evidence(question: str, question_id: str, top_k: int = 10) -> list[dict]:
    """Run hybrid search (vector + FTS) against benchmark tables for a specific question."""
    import psycopg

    model = _get_st_model()
    q_emb = model.encode(question, normalize_embeddings=True)

    with psycopg.connect(get_db_dsn()) as conn:
        with conn.cursor() as cur:
            # Hybrid: vector similarity + FTS
            cur.execute("""
                WITH vec AS (
                    SELECT e.item_id, 1 - (e.embedding <=> %s::vector) AS vec_score
                    FROM lme_embeddings e
                    JOIN lme_memory_items m ON m.id = e.item_id
                    WHERE m.question_id = %s
                    ORDER BY e.embedding <=> %s::vector
                    LIMIT %s
                ),
                fts AS (
                    SELECT m.id AS item_id,
                           ts_rank_cd(to_tsvector('english', m.text), websearch_to_tsquery('english', %s)) AS fts_score
                    FROM lme_memory_items m
                    WHERE m.question_id = %s
                      AND to_tsvector('english', m.text) @@ websearch_to_tsquery('english', %s)
                    ORDER BY fts_score DESC
                    LIMIT %s
                ),
                combined AS (
                    SELECT COALESCE(v.item_id, f.item_id) AS item_id,
                           COALESCE(v.vec_score, 0) AS vec_score,
                           COALESCE(f.fts_score, 0) AS fts_score
                    FROM vec v
                    FULL OUTER JOIN fts f ON v.item_id = f.item_id
                )
                SELECT c.item_id, c.vec_score, c.fts_score,
                       (0.6 * c.vec_score + 0.4 * c.fts_score) AS hybrid_score,
                       m.text, m.role, m.session_id, m.session_date
                FROM combined c
                JOIN lme_memory_items m ON m.id = c.item_id
                ORDER BY (0.6 * c.vec_score + 0.4 * c.fts_score) DESC
                LIMIT %s
            """, (q_emb.tolist(), question_id, q_emb.tolist(), top_k * 2,
                  question, question_id, question, top_k * 2,
                  top_k))

            rows = cur.fetchall()
            cols = [d[0] for d in cur.description]
            return [dict(zip(cols, row)) for row in rows]


def generate_answer(question: str, evidence: list[dict], model_name: str = "gpt-4o-mini") -> str:
    """Use an LLM to answer the question given retrieved evidence."""
    import requests

    # Build evidence context
    evidence_text = ""
    for i, e in enumerate(evidence, 1):
        date = e.get("session_date", "")
        role = e.get("role", "")
        text = e.get("text", "")[:500]
        evidence_text += f"\n[{i}] ({date}, {role}): {text}\n"

    prompt = f"""You are answering a question about past conversations. Use ONLY the evidence provided.
If the evidence doesn't contain the answer, say "I don't know."
Be concise - answer in a few words or a short sentence.

Evidence from past conversations:
{evidence_text}

Question: {question}
Answer:"""

    # Try OpenAI API
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        cfg_path = Path.home() / ".openclaw" / "openclaw.json"
        if cfg_path.exists():
            raw = cfg_path.read_text()
            cleaned = re.sub(r',\s*([}\]])', r'\1', raw)
            try:
                cfg = json.loads(cleaned)
                api_key = cfg.get("models", {}).get("providers", {}).get("openai", {}).get("apiKey", "")
            except Exception:
                pass

    if not api_key:
        # Fallback: extract answer from top evidence
        return evidence[0]["text"][:200] if evidence else "I don't know"

    resp = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={
            "model": model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 100,
            "temperature": 0,
        },
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()


# ---------------------------------------------------------------------------
# Evaluation (LongMemEval's scoring)
# ---------------------------------------------------------------------------

def normalize_answer(s: str) -> str:
    """Normalize answer for comparison."""
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9\s]", "", s)
    s = re.sub(r"\s+", " ", s)
    return s


def compute_f1(prediction: str, gold: str) -> float:
    """Token-level F1 score."""
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(gold).split()

    if not pred_tokens or not gold_tokens:
        return 1.0 if pred_tokens == gold_tokens else 0.0

    common = set(pred_tokens) & set(gold_tokens)
    if not common:
        return 0.0

    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def compute_em(prediction: str, gold: str) -> float:
    """Exact match score."""
    return 1.0 if normalize_answer(prediction) == normalize_answer(gold) else 0.0


def contains_answer(prediction: str, gold: str) -> float:
    """Check if gold answer is contained in prediction."""
    return 1.0 if normalize_answer(gold) in normalize_answer(prediction) else 0.0


# ---------------------------------------------------------------------------
# Main benchmark runner
# ---------------------------------------------------------------------------

def run_benchmark(
    dataset_path: str,
    limit: int = 500,
    top_k: int = 10,
    answer_model: str = "gpt-4o-mini",
    skip_answer: bool = False,
):
    """Run the full LongMemEval benchmark."""
    print(f"Loading dataset: {dataset_path}")
    with open(dataset_path) as f:
        data = json.load(f)

    if limit:
        data = data[:limit]

    print(f"Running {len(data)} questions, top_k={top_k}, answer_model={answer_model}")
    print()

    setup_benchmark_tables()

    # Pre-load embedding model once
    print("Loading embedding model...")
    _get_st_model()

    results = []
    total_ingest_time = 0
    total_embed_time = 0
    total_retrieve_time = 0
    total_answer_time = 0

    for i, item in enumerate(data):
        qid = item["question_id"]
        qtype = item["question_type"]
        question = item["question"]
        gold_answer = item["answer"]
        sessions = item.get("haystack_sessions", [])
        session_ids = item.get("haystack_session_ids", [])
        dates = item.get("haystack_dates", [])

        is_abstention = qid.endswith("_abs")

        print(f"[{i+1}/{len(data)}] {qid} ({qtype}) — {len(sessions)} sessions", flush=True)

        # 1. Clear previous data for this question
        clear_benchmark_tables(qid)

        # 2. Ingest sessions
        t0 = time.monotonic()
        n_items = ingest_sessions(qid, sessions, session_ids, dates)
        ingest_time = time.monotonic() - t0
        total_ingest_time += ingest_time

        # 3. Embed
        t0 = time.monotonic()
        n_embedded = embed_items(qid)
        embed_time = time.monotonic() - t0
        total_embed_time += embed_time

        # 4. Retrieve
        t0 = time.monotonic()
        evidence = retrieve_evidence(question, qid, top_k=top_k)
        retrieve_time = time.monotonic() - t0
        total_retrieve_time += retrieve_time

        # Check if evidence contains answer sessions
        answer_sids = set(item.get("answer_session_ids", []))
        retrieved_sids = set(e.get("session_id", "") for e in evidence)
        retrieval_hit = bool(answer_sids & retrieved_sids)

        # 5. Generate answer
        predicted = ""
        answer_time = 0
        if not skip_answer and evidence:
            t0 = time.monotonic()
            try:
                predicted = generate_answer(question, evidence, model_name=answer_model)
            except Exception as e:
                predicted = f"[ERROR: {e}]"
            answer_time = time.monotonic() - t0
            total_answer_time += answer_time

        # 6. Score
        f1 = compute_f1(predicted, gold_answer)
        em = compute_em(predicted, gold_answer)
        contains = contains_answer(predicted, gold_answer)

        result = {
            "question_id": qid,
            "question_type": qtype,
            "question": question,
            "gold_answer": gold_answer,
            "predicted_answer": predicted,
            "is_abstention": is_abstention,
            "n_sessions": len(sessions),
            "n_items_ingested": n_items,
            "n_embedded": n_embedded,
            "n_evidence": len(evidence),
            "retrieval_hit": retrieval_hit,
            "f1": f1,
            "em": em,
            "contains": contains,
            "ingest_ms": round(ingest_time * 1000, 1),
            "embed_ms": round(embed_time * 1000, 1),
            "retrieve_ms": round(retrieve_time * 1000, 1),
            "answer_ms": round(answer_time * 1000, 1),
        }
        results.append(result)

        status = "✓" if contains > 0.5 else "✗"
        print(f"  {status} F1={f1:.2f} EM={em:.0f} contains={contains:.0f} "
              f"retrieval_hit={retrieval_hit} "
              f"[ingest={ingest_time:.1f}s embed={embed_time:.1f}s retrieve={retrieve_time*1000:.0f}ms]",
              flush=True)

    # Aggregate
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    from collections import defaultdict
    by_type = defaultdict(list)
    for r in results:
        by_type[r["question_type"]].append(r)
        by_type["ALL"].append(r)

    aggregate = {}
    for qtype, items in sorted(by_type.items()):
        n = len(items)
        avg_f1 = sum(r["f1"] for r in items) / n
        avg_em = sum(r["em"] for r in items) / n
        avg_contains = sum(r["contains"] for r in items) / n
        avg_retrieval_hit = sum(r["retrieval_hit"] for r in items) / n
        avg_retrieve_ms = sum(r["retrieve_ms"] for r in items) / n

        aggregate[qtype] = {
            "count": n,
            "f1": round(avg_f1, 4),
            "em": round(avg_em, 4),
            "contains": round(avg_contains, 4),
            "retrieval_hit": round(avg_retrieval_hit, 4),
            "avg_retrieve_ms": round(avg_retrieve_ms, 1),
        }

        print(f"\n{qtype} (n={n}):")
        print(f"  F1:             {avg_f1:.4f}")
        print(f"  Exact Match:    {avg_em:.4f}")
        print(f"  Contains:       {avg_contains:.4f}")
        print(f"  Retrieval Hit:  {avg_retrieval_hit:.4f}")
        print(f"  Avg Retrieve:   {avg_retrieve_ms:.1f}ms")

    print(f"\nTotal time: ingest={total_ingest_time:.1f}s embed={total_embed_time:.1f}s "
          f"retrieve={total_retrieve_time:.1f}s answer={total_answer_time:.1f}s")

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ds_name = Path(dataset_path).stem
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"longmemeval_{ds_name}_{ts}.json"

    output = {
        "benchmark": "LongMemEval",
        "dataset": ds_name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": {
            "limit": limit,
            "top_k": top_k,
            "answer_model": answer_model,
            "skip_answer": skip_answer,
        },
        "aggregate": aggregate,
        "per_question": results,
    }

    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")

    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LongMemEval against OpenClaw memory system")
    parser.add_argument("--dataset", default=str(BENCHMARK_DIR / "longmemeval_s.json"))
    parser.add_argument("--limit", type=int, default=50, help="Number of questions to run")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--answer-model", default="gpt-4o-mini")
    parser.add_argument("--skip-answer", action="store_true", help="Skip LLM answer generation (retrieval-only eval)")
    args = parser.parse_args()

    run_benchmark(
        dataset_path=args.dataset,
        limit=args.limit,
        top_k=args.top_k,
        answer_model=args.answer_model,
        skip_answer=args.skip_answer,
    )
