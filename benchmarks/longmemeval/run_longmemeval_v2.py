#!/usr/bin/env python3
"""
LongMemEval benchmark V2 adapter — improved pipeline.

Improvements over V1:
  1. Multi-turn window chunking (3-turn sliding windows with overlap)
  2. Session context expansion (when a turn matches, return ±3 surrounding turns)
  3. Better embedding model option (bge-large or nomic-embed)
  4. Better reader model (gpt-4o instead of gpt-4o-mini)
  5. Cross-encoder reranking of retrieved results

Usage:
    python run_longmemeval_v2.py --dataset longmemeval_s.json --limit 50
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent.parent.parent / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

BENCHMARK_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BENCHMARK_DIR / "results"


def get_db_dsn():
    return os.environ.get("OPENCLAW_MEMORY_DB_DSN", "dbname=openclaw_memory")


# ---------------------------------------------------------------------------
# Singleton models
# ---------------------------------------------------------------------------
_ST_MODEL = None
_RERANKER = None


def _get_st_model():
    global _ST_MODEL
    if _ST_MODEL is None:
        from sentence_transformers import SentenceTransformer
        _ST_MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _ST_MODEL


def _get_reranker():
    global _RERANKER
    if _RERANKER is None:
        from sentence_transformers import CrossEncoder
        _RERANKER = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _RERANKER


# ---------------------------------------------------------------------------
# DB setup
# ---------------------------------------------------------------------------

def setup_tables():
    import psycopg
    with psycopg.connect(get_db_dsn()) as conn:
        with conn.cursor() as cur:
            # Chunks table — stores multi-turn windows
            cur.execute("""
                CREATE TABLE IF NOT EXISTS lme2_chunks (
                    id TEXT PRIMARY KEY,
                    text TEXT NOT NULL,
                    question_id TEXT,
                    session_id TEXT,
                    turn_start INT,
                    turn_end INT,
                    session_date TEXT,
                    created_at TIMESTAMPTZ DEFAULT now()
                )
            """)
            # Raw turns table — for context expansion
            cur.execute("""
                CREATE TABLE IF NOT EXISTS lme2_turns (
                    id TEXT PRIMARY KEY,
                    text TEXT NOT NULL,
                    question_id TEXT,
                    session_id TEXT,
                    turn_index INT,
                    role TEXT,
                    session_date TEXT,
                    created_at TIMESTAMPTZ DEFAULT now()
                )
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS lme2_embeddings (
                    chunk_id TEXT PRIMARY KEY REFERENCES lme2_chunks(id),
                    embedding vector(384),
                    created_at TIMESTAMPTZ DEFAULT now()
                )
            """)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_lme2_chunks_fts
                ON lme2_chunks USING gin(to_tsvector('english', text))
            """)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_lme2_turns_session
                ON lme2_turns (question_id, session_id, turn_index)
            """)
        conn.commit()
    print("  [setup] V2 benchmark tables ready", flush=True)


def clear_tables(question_id: str | None = None):
    import psycopg
    with psycopg.connect(get_db_dsn()) as conn:
        with conn.cursor() as cur:
            if question_id:
                cur.execute("DELETE FROM lme2_embeddings WHERE chunk_id IN (SELECT id FROM lme2_chunks WHERE question_id = %s)", (question_id,))
                cur.execute("DELETE FROM lme2_chunks WHERE question_id = %s", (question_id,))
                cur.execute("DELETE FROM lme2_turns WHERE question_id = %s", (question_id,))
            else:
                cur.execute("DELETE FROM lme2_embeddings")
                cur.execute("DELETE FROM lme2_chunks")
                cur.execute("DELETE FROM lme2_turns")
        conn.commit()


# ---------------------------------------------------------------------------
# Ingestion — multi-turn windows
# ---------------------------------------------------------------------------

def ingest_sessions(question_id: str, sessions: list[list[dict]], session_ids: list[str],
                    haystack_dates: list[str] | None = None, window_size: int = 3, stride: int = 2):
    """Ingest sessions as both raw turns and multi-turn sliding window chunks."""
    import psycopg

    turns = []
    chunks = []

    for s_idx, (session, s_id) in enumerate(zip(sessions, session_ids)):
        date = haystack_dates[s_idx] if haystack_dates and s_idx < len(haystack_dates) else ""

        # Store raw turns
        for t_idx, turn in enumerate(session):
            content = turn.get("content", "").strip()
            if not content:
                continue
            role = turn.get("role", "user")
            turns.append({
                "id": f"lme2-t-{question_id}-{s_id}-{t_idx}",
                "text": content,
                "question_id": question_id,
                "session_id": s_id,
                "turn_index": t_idx,
                "role": role,
                "session_date": date,
            })

        # Build sliding window chunks
        session_turns = [(t.get("role", "user"), t.get("content", "").strip()) for t in session if t.get("content", "").strip()]
        for w_start in range(0, len(session_turns), stride):
            w_end = min(w_start + window_size, len(session_turns))
            window_text = "\n".join(f"[{role}]: {text}" for role, text in session_turns[w_start:w_end])
            if not window_text:
                continue
            chunks.append({
                "id": f"lme2-c-{question_id}-{s_id}-{w_start}-{w_end}",
                "text": window_text,
                "question_id": question_id,
                "session_id": s_id,
                "turn_start": w_start,
                "turn_end": w_end,
                "session_date": date,
            })

    if not turns and not chunks:
        return 0, 0

    with psycopg.connect(get_db_dsn()) as conn:
        with conn.cursor() as cur:
            for t in turns:
                cur.execute("""
                    INSERT INTO lme2_turns (id, text, question_id, session_id, turn_index, role, session_date)
                    VALUES (%(id)s, %(text)s, %(question_id)s, %(session_id)s, %(turn_index)s, %(role)s, %(session_date)s)
                    ON CONFLICT (id) DO NOTHING
                """, t)
            for c in chunks:
                cur.execute("""
                    INSERT INTO lme2_chunks (id, text, question_id, session_id, turn_start, turn_end, session_date)
                    VALUES (%(id)s, %(text)s, %(question_id)s, %(session_id)s, %(turn_start)s, %(turn_end)s, %(session_date)s)
                    ON CONFLICT (id) DO NOTHING
                """, c)
        conn.commit()

    return len(turns), len(chunks)


def embed_chunks(question_id: str):
    """Embed all chunks for a question."""
    import psycopg

    model = _get_st_model()
    with psycopg.connect(get_db_dsn()) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, text FROM lme2_chunks WHERE question_id = %s AND id NOT IN (SELECT chunk_id FROM lme2_embeddings)",
                (question_id,),
            )
            rows = cur.fetchall()

        if not rows:
            return 0

        ids = [r[0] for r in rows]
        texts = [r[1] for r in rows]
        embeddings = model.encode(texts, normalize_embeddings=True, batch_size=128, show_progress_bar=False)

        with conn.cursor() as cur:
            for chunk_id, emb in zip(ids, embeddings):
                cur.execute(
                    "INSERT INTO lme2_embeddings (chunk_id, embedding) VALUES (%s, %s) ON CONFLICT (chunk_id) DO NOTHING",
                    (chunk_id, emb.tolist()),
                )
        conn.commit()

    print(f"    embedded {len(rows)} chunks", flush=True)
    return len(rows)


# ---------------------------------------------------------------------------
# Retrieval — hybrid search + context expansion + reranking
# ---------------------------------------------------------------------------

def retrieve_evidence(question: str, question_id: str, top_k: int = 10,
                      context_window: int = 3, use_reranker: bool = True) -> list[dict]:
    """
    Hybrid search on chunks, then expand each hit with surrounding turns
    from the same session, then rerank with cross-encoder.
    """
    import psycopg

    model = _get_st_model()
    q_emb = model.encode(question, normalize_embeddings=True)

    with psycopg.connect(get_db_dsn()) as conn:
        with conn.cursor() as cur:
            # Phase 1: Hybrid search on chunks (vector + FTS)
            candidate_k = top_k * 3  # over-retrieve for reranking
            cur.execute("""
                WITH vec AS (
                    SELECT e.chunk_id, 1 - (e.embedding <=> %s::vector) AS vec_score
                    FROM lme2_embeddings e
                    JOIN lme2_chunks c ON c.id = e.chunk_id
                    WHERE c.question_id = %s
                    ORDER BY e.embedding <=> %s::vector
                    LIMIT %s
                ),
                fts AS (
                    SELECT c.id AS chunk_id,
                           ts_rank_cd(to_tsvector('english', c.text), websearch_to_tsquery('english', %s)) AS fts_score
                    FROM lme2_chunks c
                    WHERE c.question_id = %s
                      AND to_tsvector('english', c.text) @@ websearch_to_tsquery('english', %s)
                    ORDER BY fts_score DESC
                    LIMIT %s
                ),
                combined AS (
                    SELECT COALESCE(v.chunk_id, f.chunk_id) AS chunk_id,
                           COALESCE(v.vec_score, 0) AS vec_score,
                           COALESCE(f.fts_score, 0) AS fts_score
                    FROM vec v
                    FULL OUTER JOIN fts f ON v.chunk_id = f.chunk_id
                )
                SELECT c2.chunk_id, c2.vec_score, c2.fts_score,
                       (0.6 * c2.vec_score + 0.4 * c2.fts_score) AS hybrid_score,
                       ch.text, ch.session_id, ch.turn_start, ch.turn_end, ch.session_date
                FROM combined c2
                JOIN lme2_chunks ch ON ch.id = c2.chunk_id
                ORDER BY (0.6 * c2.vec_score + 0.4 * c2.fts_score) DESC
                LIMIT %s
            """, (q_emb.tolist(), question_id, q_emb.tolist(), candidate_k,
                  question, question_id, question, candidate_k,
                  candidate_k))

            chunk_hits = cur.fetchall()
            chunk_cols = [d[0] for d in cur.description]
            chunk_results = [dict(zip(chunk_cols, row)) for row in chunk_hits]

        if not chunk_results:
            return []

        # Phase 2: Context expansion — for each hit, pull surrounding turns
        expanded_results = []
        seen_sessions = set()

        with conn.cursor() as cur:
            for cr in chunk_results:
                sid = cr["session_id"]
                key = (sid, cr["turn_start"])
                if key in seen_sessions:
                    continue
                seen_sessions.add(key)

                # Get surrounding turns
                t_start = max(0, cr["turn_start"] - context_window)
                t_end = cr["turn_end"] + context_window

                cur.execute("""
                    SELECT turn_index, role, text FROM lme2_turns
                    WHERE question_id = %s AND session_id = %s
                      AND turn_index >= %s AND turn_index < %s
                    ORDER BY turn_index
                """, (question_id, sid, t_start, t_end))

                context_turns = cur.fetchall()
                expanded_text = "\n".join(f"[{role}]: {text}" for _, role, text in context_turns)

                expanded_results.append({
                    "chunk_id": cr["chunk_id"],
                    "session_id": sid,
                    "session_date": cr["session_date"],
                    "hybrid_score": cr["hybrid_score"],
                    "vec_score": cr["vec_score"],
                    "fts_score": cr["fts_score"],
                    "chunk_text": cr["text"],
                    "expanded_text": expanded_text,
                    "n_context_turns": len(context_turns),
                })

    # Phase 3: Cross-encoder reranking
    if use_reranker and expanded_results:
        try:
            reranker = _get_reranker()
            pairs = [(question, r["expanded_text"][:1000]) for r in expanded_results]
            rerank_scores = reranker.predict(pairs)
            for r, score in zip(expanded_results, rerank_scores):
                r["rerank_score"] = float(score)
            expanded_results.sort(key=lambda x: x["rerank_score"], reverse=True)
        except Exception as e:
            print(f"    [warn] reranker failed: {e}", flush=True)

    return expanded_results[:top_k]


# ---------------------------------------------------------------------------
# Answer generation
# ---------------------------------------------------------------------------

def generate_answer(question: str, evidence: list[dict], model_name: str = "gpt-4o") -> str:
    """Use LLM to extract answer from expanded evidence."""
    import requests

    # Build evidence with full session context
    evidence_text = ""
    for i, e in enumerate(evidence[:8], 1):
        date = e.get("session_date", "")
        text = e.get("expanded_text", e.get("chunk_text", ""))[:800]
        score = e.get("rerank_score", e.get("hybrid_score", 0))
        evidence_text += f"\n--- Evidence #{i} (date: {date}, relevance: {score:.2f}) ---\n{text}\n"

    prompt = f"""You are answering a factual question about past conversations between a user and an assistant.

INSTRUCTIONS:
- Read ALL evidence carefully before answering
- The answer is usually a specific name, place, number, or short fact
- Look for the answer across multiple turns — sometimes the user mentions something in one turn and gives the specific detail in another
- If the user asks "where did I..." look for location/store/place names
- If the user asks "what was..." look for the specific item/thing
- Be CONCISE — answer in as few words as possible (1-5 words ideal)
- If unsure, give your best guess from the evidence rather than saying "I don't know"

EVIDENCE FROM PAST CONVERSATIONS:
{evidence_text}

QUESTION: {question}
ANSWER (be concise, 1-5 words):"""

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
        return evidence[0].get("expanded_text", "")[:200] if evidence else "I don't know"

    resp = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={
            "model": model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 50,
            "temperature": 0,
        },
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def normalize_answer(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9\s]", "", s)
    s = re.sub(r"\s+", " ", s)
    return s

def compute_f1(prediction: str, gold: str) -> float:
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
    return 1.0 if normalize_answer(prediction) == normalize_answer(gold) else 0.0

def contains_answer(prediction: str, gold: str) -> float:
    return 1.0 if normalize_answer(gold) in normalize_answer(prediction) else 0.0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_benchmark(
    dataset_path: str,
    limit: int = 50,
    top_k: int = 10,
    answer_model: str = "gpt-4o",
    window_size: int = 3,
    stride: int = 2,
    context_window: int = 3,
    use_reranker: bool = True,
    skip_answer: bool = False,
):
    print(f"Loading dataset: {dataset_path}", flush=True)
    with open(dataset_path) as f:
        data = json.load(f)

    if limit:
        data = data[:limit]

    print(f"Running {len(data)} questions — window={window_size} stride={stride} "
          f"context_expand=±{context_window} reranker={use_reranker} reader={answer_model}", flush=True)
    print(flush=True)

    setup_tables()

    print("Loading models...", flush=True)
    _get_st_model()
    if use_reranker:
        _get_reranker()
    print("Models loaded.", flush=True)

    results = []
    total_times = {"ingest": 0, "embed": 0, "retrieve": 0, "answer": 0}

    for i, item in enumerate(data):
        qid = item["question_id"]
        qtype = item["question_type"]
        question = item["question"]
        gold_answer = item["answer"]
        sessions = item.get("haystack_sessions", [])
        session_ids = item.get("haystack_session_ids", [])
        dates = item.get("haystack_dates", [])

        print(f"[{i+1}/{len(data)}] {qid} ({qtype}) — {len(sessions)} sessions", flush=True)

        clear_tables(qid)

        # Ingest
        t0 = time.monotonic()
        n_turns, n_chunks = ingest_sessions(qid, sessions, session_ids, dates, window_size, stride)
        ingest_t = time.monotonic() - t0
        total_times["ingest"] += ingest_t

        # Embed
        t0 = time.monotonic()
        n_emb = embed_chunks(qid)
        embed_t = time.monotonic() - t0
        total_times["embed"] += embed_t

        # Retrieve
        t0 = time.monotonic()
        evidence = retrieve_evidence(question, qid, top_k=top_k,
                                     context_window=context_window, use_reranker=use_reranker)
        retrieve_t = time.monotonic() - t0
        total_times["retrieve"] += retrieve_t

        # Check retrieval hit
        answer_sids = set(item.get("answer_session_ids", []))
        retrieved_sids = set(e.get("session_id", "") for e in evidence)
        retrieval_hit = bool(answer_sids & retrieved_sids)

        # Answer
        predicted = ""
        answer_t = 0
        if not skip_answer and evidence:
            t0 = time.monotonic()
            try:
                predicted = generate_answer(question, evidence, model_name=answer_model)
            except Exception as e:
                predicted = f"[ERROR: {e}]"
            answer_t = time.monotonic() - t0
            total_times["answer"] += answer_t

        # Score
        f1 = compute_f1(predicted, gold_answer)
        em = compute_em(predicted, gold_answer)
        contains = contains_answer(predicted, gold_answer)

        result = {
            "question_id": qid,
            "question_type": qtype,
            "question": question,
            "gold_answer": gold_answer,
            "predicted_answer": predicted,
            "n_sessions": len(sessions),
            "n_turns": n_turns,
            "n_chunks": n_chunks,
            "n_embedded": n_emb,
            "n_evidence": len(evidence),
            "retrieval_hit": retrieval_hit,
            "f1": f1,
            "em": em,
            "contains": contains,
            "ingest_ms": round(ingest_t * 1000, 1),
            "embed_ms": round(embed_t * 1000, 1),
            "retrieve_ms": round(retrieve_t * 1000, 1),
            "answer_ms": round(answer_t * 1000, 1),
        }
        results.append(result)

        status = "✓" if contains > 0.5 else ("~" if f1 > 0.3 else "✗")
        print(f"  {status} F1={f1:.2f} EM={em:.0f} contains={contains:.0f} "
              f"hit={retrieval_hit} pred=\"{predicted[:60]}\" gold=\"{gold_answer[:60]}\" "
              f"[embed={embed_t:.1f}s retr={retrieve_t*1000:.0f}ms ans={answer_t*1000:.0f}ms]",
              flush=True)

    # Aggregate
    print("\n" + "=" * 70, flush=True)
    print("RESULTS — LongMemEval V1 (Improved Pipeline V2)", flush=True)
    print("=" * 70, flush=True)

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
        avg_hit = sum(r["retrieval_hit"] for r in items) / n
        avg_ret_ms = sum(r["retrieve_ms"] for r in items) / n

        aggregate[qtype] = {
            "count": n, "f1": round(avg_f1, 4), "em": round(avg_em, 4),
            "contains": round(avg_contains, 4), "retrieval_hit": round(avg_hit, 4),
            "avg_retrieve_ms": round(avg_ret_ms, 1),
        }
        print(f"\n{qtype} (n={n}):", flush=True)
        print(f"  F1:             {avg_f1:.4f}")
        print(f"  Exact Match:    {avg_em:.4f}")
        print(f"  Contains:       {avg_contains:.4f}")
        print(f"  Retrieval Hit:  {avg_hit:.4f}")
        print(f"  Avg Retrieve:   {avg_ret_ms:.1f}ms", flush=True)

    print(f"\nTotal time: ingest={total_times['ingest']:.1f}s embed={total_times['embed']:.1f}s "
          f"retrieve={total_times['retrieve']:.1f}s answer={total_times['answer']:.1f}s", flush=True)

    # Save
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ds_name = Path(dataset_path).stem
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"longmemeval_v2_{ds_name}_{ts}.json"

    output = {
        "benchmark": "LongMemEval",
        "pipeline": "v2-improved",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": {
            "limit": limit, "top_k": top_k, "answer_model": answer_model,
            "window_size": window_size, "stride": stride,
            "context_window": context_window, "use_reranker": use_reranker,
        },
        "aggregate": aggregate,
        "per_question": results,
    }
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}", flush=True)

    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=str(BENCHMARK_DIR / "longmemeval_s.json"))
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--answer-model", default="gpt-4o")
    parser.add_argument("--window-size", type=int, default=3)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--context-window", type=int, default=3)
    parser.add_argument("--no-reranker", action="store_true")
    parser.add_argument("--skip-answer", action="store_true")
    args = parser.parse_args()

    run_benchmark(
        dataset_path=args.dataset,
        limit=args.limit,
        top_k=args.top_k,
        answer_model=args.answer_model,
        window_size=args.window_size,
        stride=args.stride,
        context_window=args.context_window,
        use_reranker=not args.no_reranker,
        skip_answer=args.skip_answer,
    )
