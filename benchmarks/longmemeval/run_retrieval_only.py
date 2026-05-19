#!/usr/bin/env python3
"""
LongMemEval retrieval-only benchmark — apples-to-apples R@K comparison with MemPalace.

Reports R@1, R@3, R@5, R@8, R@10 session-level recall (no LLM calls).
Uses the same V3 hybrid pipeline (turn-level embed + session grouping).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import defaultdict, OrderedDict
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent.parent.parent / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

BENCHMARK_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BENCHMARK_DIR / "results"


def get_db_dsn():
    return os.environ.get("OPENCLAW_MEMORY_DB_DSN", "dbname=openclaw_memory")


_ST_MODEL = None
def _get_st_model():
    global _ST_MODEL
    if _ST_MODEL is None:
        os.environ["HF_HUB_OFFLINE"] = "1"
        from sentence_transformers import SentenceTransformer
        _ST_MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _ST_MODEL


def setup_tables():
    import psycopg
    with psycopg.connect(get_db_dsn()) as conn:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS lme3_turns (
                    id TEXT PRIMARY KEY,
                    text TEXT NOT NULL,
                    question_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    turn_index INT NOT NULL,
                    role TEXT,
                    session_date TEXT,
                    created_at TIMESTAMPTZ DEFAULT now()
                )
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS lme3_embeddings (
                    turn_id TEXT PRIMARY KEY REFERENCES lme3_turns(id),
                    embedding vector(384),
                    created_at TIMESTAMPTZ DEFAULT now()
                )
            """)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_lme3_turns_fts
                ON lme3_turns USING gin(to_tsvector('english', text))
            """)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_lme3_turns_session
                ON lme3_turns (question_id, session_id, turn_index)
            """)
        conn.commit()


def clear_tables(question_id: str | None = None):
    import psycopg
    with psycopg.connect(get_db_dsn()) as conn:
        with conn.cursor() as cur:
            if question_id:
                cur.execute("DELETE FROM lme3_embeddings WHERE turn_id IN (SELECT id FROM lme3_turns WHERE question_id = %s)", (question_id,))
                cur.execute("DELETE FROM lme3_turns WHERE question_id = %s", (question_id,))
            else:
                cur.execute("DELETE FROM lme3_embeddings")
                cur.execute("DELETE FROM lme3_turns")
        conn.commit()


def ingest_sessions(question_id: str, sessions: list[list[dict]], session_ids: list[str],
                    haystack_dates: list[str] | None = None) -> int:
    import psycopg
    items = []
    for s_idx, (session, s_id) in enumerate(zip(sessions, session_ids)):
        date = haystack_dates[s_idx] if haystack_dates and s_idx < len(haystack_dates) else ""
        for t_idx, turn in enumerate(session):
            content = turn.get("content", "").strip()
            if not content:
                continue
            items.append({
                "id": f"lme3-{question_id}-{s_id}-{t_idx}",
                "text": content,
                "question_id": question_id,
                "session_id": s_id,
                "turn_index": t_idx,
                "role": turn.get("role", "user"),
                "session_date": date,
            })

    if not items:
        return 0

    with psycopg.connect(get_db_dsn()) as conn:
        with conn.cursor() as cur:
            for item in items:
                cur.execute("""
                    INSERT INTO lme3_turns (id, text, question_id, session_id, turn_index, role, session_date)
                    VALUES (%(id)s, %(text)s, %(question_id)s, %(session_id)s, %(turn_index)s, %(role)s, %(session_date)s)
                    ON CONFLICT (id) DO NOTHING
                """, item)
        conn.commit()
    return len(items)


def embed_turns(question_id: str) -> int:
    import psycopg
    model = _get_st_model()

    with psycopg.connect(get_db_dsn()) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, text FROM lme3_turns WHERE question_id = %s AND id NOT IN (SELECT turn_id FROM lme3_embeddings)",
                (question_id,),
            )
            rows = cur.fetchall()

        if not rows:
            return 0

        ids = [r[0] for r in rows]
        texts = [r[1] for r in rows]
        embeddings = model.encode(texts, normalize_embeddings=True, batch_size=256, show_progress_bar=False)

        with conn.cursor() as cur:
            for turn_id, emb in zip(ids, embeddings):
                cur.execute(
                    "INSERT INTO lme3_embeddings (turn_id, embedding) VALUES (%s, %s) ON CONFLICT (turn_id) DO NOTHING",
                    (turn_id, emb.tolist()),
                )
        conn.commit()

    return len(rows)


def retrieve_ranked_sessions(question: str, question_id: str, top_k: int = 30) -> list[tuple[str, float]]:
    """
    Hybrid search on individual turns, group by session, return ranked session list.
    Returns list of (session_id, best_hybrid_score) sorted by score descending.
    """
    import psycopg
    model = _get_st_model()
    q_emb = model.encode(question, normalize_embeddings=True)

    with psycopg.connect(get_db_dsn()) as conn:
        with conn.cursor() as cur:
            cur.execute("""
                WITH vec AS (
                    SELECT e.turn_id, 1 - (e.embedding <=> %s::vector) AS vec_score
                    FROM lme3_embeddings e
                    JOIN lme3_turns t ON t.id = e.turn_id
                    WHERE t.question_id = %s
                    ORDER BY e.embedding <=> %s::vector
                    LIMIT %s
                ),
                fts AS (
                    SELECT t.id AS turn_id,
                           ts_rank_cd(to_tsvector('english', t.text), websearch_to_tsquery('english', %s)) AS fts_score
                    FROM lme3_turns t
                    WHERE t.question_id = %s
                      AND to_tsvector('english', t.text) @@ websearch_to_tsquery('english', %s)
                    ORDER BY fts_score DESC
                    LIMIT %s
                ),
                combined AS (
                    SELECT COALESCE(v.turn_id, f.turn_id) AS turn_id,
                           COALESCE(v.vec_score, 0) AS vec_score,
                           COALESCE(f.fts_score, 0) AS fts_score
                    FROM vec v
                    FULL OUTER JOIN fts f ON v.turn_id = f.turn_id
                )
                SELECT c.turn_id,
                       (0.6 * c.vec_score + 0.4 * c.fts_score) AS hybrid_score,
                       t.session_id
                FROM combined c
                JOIN lme3_turns t ON t.id = c.turn_id
                ORDER BY (0.6 * c.vec_score + 0.4 * c.fts_score) DESC
            """, (q_emb.tolist(), question_id, q_emb.tolist(), top_k * 3,
                  question, question_id, question, top_k * 3))

            hits = cur.fetchall()

    # Group by session, take best score per session
    session_scores: dict[str, float] = {}
    for _, score, sid in hits:
        if sid not in session_scores or score > session_scores[sid]:
            session_scores[sid] = score

    # Sort by score descending
    ranked = sorted(session_scores.items(), key=lambda x: x[1], reverse=True)
    return ranked


def run_benchmark(dataset_path: str, limit: int = 0, top_k: int = 10):
    print(f"Loading dataset: {dataset_path}", flush=True)
    with open(dataset_path) as f:
        data = json.load(f)

    if limit:
        data = data[:limit]

    print(f"Retrieval-only benchmark: {len(data)} questions", flush=True)
    print(f"  Hybrid search: top_k={top_k}, 0.6*vec + 0.4*fts", flush=True)
    print(f"  Embedding model: all-MiniLM-L6-v2 (384d)", flush=True)
    print(flush=True)

    setup_tables()
    print("Loading embedding model...", flush=True)
    _get_st_model()
    print("Ready.\n", flush=True)

    K_VALUES = [1, 3, 5, 8, 10, 20]
    results = []

    for i, item in enumerate(data):
        qid = item["question_id"]
        qtype = item["question_type"]
        question = item["question"]
        sessions = item.get("haystack_sessions", [])
        session_ids = item.get("haystack_session_ids", [])
        dates = item.get("haystack_dates", [])
        answer_sids = set(item.get("answer_session_ids", []))

        clear_tables(qid)

        t0 = time.monotonic()
        n_items = ingest_sessions(qid, sessions, session_ids, dates)
        ingest_t = time.monotonic() - t0

        t0 = time.monotonic()
        n_emb = embed_turns(qid)
        embed_t = time.monotonic() - t0

        t0 = time.monotonic()
        ranked_sessions = retrieve_ranked_sessions(question, qid, top_k=top_k)
        retrieve_t = time.monotonic() - t0

        # Compute R@K for each K
        recalls = {}
        for k in K_VALUES:
            top_k_sids = set(sid for sid, _ in ranked_sessions[:k])
            recalls[f"r@{k}"] = 1.0 if (answer_sids & top_k_sids) else 0.0

        result = {
            "question_id": qid,
            "question_type": qtype,
            "n_sessions": len(sessions),
            "n_turns": n_items,
            "n_answer_sessions": len(answer_sids),
            "ingest_ms": round(ingest_t * 1000, 1),
            "embed_ms": round(embed_t * 1000, 1),
            "retrieve_ms": round(retrieve_t * 1000, 1),
            **recalls,
        }
        results.append(result)

        elapsed = ingest_t + embed_t + retrieve_t
        hit5 = "✓" if recalls["r@5"] else "✗"
        print(f"[{i+1:3d}/{len(data)}] {hit5} R@5={recalls['r@5']:.0f} R@10={recalls['r@10']:.0f} "
              f"{qid} ({qtype}) — {n_items} turns, {elapsed:.1f}s",
              flush=True)

        # Clean up after each question to avoid DB bloat
        clear_tables(qid)

    # Aggregate
    print("\n" + "=" * 70, flush=True)
    print("RETRIEVAL RECALL — LongMemEval (Our Hybrid Pipeline)", flush=True)
    print("=" * 70, flush=True)

    by_type = defaultdict(list)
    for r in results:
        by_type[r["question_type"]].append(r)
        by_type["ALL"].append(r)

    aggregate = {}
    for qtype in sorted(by_type.keys()):
        items = by_type[qtype]
        n = len(items)
        metrics = {"count": n}
        for k in K_VALUES:
            key = f"r@{k}"
            metrics[key] = round(sum(r[key] for r in items) / n * 100, 1)
        metrics["avg_retrieve_ms"] = round(sum(r["retrieve_ms"] for r in items) / n, 1)
        aggregate[qtype] = metrics

        print(f"\n{qtype} (n={n}):", flush=True)
        for k in K_VALUES:
            key = f"r@{k}"
            print(f"  {key:8s} {metrics[key]:5.1f}%", flush=True)
        print(f"  {'avg_ms':8s} {metrics['avg_retrieve_ms']:.1f}ms", flush=True)

    # Comparison line
    all_m = aggregate.get("ALL", {})
    print(f"\n{'='*70}", flush=True)
    print(f"HEAD-TO-HEAD vs MemPalace (R@5):", flush=True)
    print(f"  Our system:   R@5 = {all_m.get('r@5', '?')}%", flush=True)
    print(f"  MemPalace:    R@5 = 96.6% (raw) / 98.4% (hybrid v4)", flush=True)
    print(f"{'='*70}", flush=True)

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ds_name = Path(dataset_path).stem
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"retrieval_only_{ds_name}_{ts}.json"

    output = {
        "benchmark": "LongMemEval",
        "pipeline": "v3-hybrid-retrieval-only",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": {"top_k": top_k, "n_questions": len(data),
                   "dataset": Path(dataset_path).name,
                   "embedding_model": "all-MiniLM-L6-v2",
                   "hybrid_weights": "0.6*vec + 0.4*fts"},
        "aggregate": aggregate,
        "per_question": results,
    }
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LongMemEval retrieval-only R@K benchmark")
    parser.add_argument("--dataset", default=str(BENCHMARK_DIR / "longmemeval_s_cleaned.json"))
    parser.add_argument("--limit", type=int, default=0, help="0 = all questions")
    parser.add_argument("--top-k", type=int, default=10, help="top_k for initial turn retrieval")
    args = parser.parse_args()

    run_benchmark(dataset_path=args.dataset, limit=args.limit, top_k=args.top_k)
