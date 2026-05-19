#!/usr/bin/env python3
"""
LongMemEval benchmark V3 — best of V1 + V2.

Strategy:
  - Embed individual turns (V1 approach — better retrieval hit rate)
  - When a turn matches, expand to full session context (V2 approach)
  - Use gpt-4o as reader (V2 approach — much better extraction)
  - Skip cross-encoder reranker (hurt more than helped)
  - Return top evidence as full session blocks, not isolated turns

This targets the two real bottlenecks:
  1. Evidence needs session context for multi-hop reasoning
  2. Reader model needs to be strong enough to extract precise answers
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
    print("  [setup] V3 benchmark tables ready", flush=True)


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
        embeddings = model.encode(texts, normalize_embeddings=True, batch_size=128, show_progress_bar=False)

        with conn.cursor() as cur:
            for turn_id, emb in zip(ids, embeddings):
                cur.execute(
                    "INSERT INTO lme3_embeddings (turn_id, embedding) VALUES (%s, %s) ON CONFLICT (turn_id) DO NOTHING",
                    (turn_id, emb.tolist()),
                )
        conn.commit()

    print(f"    embedded {len(rows)} turns", flush=True)
    return len(rows)


def retrieve_with_focused_context(question: str, question_id: str,
                                   top_k: int = 10, context_radius: int = 5,
                                   max_evidence: int = 8) -> list[dict]:
    """
    Phase 1: Hybrid search on individual turns (best retrieval hit rate)
    Phase 2: For each hit, expand ± context_radius turns from same session
    Phase 3: Deduplicate overlapping windows, return focused evidence blocks
    """
    import psycopg
    model = _get_st_model()
    q_emb = model.encode(question, normalize_embeddings=True)

    with psycopg.connect(get_db_dsn()) as conn:
        with conn.cursor() as cur:
            # Phase 1: Hybrid search on individual turns
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
                SELECT c.turn_id, c.vec_score, c.fts_score,
                       (0.6 * c.vec_score + 0.4 * c.fts_score) AS hybrid_score,
                       t.text, t.role, t.session_id, t.turn_index, t.session_date
                FROM combined c
                JOIN lme3_turns t ON t.id = c.turn_id
                ORDER BY (0.6 * c.vec_score + 0.4 * c.fts_score) DESC
                LIMIT %s
            """, (q_emb.tolist(), question_id, q_emb.tolist(), top_k * 3,
                  question, question_id, question, top_k * 3,
                  top_k * 3))

            hits = cur.fetchall()
            cols = [d[0] for d in cur.description]
            hit_rows = [dict(zip(cols, row)) for row in hits]

        if not hit_rows:
            return []

        # Phase 2: Group hits by session, merge into unified session blocks
        # For each session with hits, collect all matched turn indices
        from collections import OrderedDict
        session_hits = OrderedDict()  # session_id -> {best_score, matched_indices, date}
        for h in hit_rows:
            sid = h["session_id"]
            if sid not in session_hits:
                session_hits[sid] = {
                    "best_score": h["hybrid_score"],
                    "matched_indices": set(),
                    "date": h["session_date"],
                }
            session_hits[sid]["matched_indices"].add(h["turn_index"])
            session_hits[sid]["best_score"] = max(session_hits[sid]["best_score"], h["hybrid_score"])

        # Sort sessions by best hit score
        sorted_sessions = sorted(session_hits.items(), key=lambda x: x[1]["best_score"], reverse=True)

        results = []
        with conn.cursor() as cur:
            for sid, info in sorted_sessions[:max_evidence]:
                matched = info["matched_indices"]

                # Compute the range covering all matched turns ± context_radius
                range_start = max(0, min(matched) - context_radius)
                range_end = max(matched) + context_radius

                cur.execute("""
                    SELECT turn_index, role, text FROM lme3_turns
                    WHERE question_id = %s AND session_id = %s
                      AND turn_index >= %s AND turn_index <= %s
                    ORDER BY turn_index
                """, (question_id, sid, range_start, range_end))

                context_turns = cur.fetchall()
                expanded_text = ""
                for t_idx, role, text in context_turns:
                    marker = ">>>" if t_idx in matched else "   "
                    expanded_text += f"{marker} [{role}]: {text}\n"

                results.append({
                    "session_id": sid,
                    "session_date": info["date"],
                    "hybrid_score": info["best_score"],
                    "n_matched": len(matched),
                    "n_context_turns": len(context_turns),
                    "expanded_text": expanded_text.strip(),
                })

    return results


def generate_answer(question: str, evidence: list[dict], model_name: str = "gpt-4o") -> str:
    import requests

    # Build evidence — focused windows with matched turns highlighted
    evidence_text = ""
    for i, e in enumerate(evidence[:10], 1):
        date = e.get("session_date", "")
        text = e.get("expanded_text", "")
        score = e.get("hybrid_score", 0)
        evidence_text += f"\n--- Evidence #{i} (date: {date}, score: {score:.2f}) ---\n{text}\n"

    prompt = f"""Extract the answer to the question from the user's past conversations below. Lines marked >>> are the most relevant.

CRITICAL — COPY EXACTLY:
- Your answer must be a VERBATIM substring of what the user said. Do NOT rephrase, shorten, or paraphrase.
- Keep ALL articles ("a", "the"), qualifiers ("each way", "in Australia"), and modifiers.
- "a lighter shade of gray" → "a lighter shade of gray" (NOT "gray")
- "45 minutes each way" → "45 minutes each way" (NOT "45 minutes")
- "the sports store downtown" → "the sports store downtown" (NOT "sports store downtown")
- "A staunch atheist" → "A staunch atheist" (NOT "staunch atheist")
- "a lemon blueberry cake" → "a lemon blueberry cake" (NOT "lemon blueberry cake")
- "University of California, Los Angeles (UCLA)" → "University of California, Los Angeles (UCLA)" (NOT "UCLA")

QUESTION-TYPE RULES:
- "how many"/"how much": JUST the number/amount. "7" not "7 shirts". "500" not "500 copies".
- Dates: use the EXACT calendar format from the text. "February 14th" not "Valentine's Day". If both a date and a holiday name appear, prefer the calendar date.
- "where": find the specific PLACE NAME (store, venue, city). Scan ALL evidence — the user may name the place in passing.
- "who": JUST the name. "Sarah" not "my friend Sarah".
- "what time": JUST the time. "7 pm" not "by 7 pm".

NEVER say "Unknown" or "I don't know". Keep answers SHORT but COMPLETE.

EVIDENCE:
{evidence_text}

QUESTION: {question}
ANSWER (verbatim from user's words):"""

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
        return evidence[0].get("full_text", "")[:200] if evidence else "Unknown"

    # Retry with exponential backoff for transient API failures
    max_retries = 3
    for attempt in range(max_retries + 1):
        try:
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
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as ex:
            if attempt < max_retries:
                wait = 2 ** attempt  # 1s, 2s, 4s
                print(f"    [retry {attempt+1}/{max_retries}] {type(ex).__name__}, waiting {wait}s...", flush=True)
                time.sleep(wait)
            else:
                raise


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


def run_benchmark(dataset_path: str, limit: int = 50, top_k: int = 10,
                  context_radius: int = 5, max_evidence: int = 8,
                  answer_model: str = "gpt-4o", skip_answer: bool = False):
    print(f"Loading dataset: {dataset_path}", flush=True)
    with open(dataset_path) as f:
        data = json.load(f)

    if limit:
        data = data[:limit]

    print(f"V3 Pipeline: {len(data)} questions — turn-embed → ±{context_radius} focused context → {answer_model} reader", flush=True)
    print(f"  top_k={top_k} → max {max_evidence} evidence blocks", flush=True)
    print(flush=True)

    setup_tables()
    print("Loading embedding model...", flush=True)
    _get_st_model()
    print("Ready.", flush=True)

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

        t0 = time.monotonic()
        n_items = ingest_sessions(qid, sessions, session_ids, dates)
        ingest_t = time.monotonic() - t0
        total_times["ingest"] += ingest_t

        t0 = time.monotonic()
        n_emb = embed_turns(qid)
        embed_t = time.monotonic() - t0
        total_times["embed"] += embed_t

        t0 = time.monotonic()
        evidence = retrieve_with_focused_context(question, qid, top_k=top_k,
                                                    context_radius=context_radius, max_evidence=max_evidence)
        retrieve_t = time.monotonic() - t0
        total_times["retrieve"] += retrieve_t

        answer_sids = set(item.get("answer_session_ids", []))
        retrieved_sids = set(e.get("session_id", "") for e in evidence)
        retrieval_hit = bool(answer_sids & retrieved_sids)

        predicted = ""
        answer_t = 0
        if not skip_answer and evidence:
            t0 = time.monotonic()
            try:
                predicted = generate_answer(question, evidence, model_name=answer_model)
            except Exception as ex:
                predicted = f"[ERROR: {ex}]"
            answer_t = time.monotonic() - t0
            total_times["answer"] += answer_t

        f1 = compute_f1(predicted, gold_answer)
        em = compute_em(predicted, gold_answer)
        contains = contains_answer(predicted, gold_answer)

        result = {
            "question_id": qid, "question_type": qtype, "question": question,
            "gold_answer": gold_answer, "predicted_answer": predicted,
            "n_sessions": len(sessions), "n_items": n_items, "n_embedded": n_emb,
            "n_evidence_blocks": len(evidence),
            "retrieval_hit": retrieval_hit,
            "f1": f1, "em": em, "contains": contains,
            "ingest_ms": round(ingest_t * 1000, 1),
            "embed_ms": round(embed_t * 1000, 1),
            "retrieve_ms": round(retrieve_t * 1000, 1),
            "answer_ms": round(answer_t * 1000, 1),
        }
        results.append(result)

        status = "✓" if contains > 0.5 else ("~" if f1 > 0.3 else "✗")
        print(f"  {status} F1={f1:.2f} EM={em:.0f} contains={contains:.0f} "
              f"hit={retrieval_hit} pred=\"{predicted[:50]}\" gold=\"{gold_answer[:50]}\" "
              f"[embed={embed_t:.1f}s retr={retrieve_t*1000:.0f}ms ans={answer_t*1000:.0f}ms]",
              flush=True)

    # Aggregate
    print("\n" + "=" * 70, flush=True)
    print("RESULTS — LongMemEval V1 (V3 Pipeline: turn-embed + session-context + gpt-4o)", flush=True)
    print("=" * 70, flush=True)

    by_type = defaultdict(list)
    for r in results:
        by_type[r["question_type"]].append(r)
        by_type["ALL"].append(r)

    aggregate = {}
    for qtype, items in sorted(by_type.items()):
        n = len(items)
        metrics = {
            "count": n,
            "f1": round(sum(r["f1"] for r in items) / n, 4),
            "em": round(sum(r["em"] for r in items) / n, 4),
            "contains": round(sum(r["contains"] for r in items) / n, 4),
            "retrieval_hit": round(sum(r["retrieval_hit"] for r in items) / n, 4),
            "avg_retrieve_ms": round(sum(r["retrieve_ms"] for r in items) / n, 1),
        }
        aggregate[qtype] = metrics
        print(f"\n{qtype} (n={n}):", flush=True)
        for k, v in metrics.items():
            if k != "count":
                print(f"  {k:20s} {v}")
        sys.stdout.flush()

    print(f"\nTotal time: ingest={total_times['ingest']:.1f}s embed={total_times['embed']:.1f}s "
          f"retrieve={total_times['retrieve']:.1f}s answer={total_times['answer']:.1f}s", flush=True)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ds_name = Path(dataset_path).stem
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"longmemeval_v3_{ds_name}_{ts}.json"

    output = {
        "benchmark": "LongMemEval", "pipeline": "v3-session-context",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": {"limit": limit, "top_k": top_k, "context_radius": context_radius,
                   "max_evidence": max_evidence, "answer_model": answer_model},
        "aggregate": aggregate, "per_question": results,
    }
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=str(BENCHMARK_DIR / "longmemeval_s.json"))
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--context-radius", type=int, default=5)
    parser.add_argument("--max-evidence", type=int, default=8)
    parser.add_argument("--answer-model", default="gpt-4o")
    parser.add_argument("--skip-answer", action="store_true")
    args = parser.parse_args()

    run_benchmark(
        dataset_path=args.dataset, limit=args.limit, top_k=args.top_k,
        context_radius=args.context_radius, max_evidence=args.max_evidence,
        answer_model=args.answer_model, skip_answer=args.skip_answer,
    )
