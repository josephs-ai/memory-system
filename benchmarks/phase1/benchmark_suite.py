#!/usr/bin/env python3
"""
benchmark_suite.py — Expanded benchmark runner for Phase 1.

Runs the full LongMemEval dataset (500 questions, 6 query classes) with:
- Per-class metric breakdown
- Bootstrap confidence intervals
- Multiple run support for stability
- Configurable retrieval pipeline (for ablation integration)
- JSON + markdown output

Usage:
    python benchmark_suite.py run --dataset ../longmemeval/longmemeval_s.json
    python benchmark_suite.py run --dataset ../longmemeval/longmemeval_s.json --limit 100
    python benchmark_suite.py run --runs 3  # Multiple runs for CI
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

PHASE1_DIR = Path(__file__).resolve().parent
BENCHMARK_DIR = PHASE1_DIR.parent
LONGMEMEVAL_DIR = BENCHMARK_DIR / "longmemeval"
RESULTS_DIR = PHASE1_DIR / "results"
SCRIPT_DIR = BENCHMARK_DIR.parent / "scripts"

sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(PHASE1_DIR))

from statistical_utils import bootstrap_ci, aggregate_metrics


# ---------------------------------------------------------------------------
# Retrieval pipeline configuration (feature toggles for ablations)
# ---------------------------------------------------------------------------

class PipelineConfig:
    """Controls which retrieval components are active."""

    def __init__(
        self,
        name: str = "full",
        use_vector: bool = True,
        use_fts: bool = True,
        use_rerank: bool = True,
        use_graph: bool = False,       # Graph not used in LongMemEval (no pre-loaded graph)
        use_temporal: bool = True,
        use_feedback: bool = True,
        use_lifecycle_filter: bool = True,
        vector_weight: float = 0.6,
        fts_weight: float = 0.4,
        top_k: int = 10,
        context_radius: int = 5,
        max_evidence: int = 8,
        answer_model: str = "gpt-4o",
    ):
        self.name = name
        self.use_vector = use_vector
        self.use_fts = use_fts
        self.use_rerank = use_rerank
        self.use_graph = use_graph
        self.use_temporal = use_temporal
        self.use_feedback = use_feedback
        self.use_lifecycle_filter = use_lifecycle_filter
        self.vector_weight = vector_weight
        self.fts_weight = fts_weight
        self.top_k = top_k
        self.context_radius = context_radius
        self.max_evidence = max_evidence
        self.answer_model = answer_model

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "use_vector": self.use_vector,
            "use_fts": self.use_fts,
            "use_rerank": self.use_rerank,
            "use_graph": self.use_graph,
            "use_temporal": self.use_temporal,
            "use_feedback": self.use_feedback,
            "use_lifecycle_filter": self.use_lifecycle_filter,
            "vector_weight": self.vector_weight,
            "fts_weight": self.fts_weight,
            "top_k": self.top_k,
            "context_radius": self.context_radius,
            "max_evidence": self.max_evidence,
            "answer_model": self.answer_model,
        }


# ---------------------------------------------------------------------------
# Database and embedding helpers (shared with longmemeval v3)
# ---------------------------------------------------------------------------

def get_db_dsn():
    return os.environ.get("OPENCLAW_MEMORY_DB_DSN", "dbname=openclaw_memory")


_ST_MODEL = None
def _get_st_model():
    global _ST_MODEL
    if _ST_MODEL is None:
        from sentence_transformers import SentenceTransformer
        _ST_MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _ST_MODEL


_RERANKER = None
def _get_reranker():
    global _RERANKER
    if _RERANKER is None:
        from sentence_transformers import CrossEncoder
        _RERANKER = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _RERANKER


def setup_tables():
    import psycopg
    with psycopg.connect(get_db_dsn()) as conn:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS bench_turns (
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
                CREATE TABLE IF NOT EXISTS bench_embeddings (
                    turn_id TEXT PRIMARY KEY REFERENCES bench_turns(id),
                    embedding vector(384),
                    created_at TIMESTAMPTZ DEFAULT now()
                )
            """)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_bench_turns_fts
                ON bench_turns USING gin(to_tsvector('english', text))
            """)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_bench_turns_session
                ON bench_turns (question_id, session_id, turn_index)
            """)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_bench_turns_qid
                ON bench_turns (question_id)
            """)
        conn.commit()
    print("  [setup] Benchmark tables ready", flush=True)


def clear_question(question_id: str):
    import psycopg
    with psycopg.connect(get_db_dsn()) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM bench_embeddings WHERE turn_id IN (SELECT id FROM bench_turns WHERE question_id = %s)",
                (question_id,),
            )
            cur.execute("DELETE FROM bench_turns WHERE question_id = %s", (question_id,))
        conn.commit()


def ingest_sessions(question_id: str, sessions: list, session_ids: list, dates: list | None = None) -> int:
    import psycopg
    items = []
    for s_idx, (session, s_id) in enumerate(zip(sessions, session_ids)):
        date = dates[s_idx] if dates and s_idx < len(dates) else ""
        for t_idx, turn in enumerate(session):
            content = turn.get("content", "").strip()
            if not content:
                continue
            items.append({
                "id": f"bench-{question_id}-{s_id}-{t_idx}",
                "text": content,
                "question_id": question_id,
                "session_id": s_id,
                "turn_index": t_idx,
                "role": turn.get("role", "user"),
                "session_date": date,
            })
    if not items:
        return 0

    import psycopg
    with psycopg.connect(get_db_dsn()) as conn:
        with conn.cursor() as cur:
            for item in items:
                cur.execute("""
                    INSERT INTO bench_turns (id, text, question_id, session_id, turn_index, role, session_date)
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
                "SELECT id, text FROM bench_turns WHERE question_id = %s AND id NOT IN (SELECT turn_id FROM bench_embeddings)",
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
                    "INSERT INTO bench_embeddings (turn_id, embedding) VALUES (%s, %s) ON CONFLICT (turn_id) DO NOTHING",
                    (turn_id, emb.tolist()),
                )
        conn.commit()
    return len(rows)


# ---------------------------------------------------------------------------
# Configurable retrieval
# ---------------------------------------------------------------------------

def retrieve(query: str, question_id: str, config: PipelineConfig) -> list[dict]:
    """
    Configurable hybrid retrieval with ablation support.

    Supports toggling: vector, FTS, rerank, temporal scoring.
    Graph and lifecycle are controlled but N/A for LongMemEval (no pre-loaded items).
    """
    import psycopg
    from datetime import datetime as _dt

    results_by_session: dict[str, dict] = {}

    with psycopg.connect(get_db_dsn()) as conn:
        with conn.cursor() as cur:
            candidates: list[dict] = []

            # --- Vector search ---
            if config.use_vector:
                model = _get_st_model()
                q_emb = model.encode(query, normalize_embeddings=True)
                cur.execute("""
                    SELECT e.turn_id, 1 - (e.embedding <=> %s::vector) AS vec_score
                    FROM bench_embeddings e
                    JOIN bench_turns t ON t.id = e.turn_id
                    WHERE t.question_id = %s
                    ORDER BY e.embedding <=> %s::vector
                    LIMIT %s
                """, (q_emb.tolist(), question_id, q_emb.tolist(), config.top_k * 3))
                for turn_id, vec_score in cur.fetchall():
                    candidates.append({"turn_id": turn_id, "vec_score": float(vec_score), "fts_score": 0.0})

            # --- FTS search ---
            if config.use_fts:
                cur.execute("""
                    SELECT t.id AS turn_id,
                           ts_rank_cd(to_tsvector('english', t.text), websearch_to_tsquery('english', %s)) AS fts_score
                    FROM bench_turns t
                    WHERE t.question_id = %s
                      AND to_tsvector('english', t.text) @@ websearch_to_tsquery('english', %s)
                    ORDER BY fts_score DESC
                    LIMIT %s
                """, (query, question_id, query, config.top_k * 3))
                fts_map = {turn_id: float(score) for turn_id, score in cur.fetchall()}

                # Merge FTS scores into existing candidates or add new ones
                existing_ids = {c["turn_id"] for c in candidates}
                for c in candidates:
                    if c["turn_id"] in fts_map:
                        c["fts_score"] = fts_map[c["turn_id"]]

                for turn_id, fts_score in fts_map.items():
                    if turn_id not in existing_ids:
                        candidates.append({"turn_id": turn_id, "vec_score": 0.0, "fts_score": fts_score})

            if not candidates:
                return []

            # --- Compute hybrid scores ---
            for c in candidates:
                c["hybrid_score"] = (
                    config.vector_weight * c["vec_score"]
                    + config.fts_weight * c["fts_score"]
                )

            # --- Temporal scoring ---
            if config.use_temporal:
                try:
                    from temporal_scoring import compute_temporal_boost, is_temporal_query
                    is_temporal = is_temporal_query(query)
                    now = _dt.now(tz=__import__("datetime").timezone.utc)

                    # Fetch dates for temporal scoring
                    turn_ids = [c["turn_id"] for c in candidates]
                    placeholders = ",".join(["%s"] * len(turn_ids))
                    cur.execute(f"""
                        SELECT id, session_date FROM bench_turns
                        WHERE id IN ({placeholders})
                    """, turn_ids)
                    date_map = {row[0]: row[1] for row in cur.fetchall()}

                    for c in candidates:
                        date_str = date_map.get(c["turn_id"], "")
                        if date_str:
                            try:
                                item_dt = _dt.fromisoformat(date_str.replace("Z", "+00:00"))
                                boost = compute_temporal_boost(now, item_dt, is_temporal)
                                c["hybrid_score"] += boost
                                c["temporal_boost"] = boost
                            except (ValueError, TypeError):
                                c["temporal_boost"] = 0.0
                        else:
                            c["temporal_boost"] = 0.0
                except ImportError:
                    pass  # temporal_scoring not available

            # --- Sort by hybrid score ---
            candidates.sort(key=lambda x: x["hybrid_score"], reverse=True)

            # --- Reranking ---
            if config.use_rerank:
                reranker = _get_reranker()
                rerank_pool = candidates[:config.top_k * 2]

                # Fetch turn texts for reranking
                turn_ids = [c["turn_id"] for c in rerank_pool]
                placeholders = ",".join(["%s"] * len(turn_ids))
                cur.execute(f"SELECT id, text FROM bench_turns WHERE id IN ({placeholders})", turn_ids)
                text_map = {row[0]: row[1] for row in cur.fetchall()}

                pairs = [(query, text_map.get(c["turn_id"], "")[:1000]) for c in rerank_pool]
                scores = reranker.predict(pairs)

                for c, score in zip(rerank_pool, scores):
                    c["rerank_score"] = float(score)

                rerank_pool.sort(key=lambda x: x.get("rerank_score", -1e9), reverse=True)
                candidates = rerank_pool

            # --- Top-K selection ---
            top_candidates = candidates[:config.top_k * 2]

            # --- Context expansion (same session window approach as V3) ---
            # Fetch turn metadata for session grouping
            turn_ids = [c["turn_id"] for c in top_candidates]
            if not turn_ids:
                return []

            placeholders = ",".join(["%s"] * len(turn_ids))
            cur.execute(f"""
                SELECT id, session_id, turn_index, session_date
                FROM bench_turns WHERE id IN ({placeholders})
            """, turn_ids)
            meta_map = {row[0]: {"session_id": row[1], "turn_index": row[2], "session_date": row[3]} for row in cur.fetchall()}

            # Group by session, keep best score
            from collections import OrderedDict
            session_hits: dict[str, dict] = OrderedDict()
            for c in top_candidates:
                meta = meta_map.get(c["turn_id"])
                if not meta:
                    continue
                sid = meta["session_id"]
                score = c.get("rerank_score", c["hybrid_score"])
                if sid not in session_hits:
                    session_hits[sid] = {
                        "best_score": score,
                        "matched_indices": set(),
                        "date": meta["session_date"],
                    }
                session_hits[sid]["matched_indices"].add(meta["turn_index"])
                session_hits[sid]["best_score"] = max(session_hits[sid]["best_score"], score)

            # Sort sessions by best score
            sorted_sessions = sorted(session_hits.items(), key=lambda x: x[1]["best_score"], reverse=True)

            results = []
            for sid, info in sorted_sessions[:config.max_evidence]:
                matched = info["matched_indices"]
                range_start = max(0, min(matched) - config.context_radius)
                range_end = max(matched) + config.context_radius

                cur.execute("""
                    SELECT turn_index, role, text FROM bench_turns
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


# ---------------------------------------------------------------------------
# Answer generation (same as V3, shared)
# ---------------------------------------------------------------------------

def get_openai_key() -> str:
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
    return api_key


def generate_answer(question: str, evidence: list[dict], model_name: str = "gpt-4o") -> str:
    import requests

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

QUESTION-TYPE RULES:
- "how many"/"how much": JUST the number/amount.
- Dates: use the EXACT calendar format from the text.
- "where": find the specific PLACE NAME.
- "who": JUST the name.
- "what time": JUST the time.

NEVER say "Unknown" or "I don't know". Keep answers SHORT but COMPLETE.

EVIDENCE:
{evidence_text}

QUESTION: {question}
ANSWER (verbatim from user's words):"""

    api_key = get_openai_key()
    if not api_key:
        return evidence[0].get("expanded_text", "")[:200] if evidence else "Unknown"

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
        except Exception as ex:
            if attempt < max_retries:
                time.sleep(2 ** attempt)
            else:
                return f"[ERROR: {ex}]"

    return "Unknown"


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
# Main benchmark runner
# ---------------------------------------------------------------------------

def run_benchmark(
    dataset_path: str,
    config: PipelineConfig,
    limit: int | None = None,
    skip_answer: bool = False,
    run_id: int = 1,
) -> dict:
    """Run the benchmark and return structured results."""
    print(f"\n{'='*70}", flush=True)
    print(f"Benchmark Run {run_id} — Pipeline: {config.name}", flush=True)
    print(f"{'='*70}", flush=True)

    with open(dataset_path) as f:
        data = json.load(f)

    if limit:
        data = data[:limit]

    print(f"Dataset: {dataset_path} ({len(data)} questions)", flush=True)
    print(f"Config: {json.dumps(config.to_dict(), indent=2)}", flush=True)

    setup_tables()
    print("Loading models...", flush=True)
    if config.use_vector:
        _get_st_model()
    if config.use_rerank:
        _get_reranker()
    print("Ready.\n", flush=True)

    per_question = []
    total_times = {"ingest": 0.0, "embed": 0.0, "retrieve": 0.0, "answer": 0.0}

    for i, item in enumerate(data):
        qid = item["question_id"]
        qtype = item["question_type"]
        question = item["question"]
        gold_answer = item["answer"]
        sessions = item.get("haystack_sessions", [])
        session_ids = item.get("haystack_session_ids", [])
        dates = item.get("haystack_dates", [])

        print(f"[{i+1}/{len(data)}] {qid} ({qtype})", end=" ", flush=True)

        clear_question(qid)

        t0 = time.monotonic()
        n_items = ingest_sessions(qid, sessions, session_ids, dates)
        ingest_t = time.monotonic() - t0
        total_times["ingest"] += ingest_t

        t0 = time.monotonic()
        n_emb = embed_turns(qid) if config.use_vector else 0
        embed_t = time.monotonic() - t0
        total_times["embed"] += embed_t

        t0 = time.monotonic()
        evidence = retrieve(question, qid, config)
        retrieve_t = time.monotonic() - t0
        total_times["retrieve"] += retrieve_t

        answer_sids = set(item.get("answer_session_ids", []))
        retrieved_sids = set(e.get("session_id", "") for e in evidence)
        retrieval_hit = bool(answer_sids & retrieved_sids)

        predicted = ""
        answer_t = 0.0
        if not skip_answer and evidence:
            t0 = time.monotonic()
            predicted = generate_answer(question, evidence, model_name=config.answer_model)
            answer_t = time.monotonic() - t0
            total_times["answer"] += answer_t

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
            "n_items": n_items,
            "retrieval_hit": retrieval_hit,
            "n_evidence": len(evidence),
            "f1": f1,
            "em": em,
            "contains": contains,
            "retrieve_ms": round(retrieve_t * 1000, 1),
            "answer_ms": round(answer_t * 1000, 1),
        }
        per_question.append(result)

        status = "✓" if contains > 0.5 else ("~" if f1 > 0.3 else "✗")
        print(f"{status} F1={f1:.2f} hit={retrieval_hit} [{retrieve_t*1000:.0f}ms]", flush=True)

    # Aggregate with confidence intervals
    metric_keys = ["f1", "em", "contains", "retrieval_hit", "retrieve_ms"]
    aggregated = aggregate_metrics(per_question, metric_keys, group_key="question_type")

    # Print summary
    print(f"\n{'='*70}", flush=True)
    print(f"RESULTS — {config.name} (Run {run_id})", flush=True)
    print(f"{'='*70}", flush=True)

    for group in sorted(aggregated.keys()):
        metrics = aggregated[group]
        n = metrics["f1"].n
        print(f"\n{group} (n={n}):", flush=True)
        for mk in metric_keys:
            m = metrics[mk]
            print(f"  {mk:20s} {m.mean:.4f}  [{m.ci_lower:.4f}, {m.ci_upper:.4f}]  (std={m.std:.4f})", flush=True)

    print(f"\nTotal time: ingest={total_times['ingest']:.1f}s embed={total_times['embed']:.1f}s "
          f"retrieve={total_times['retrieve']:.1f}s answer={total_times['answer']:.1f}s", flush=True)

    # Build output
    output = {
        "benchmark": "LongMemEval",
        "pipeline": config.name,
        "run_id": run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": config.to_dict(),
        "aggregate": {
            group: {mk: metrics[mk].to_dict() for mk in metric_keys}
            for group, metrics in aggregated.items()
        },
        "total_times": {k: round(v, 2) for k, v in total_times.items()},
        "per_question": per_question,
    }

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"{config.name}_run{run_id}_{ts}.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}", flush=True)

    return output


def run_multi(
    dataset_path: str,
    config: PipelineConfig,
    n_runs: int = 1,
    limit: int | None = None,
    skip_answer: bool = False,
) -> list[dict]:
    """Run the benchmark multiple times and return all results."""
    all_results = []
    for run_id in range(1, n_runs + 1):
        result = run_benchmark(dataset_path, config, limit=limit, skip_answer=skip_answer, run_id=run_id)
        all_results.append(result)
    return all_results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Phase 1 Benchmark Suite")
    sub = parser.add_subparsers(dest="command")

    run_p = sub.add_parser("run", help="Run benchmark")
    run_p.add_argument("--dataset", default=str(LONGMEMEVAL_DIR / "longmemeval_s.json"))
    run_p.add_argument("--limit", type=int, default=None, help="Limit questions (None = all 500)")
    run_p.add_argument("--runs", type=int, default=1, help="Number of repeated runs")
    run_p.add_argument("--skip-answer", action="store_true", help="Skip LLM answer generation (retrieval-only)")
    run_p.add_argument("--pipeline", default="full", help="Pipeline name")
    run_p.add_argument("--no-vector", action="store_true")
    run_p.add_argument("--no-fts", action="store_true")
    run_p.add_argument("--no-rerank", action="store_true")
    run_p.add_argument("--no-temporal", action="store_true")
    run_p.add_argument("--answer-model", default="gpt-4o")
    run_p.add_argument("--top-k", type=int, default=10)

    args = parser.parse_args()

    if args.command == "run":
        config = PipelineConfig(
            name=args.pipeline,
            use_vector=not args.no_vector,
            use_fts=not args.no_fts,
            use_rerank=not args.no_rerank,
            use_temporal=not args.no_temporal,
            answer_model=args.answer_model,
            top_k=args.top_k,
        )
        run_multi(args.dataset, config, n_runs=args.runs, limit=args.limit, skip_answer=args.skip_answer)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
