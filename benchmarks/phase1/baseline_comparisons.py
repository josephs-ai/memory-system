#!/usr/bin/env python3
"""
baseline_comparisons.py — Compare the full pipeline against simple baselines.

Baselines tested:
1. Random retrieval (lower bound)
2. Recency-only (most recent sessions)
3. BM25/FTS-only (no embeddings, no rerank)
4. Vector-only (embeddings without FTS, no rerank)
5. Hybrid without rerank
6. Full pipeline

This answers the critical question: "Is the complex system actually better
than a simpler approach?"

Usage:
    python baseline_comparisons.py run
    python baseline_comparisons.py run --limit 50 --skip-answer
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PHASE1_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PHASE1_DIR))

from benchmark_suite import (
    PipelineConfig, run_benchmark, clear_question, ingest_sessions,
    embed_turns, get_db_dsn, normalize_answer, compute_f1, compute_em,
    contains_answer, generate_answer, get_openai_key, setup_tables,
    _get_st_model, LONGMEMEVAL_DIR, RESULTS_DIR,
)
from statistical_utils import bootstrap_ci, permutation_test, cohens_d


# ---------------------------------------------------------------------------
# Random baseline (retrieves random sessions)
# ---------------------------------------------------------------------------

def retrieve_random(question_id: str, n_sessions: int, max_evidence: int = 8, seed: int = 42) -> list[dict]:
    """Return random session blocks as evidence."""
    import psycopg

    rng = random.Random(seed)

    with psycopg.connect(get_db_dsn()) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT DISTINCT session_id FROM bench_turns WHERE question_id = %s",
                (question_id,),
            )
            all_sessions = [row[0] for row in cur.fetchall()]

            if not all_sessions:
                return []

            selected = rng.sample(all_sessions, min(max_evidence, len(all_sessions)))
            results = []

            for sid in selected:
                cur.execute("""
                    SELECT turn_index, role, text FROM bench_turns
                    WHERE question_id = %s AND session_id = %s
                    ORDER BY turn_index
                    LIMIT 20
                """, (question_id, sid))

                turns = cur.fetchall()
                text = "\n".join(f"   [{role}]: {t}" for _, role, t in turns)
                results.append({
                    "session_id": sid,
                    "session_date": "",
                    "hybrid_score": 0.0,
                    "n_matched": 0,
                    "n_context_turns": len(turns),
                    "expanded_text": text,
                })

    return results


# ---------------------------------------------------------------------------
# Recency baseline (most recent sessions)
# ---------------------------------------------------------------------------

def retrieve_recency(question_id: str, max_evidence: int = 8) -> list[dict]:
    """Return the most recent sessions as evidence (by session_date)."""
    import psycopg

    with psycopg.connect(get_db_dsn()) as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT DISTINCT session_id, session_date
                FROM bench_turns
                WHERE question_id = %s AND session_date IS NOT NULL AND session_date != ''
                ORDER BY session_date DESC
                LIMIT %s
            """, (question_id, max_evidence))

            sessions = cur.fetchall()
            results = []

            for sid, date in sessions:
                cur.execute("""
                    SELECT turn_index, role, text FROM bench_turns
                    WHERE question_id = %s AND session_id = %s
                    ORDER BY turn_index
                    LIMIT 20
                """, (question_id, sid))

                turns = cur.fetchall()
                text = "\n".join(f"   [{role}]: {t}" for _, role, t in turns)
                results.append({
                    "session_id": sid,
                    "session_date": date,
                    "hybrid_score": 0.0,
                    "n_matched": 0,
                    "n_context_turns": len(turns),
                    "expanded_text": text,
                })

    return results


# ---------------------------------------------------------------------------
# Baseline runner
# ---------------------------------------------------------------------------

BASELINE_CONFIGS = {
    "random": {"description": "Random session retrieval (lower bound)"},
    "recency": {"description": "Most recent sessions (no search)"},
    "fts_only": PipelineConfig(name="fts_only", use_vector=False, use_fts=True, use_rerank=False, use_temporal=False),
    "vector_only": PipelineConfig(name="vector_only", use_vector=True, use_fts=False, use_rerank=False, use_temporal=False),
    "hybrid_no_rerank": PipelineConfig(name="hybrid_no_rerank", use_vector=True, use_fts=True, use_rerank=False, use_temporal=False),
    "full": PipelineConfig(name="full", use_vector=True, use_fts=True, use_rerank=True, use_temporal=True),
}


def run_baselines(
    dataset_path: str,
    limit: int | None = None,
    skip_answer: bool = False,
    baselines: list[str] | None = None,
) -> dict[str, dict]:
    """Run all baselines and collect results."""
    from benchmark_suite import retrieve

    names = baselines or list(BASELINE_CONFIGS.keys())

    with open(dataset_path) as f:
        data = json.load(f)
    if limit:
        data = data[:limit]

    print(f"Running {len(names)} baselines on {len(data)} questions\n", flush=True)

    setup_tables()
    if any(isinstance(BASELINE_CONFIGS.get(n), PipelineConfig) and BASELINE_CONFIGS[n].use_vector for n in names):
        _get_st_model()

    # Pre-ingest all questions once
    print("Pre-ingesting all questions...", flush=True)
    for i, item in enumerate(data):
        qid = item["question_id"]
        clear_question(qid)
        ingest_sessions(qid, item.get("haystack_sessions", []),
                       item.get("haystack_session_ids", []),
                       item.get("haystack_dates", []))
        embed_turns(qid)
        if (i + 1) % 50 == 0:
            print(f"  ingested {i+1}/{len(data)}", flush=True)
    print("Ingestion complete.\n", flush=True)

    all_results = {}

    for baseline_name in names:
        print(f"\n{'='*60}", flush=True)
        print(f"Baseline: {baseline_name}", flush=True)
        print(f"{'='*60}", flush=True)

        config = BASELINE_CONFIGS.get(baseline_name)
        per_question = []

        for i, item in enumerate(data):
            qid = item["question_id"]
            qtype = item["question_type"]
            question = item["question"]
            gold = item["answer"]

            # Retrieve based on baseline type
            t0 = time.monotonic()
            if baseline_name == "random":
                evidence = retrieve_random(qid, len(item.get("haystack_sessions", [])))
            elif baseline_name == "recency":
                evidence = retrieve_recency(qid)
            elif isinstance(config, PipelineConfig):
                evidence = retrieve(question, qid, config)
            else:
                evidence = []
            retrieve_t = time.monotonic() - t0

            answer_sids = set(item.get("answer_session_ids", []))
            retrieved_sids = set(e.get("session_id", "") for e in evidence)
            retrieval_hit = bool(answer_sids & retrieved_sids)

            predicted = ""
            answer_t = 0.0
            if not skip_answer and evidence:
                t0 = time.monotonic()
                predicted = generate_answer(question, evidence)
                answer_t = time.monotonic() - t0

            f1 = compute_f1(predicted, gold)
            em = compute_em(predicted, gold)
            contains = contains_answer(predicted, gold)

            per_question.append({
                "question_id": qid,
                "question_type": qtype,
                "retrieval_hit": retrieval_hit,
                "f1": f1,
                "em": em,
                "contains": contains,
                "retrieve_ms": round(retrieve_t * 1000, 1),
                "predicted_answer": predicted,
                "gold_answer": gold,
            })

            if (i + 1) % 25 == 0:
                running_f1 = sum(r["f1"] for r in per_question) / len(per_question)
                print(f"  [{i+1}/{len(data)}] running F1={running_f1:.3f}", flush=True)

        # Aggregate
        from statistical_utils import aggregate_metrics
        metric_keys = ["f1", "em", "contains", "retrieval_hit"]
        agg = aggregate_metrics(per_question, metric_keys, group_key="question_type")

        overall = agg.get("ALL", {})
        print(f"\n{baseline_name} results:", flush=True)
        for mk in metric_keys:
            m = overall.get(mk)
            if m:
                print(f"  {mk}: {m.mean:.4f} [{m.ci_lower:.4f}, {m.ci_upper:.4f}]", flush=True)

        result = {
            "baseline": baseline_name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "n_questions": len(data),
            "aggregate": {
                group: {mk: metrics[mk].to_dict() for mk in metric_keys}
                for group, metrics in agg.items()
            },
            "per_question": per_question,
        }

        all_results[baseline_name] = result

        # Save individual result
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = RESULTS_DIR / f"baseline_{baseline_name}_{ts}.json"
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)

    # Print comparison matrix
    print(f"\n\n{'='*80}", flush=True)
    print("BASELINE COMPARISON MATRIX", flush=True)
    print(f"{'='*80}", flush=True)

    header = f"{'Baseline':<25} {'F1':>8} {'EM':>8} {'Contains':>10} {'Hit':>8}"
    print(header, flush=True)
    print("-" * 65, flush=True)

    for name in names:
        if name in all_results:
            agg = all_results[name]["aggregate"].get("ALL", {})
            f1 = agg.get("f1", {}).get("mean", 0)
            em = agg.get("em", {}).get("mean", 0)
            cont = agg.get("contains", {}).get("mean", 0)
            hit = agg.get("retrieval_hit", {}).get("mean", 0)
            print(f"{name:<25} {f1:>8.4f} {em:>8.4f} {cont:>10.4f} {hit:>8.4f}", flush=True)

    # Pairwise significance vs full
    if "full" in all_results and len(all_results) > 1:
        print(f"\n\nSignificance tests (vs full):", flush=True)
        full_scores = {r["question_id"]: r for r in all_results["full"]["per_question"]}

        for name in names:
            if name == "full" or name not in all_results:
                continue

            abl_scores = {r["question_id"]: r for r in all_results[name]["per_question"]}
            common = sorted(set(full_scores.keys()) & set(abl_scores.keys()))

            full_f1 = [full_scores[qid]["f1"] for qid in common]
            abl_f1 = [abl_scores[qid]["f1"] for qid in common]

            sig = permutation_test(full_f1, abl_f1)
            effect = cohens_d(full_f1, abl_f1)

            marker = "**" if sig["significant_001"] else ("*" if sig["significant_005"] else "ns")
            print(f"  full vs {name}: diff={sig['observed_diff']:+.4f} p={sig['p_value']:.4f} ({marker}) d={effect:+.3f}", flush=True)

    return all_results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Baseline Comparisons")
    sub = parser.add_subparsers(dest="command")

    run_p = sub.add_parser("run", help="Run baseline comparisons")
    run_p.add_argument("--dataset", default=str(LONGMEMEVAL_DIR / "longmemeval_s.json"))
    run_p.add_argument("--limit", type=int, default=None)
    run_p.add_argument("--skip-answer", action="store_true")
    run_p.add_argument("--baselines", type=str, default=None, help="Comma-separated baseline names")

    args = parser.parse_args()

    if args.command == "run":
        bl = args.baselines.split(",") if args.baselines else None
        run_baselines(args.dataset, limit=args.limit, skip_answer=args.skip_answer, baselines=bl)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
