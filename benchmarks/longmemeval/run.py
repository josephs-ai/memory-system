"""
longmemeval/run.py — LongMemEval benchmark runner.

Usage:
    python run.py                         # Run on 's' split, 50 samples
    python run.py --split m               # Medium context split
    python run.py --max-entries 100       # Limit number of QA pairs
    python run.py --no-cleanup            # Keep items after run
    python run.py --download              # Force re-download dataset
    python run.py --no-llm-judge         # Skip LLM judge, use F1 only
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
import uuid
from pathlib import Path

BENCHMARKS_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BENCHMARKS_DIR))
sys.path.insert(0, str(BENCHMARKS_DIR.parent / "scripts"))

from common import (
    cleanup_benchmark_items,
    compute_percentiles,
    count_tokens_list,
    exact_match,
    ingest_memory_items,
    retrieve,
    save_results,
    token_f1,
)
from longmemeval.adapter import (
    download_dataset,
    extract_answer_from_context,
    load_dataset,
    sessions_to_memory_items,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
LOGGER = logging.getLogger("openclaw.benchmarks.longmemeval")

SOURCE_AGENT_PREFIX = "benchmark_longmemeval"


def run_longmemeval(
    split: str = "s",
    max_entries: int | None = 50,
    do_cleanup: bool = True,
    use_llm_judge: bool = False,
    force_download: bool = False,
) -> dict:
    """Run LongMemEval benchmark. Returns results dict."""

    # Download dataset if needed
    if force_download:
        download_dataset(split, force=True)

    entries = load_dataset(split, max_entries=max_entries)
    LOGGER.info("Running LongMemEval on %d entries (split=%s)", len(entries), split)

    all_results = []
    latencies: list[float] = []
    token_counts: list[int] = []
    f1_scores: list[float] = []
    em_scores: list[float] = []

    for i, entry in enumerate(entries):
        question_id = entry.get("question_id", f"q{i}")
        question_type = entry.get("question_type", "unknown")
        question = entry.get("question", "")
        gold_answer = entry.get("answer", "")
        sessions = entry.get("sessions", entry.get("history", []))

        if not question or not sessions:
            LOGGER.warning("Skipping entry %s: missing question or sessions", question_id)
            continue

        run_id = f"{SOURCE_AGENT_PREFIX}_{question_id}_{uuid.uuid4().hex[:6]}"

        # --- Ingest sessions ---
        memory_items = sessions_to_memory_items(
            sessions,
            source_agent=run_id,
            source_session=run_id,
        )

        if not memory_items:
            LOGGER.warning("No memory items from sessions for %s", question_id)
            continue

        LOGGER.info(
            "[%d/%d] %s | type=%s | items=%d | query='%s...'",
            i + 1, len(entries), question_id, question_type,
            len(memory_items), question[:60],
        )

        ingest_memory_items(memory_items)

        # --- Retrieve ---
        results, latency = retrieve(question, limit=10)
        latencies.append(latency)

        retrieved_texts = [r.get("text", "") for r in results]
        token_counts.append(count_tokens_list(retrieved_texts))

        # --- Score ---
        predicted = extract_answer_from_context(question, results, gold_answer)

        f1 = token_f1(predicted, gold_answer)
        em = exact_match(predicted, gold_answer)

        f1_scores.append(f1)
        em_scores.append(em)

        entry_result = {
            "question_id": question_id,
            "question_type": question_type,
            "question": question,
            "gold_answer": gold_answer,
            "predicted": predicted,
            "f1": f1,
            "exact_match": em,
            "latency": latency,
            "num_retrieved": len(results),
        }
        all_results.append(entry_result)

        # --- Cleanup ---
        if do_cleanup:
            cleanup_benchmark_items(run_id)

    # --- Aggregate ---
    if not all_results:
        return {"benchmark": "LongMemEval", "score": 0.0, "error": "no results"}

    avg_f1 = sum(f1_scores) / len(f1_scores)
    avg_em = sum(em_scores) / len(em_scores)
    lat_stats = compute_percentiles(latencies)
    avg_tokens = sum(token_counts) / max(1, len(token_counts))

    # Per question type breakdown
    type_breakdown: dict[str, dict] = {}
    for r in all_results:
        qt = r["question_type"]
        if qt not in type_breakdown:
            type_breakdown[qt] = {"f1_sum": 0.0, "em_sum": 0.0, "count": 0}
        type_breakdown[qt]["f1_sum"] += r["f1"]
        type_breakdown[qt]["em_sum"] += r["exact_match"]
        type_breakdown[qt]["count"] += 1

    type_summary = {
        qt: {
            "avg_f1": v["f1_sum"] / v["count"],
            "avg_em": v["em_sum"] / v["count"],
            "count": v["count"],
        }
        for qt, v in type_breakdown.items()
    }

    results_out = {
        "benchmark": "LongMemEval",
        "split": split,
        "score": avg_f1,        # Primary metric: token F1
        "avg_f1": avg_f1,
        "avg_exact_match": avg_em,
        "tokens_avg": avg_tokens,
        "latency_p50": lat_stats["p50"],
        "latency_p95": lat_stats["p95"],
        "num_entries": len(all_results),
        "by_question_type": type_summary,
        "per_question": all_results,
    }

    return results_out


def main():
    parser = argparse.ArgumentParser(description="LongMemEval benchmark")
    parser.add_argument("--split", choices=["s", "m", "oracle"], default="s",
                        help="Dataset split (default: s)")
    parser.add_argument("--max-entries", type=int, default=50,
                        help="Max QA entries to evaluate (default: 50)")
    parser.add_argument("--no-cleanup", action="store_true")
    parser.add_argument("--no-llm-judge", action="store_true")
    parser.add_argument("--download", action="store_true", help="Force re-download")
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()

    results = run_longmemeval(
        split=args.split,
        max_entries=args.max_entries,
        do_cleanup=not args.no_cleanup,
        use_llm_judge=not args.no_llm_judge,
        force_download=args.download,
    )

    print("\n--- LongMemEval Results ---")
    print(f"  Split:         {results.get('split', 'N/A')}")
    print(f"  Entries:       {results.get('num_entries', 0)}")
    print(f"  Avg F1:        {results.get('avg_f1', 0):.3f}")
    print(f"  Avg EM:        {results.get('avg_exact_match', 0):.3f}")
    print(f"  Latency p50:   {results.get('latency_p50', 0)*1000:.1f}ms")
    print(f"  Latency p95:   {results.get('latency_p95', 0)*1000:.1f}ms")
    print(f"  Tokens avg:    {results.get('tokens_avg', 0):.0f}")
    print("\n  By question type:")
    for qt, v in results.get("by_question_type", {}).items():
        print(f"    {qt:<30} f1={v['avg_f1']:.3f}  em={v['avg_em']:.3f}  n={v['count']}")

    if args.save:
        path = save_results("longmemeval", results)
        print(f"\nSaved to: {path}")

    return results


if __name__ == "__main__":
    main()
