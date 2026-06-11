"""
hotpotqa/run.py — HotpotQA benchmark runner.

Tests multi-hop retrieval: can the system retrieve BOTH supporting paragraphs
needed to answer a question that requires combining information from 2+ sources?

Usage:
    python -m hotpotqa.run                     # 50 examples
    python -m hotpotqa.run --max-examples 100
    python -m hotpotqa.run --save
"""

from __future__ import annotations

import argparse
import logging
import sys
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
    recall_at_k,
    precision_at_k,
    mrr,
)
from hotpotqa.adapter import (
    download_dataset,
    load_dataset,
    context_to_memory_items,
    get_supporting_titles,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
LOGGER = logging.getLogger("openclaw.benchmarks.hotpotqa")

SOURCE_AGENT_PREFIX = "benchmark_hotpotqa"


def extract_answer_from_results(results: list[dict]) -> str:
    """Extract answer from retrieved paragraphs (simple concatenation for scoring)."""
    if not results:
        return ""
    # Return the value/text of the top result
    for r in results:
        val = r.get("value", "")
        if val and len(val) > 5:
            return val
    return results[0].get("text", "")[:500]


def run_hotpotqa(
    max_examples: int | None = 50,
    do_cleanup: bool = True,
    force_download: bool = False,
) -> dict:
    """Run HotpotQA benchmark."""

    if force_download:
        download_dataset(force=True)

    examples = load_dataset(max_examples=max_examples)
    LOGGER.info("Evaluating %d HotpotQA examples", len(examples))

    all_results = []
    latencies: list[float] = []
    f1_scores: list[float] = []
    em_scores: list[float] = []
    support_recall_scores: list[float] = []
    support_precision_scores: list[float] = []
    mrr_scores: list[float] = []
    both_found_count = 0
    bridge_results: list[dict] = []
    comparison_results: list[dict] = []

    for i, example in enumerate(examples):
        ex_id = example.get("id", f"ex_{i}")
        question = example.get("question", "")
        gold_answer = example.get("answer", "")
        ex_type = example.get("type", "unknown")
        supporting_titles = get_supporting_titles(example)

        run_id = f"{SOURCE_AGENT_PREFIX}_{ex_id}_{uuid.uuid4().hex[:6]}"

        LOGGER.info(
            "[%d/%d] id=%s | type=%s | support_titles=%s",
            i + 1, len(examples), ex_id, ex_type, supporting_titles,
        )

        # Ingest context paragraphs
        memory_items = context_to_memory_items(
            example, source_agent=run_id, source_session=run_id,
        )
        if memory_items:
            ingest_memory_items(memory_items)

        # Retrieve
        retrieved, latency = retrieve(question, limit=10, source_agent_prefix=run_id)
        latencies.append(latency)

        # Check which supporting titles were found in top-k
        retrieved_titles = []
        for r in retrieved:
            entity = r.get("entity", "")
            if entity:
                retrieved_titles.append(entity)

        # Support recall: fraction of required supporting titles found
        support_hits = sum(1 for t in supporting_titles if t in retrieved_titles)
        s_recall = support_hits / len(supporting_titles) if supporting_titles else 0.0
        support_recall_scores.append(s_recall)

        # Support precision: of top-k entities, how many are supporting?
        s_precision = sum(1 for t in retrieved_titles if t in supporting_titles) / len(retrieved_titles) if retrieved_titles else 0.0
        support_precision_scores.append(s_precision)

        # Both found?
        both_found = support_hits == len(supporting_titles)
        if both_found:
            both_found_count += 1

        # MRR for first supporting title
        mrr_val = mrr(retrieved_titles, supporting_titles)
        mrr_scores.append(mrr_val)

        # Answer scoring
        predicted = extract_answer_from_results(retrieved)
        f1 = token_f1(predicted, gold_answer)
        em = exact_match(predicted, gold_answer)
        f1_scores.append(f1)
        em_scores.append(em)

        result = {
            "id": ex_id,
            "type": ex_type,
            "question": question,
            "gold_answer": gold_answer,
            "predicted": predicted[:200],
            "f1": f1,
            "exact_match": em,
            "support_recall": s_recall,
            "support_precision": s_precision,
            "both_supporting_found": both_found,
            "mrr": mrr_val,
            "latency": latency,
            "supporting_titles": list(supporting_titles),
            "retrieved_titles": retrieved_titles[:10],
        }
        all_results.append(result)

        if ex_type == "bridge":
            bridge_results.append(result)
        elif ex_type == "comparison":
            comparison_results.append(result)

        # Cleanup
        if do_cleanup:
            deleted = cleanup_benchmark_items(run_id)
            LOGGER.info("Cleaned up %d items for %s", deleted, ex_id)

    # Aggregate
    if not all_results:
        return {"benchmark": "HotpotQA", "score": 0.0, "error": "no results"}

    n = len(all_results)
    avg_f1 = sum(f1_scores) / n
    avg_em = sum(em_scores) / n
    avg_support_recall = sum(support_recall_scores) / n
    avg_support_precision = sum(support_precision_scores) / n
    avg_mrr = sum(mrr_scores) / n
    both_found_rate = both_found_count / n
    lat = compute_percentiles(latencies)

    # Per-type breakdowns
    def avg_or_zero(lst, key):
        vals = [r[key] for r in lst]
        return sum(vals) / len(vals) if vals else 0.0

    return {
        "benchmark": "HotpotQA",
        "score": avg_f1,
        "avg_f1": avg_f1,
        "avg_exact_match": avg_em,
        "avg_support_recall": avg_support_recall,
        "avg_support_precision": avg_support_precision,
        "both_supporting_found_rate": both_found_rate,
        "avg_mrr": avg_mrr,
        "latency_p50": lat["p50"],
        "latency_p95": lat["p95"],
        "num_examples": n,
        "bridge": {
            "count": len(bridge_results),
            "avg_f1": avg_or_zero(bridge_results, "f1"),
            "avg_support_recall": avg_or_zero(bridge_results, "support_recall"),
            "both_found_rate": sum(1 for r in bridge_results if r["both_supporting_found"]) / max(1, len(bridge_results)),
        },
        "comparison": {
            "count": len(comparison_results),
            "avg_f1": avg_or_zero(comparison_results, "f1"),
            "avg_support_recall": avg_or_zero(comparison_results, "support_recall"),
            "both_found_rate": sum(1 for r in comparison_results if r["both_supporting_found"]) / max(1, len(comparison_results)),
        },
        "per_question": all_results,
    }


def main():
    parser = argparse.ArgumentParser(description="HotpotQA benchmark")
    parser.add_argument("--max-examples", type=int, default=50)
    parser.add_argument("--no-cleanup", action="store_true")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()

    results = run_hotpotqa(
        max_examples=args.max_examples,
        do_cleanup=not args.no_cleanup,
        force_download=args.download,
    )

    print("\n--- HotpotQA Results ---")
    print(f"  Examples:              {results.get('num_examples', 0)}")
    print(f"  Avg F1:                {results.get('avg_f1', 0):.3f}")
    print(f"  Avg EM:                {results.get('avg_exact_match', 0):.3f}")
    print(f"  Support Recall:        {results.get('avg_support_recall', 0):.3f}  (found both supporting docs)")
    print(f"  Support Precision:     {results.get('avg_support_precision', 0):.3f}")
    print(f"  Both Found Rate:       {results.get('both_supporting_found_rate', 0):.3f}")
    print(f"  MRR:                   {results.get('avg_mrr', 0):.3f}")
    print(f"  Latency p50:           {results.get('latency_p50', 0)*1000:.1f}ms")
    print(f"  Latency p95:           {results.get('latency_p95', 0)*1000:.1f}ms")
    b = results.get("bridge", {})
    c = results.get("comparison", {})
    print(f"  Bridge ({b.get('count',0)}):  F1={b.get('avg_f1',0):.3f}  SupportRecall={b.get('avg_support_recall',0):.3f}  BothFound={b.get('both_found_rate',0):.3f}")
    print(f"  Comparison ({c.get('count',0)}): F1={c.get('avg_f1',0):.3f}  SupportRecall={c.get('avg_support_recall',0):.3f}  BothFound={c.get('both_found_rate',0):.3f}")

    if args.save:
        path = save_results("hotpotqa", results)
        print(f"\nSaved to: {path}")

    return results


if __name__ == "__main__":
    main()
