"""
musique/run.py — MuSiQue benchmark runner.

Tests multi-step compositional reasoning (2-4 hops).
Harder than HotpotQA: measures per-hop retrieval coverage.

Usage:
    python -m musique.run
    python -m musique.run --max-examples 50 --save
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
    ingest_memory_items,
    retrieve,
    save_results,
    token_f1,
    exact_match,
)
from musique.adapter import (
    download_dataset,
    load_dataset,
    paragraphs_to_memory_items,
    get_supporting_indices,
    get_hop_count,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
LOGGER = logging.getLogger("openclaw.benchmarks.musique")

SOURCE_AGENT_PREFIX = "benchmark_musique"


def run_musique(
    max_examples: int | None = 50,
    do_cleanup: bool = True,
    force_download: bool = False,
) -> dict:
    """Run MuSiQue benchmark."""

    if force_download:
        download_dataset(force=True)

    examples = load_dataset(max_examples=max_examples)
    LOGGER.info("Evaluating %d MuSiQue examples", len(examples))

    all_results = []
    latencies: list[float] = []
    f1_scores: list[float] = []
    em_scores: list[float] = []
    support_recall_scores: list[float] = []
    all_found_scores: list[float] = []

    # Per-hop stats
    hop_stats: dict[int, list[dict]] = {}  # hop_count -> list of results

    for i, example in enumerate(examples):
        ex_id = example.get("id", f"ex_{i}")
        question = example.get("question", "")
        gold_answer = example.get("answer", "")
        aliases = example.get("answer_aliases", [])
        supporting_idxs = get_supporting_indices(example)
        num_hops = get_hop_count(example)

        run_id = f"{SOURCE_AGENT_PREFIX}_{ex_id}_{uuid.uuid4().hex[:6]}"

        LOGGER.info(
            "[%d/%d] id=%s | hops=%d | supporting=%d paragraphs",
            i + 1, len(examples), ex_id, num_hops, len(supporting_idxs),
        )

        # Ingest paragraphs
        memory_items = paragraphs_to_memory_items(
            example, source_agent=run_id, source_session=run_id,
        )
        if memory_items:
            ingest_memory_items(memory_items)

        # Retrieve
        retrieved, latency = retrieve(question, limit=10, source_agent_prefix=run_id)
        latencies.append(latency)

        # Check which supporting paragraphs were found
        # Map retrieved items back to paragraph indices via tags
        retrieved_para_idxs = set()
        for r in retrieved:
            tags = r.get("tags", [])
            for tag in (tags if isinstance(tags, list) else []):
                if tag.startswith("para_"):
                    try:
                        retrieved_para_idxs.add(int(tag.split("_")[1]))
                    except ValueError:
                        pass

        support_hits = len(supporting_idxs & retrieved_para_idxs)
        s_recall = support_hits / len(supporting_idxs) if supporting_idxs else 0.0
        all_found = support_hits == len(supporting_idxs)
        support_recall_scores.append(s_recall)
        all_found_scores.append(float(all_found))

        # Answer scoring (check against answer + aliases)
        predicted_text = ""
        for r in retrieved:
            val = r.get("value", "")
            if val and len(val) > 5:
                predicted_text = val
                break
        if not predicted_text and retrieved:
            predicted_text = retrieved[0].get("text", "")[:500]

        best_f1 = token_f1(predicted_text, gold_answer)
        best_em = exact_match(predicted_text, gold_answer)
        for alias in aliases:
            f1_a = token_f1(predicted_text, alias)
            em_a = exact_match(predicted_text, alias)
            best_f1 = max(best_f1, f1_a)
            best_em = max(best_em, em_a)

        f1_scores.append(best_f1)
        em_scores.append(best_em)

        result = {
            "id": ex_id,
            "question": question,
            "gold_answer": gold_answer,
            "predicted": predicted_text[:200],
            "f1": best_f1,
            "exact_match": best_em,
            "num_hops": num_hops,
            "support_recall": s_recall,
            "all_supporting_found": all_found,
            "supporting_idxs": list(supporting_idxs),
            "retrieved_para_idxs": list(retrieved_para_idxs),
            "latency": latency,
        }
        all_results.append(result)

        hop_stats.setdefault(num_hops, []).append(result)

        if do_cleanup:
            deleted = cleanup_benchmark_items(run_id)
            LOGGER.info("Cleaned up %d items", deleted)

    # Aggregate
    if not all_results:
        return {"benchmark": "MuSiQue", "score": 0.0, "error": "no results"}

    n = len(all_results)
    avg_f1 = sum(f1_scores) / n
    avg_em = sum(em_scores) / n
    avg_support_recall = sum(support_recall_scores) / n
    all_found_rate = sum(all_found_scores) / n
    lat = compute_percentiles(latencies)

    # Per-hop breakdown
    per_hop = {}
    for hops, results_list in sorted(hop_stats.items()):
        per_hop[f"{hops}_hop"] = {
            "count": len(results_list),
            "avg_f1": sum(r["f1"] for r in results_list) / len(results_list),
            "avg_support_recall": sum(r["support_recall"] for r in results_list) / len(results_list),
            "all_found_rate": sum(1 for r in results_list if r["all_supporting_found"]) / len(results_list),
        }

    return {
        "benchmark": "MuSiQue",
        "score": avg_f1,
        "avg_f1": avg_f1,
        "avg_exact_match": avg_em,
        "avg_support_recall": avg_support_recall,
        "all_supporting_found_rate": all_found_rate,
        "latency_p50": lat["p50"],
        "latency_p95": lat["p95"],
        "num_examples": n,
        "per_hop": per_hop,
        "per_question": all_results,
    }


def main():
    parser = argparse.ArgumentParser(description="MuSiQue benchmark")
    parser.add_argument("--max-examples", type=int, default=50)
    parser.add_argument("--no-cleanup", action="store_true")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()

    results = run_musique(
        max_examples=args.max_examples,
        do_cleanup=not args.no_cleanup,
        force_download=args.download,
    )

    print("\n--- MuSiQue Results ---")
    print(f"  Examples:              {results.get('num_examples', 0)}")
    print(f"  Avg F1:                {results.get('avg_f1', 0):.3f}")
    print(f"  Avg EM:                {results.get('avg_exact_match', 0):.3f}")
    print(f"  Support Recall:        {results.get('avg_support_recall', 0):.3f}")
    print(f"  All Found Rate:        {results.get('all_supporting_found_rate', 0):.3f}")
    print(f"  Latency p50:           {results.get('latency_p50', 0)*1000:.1f}ms")
    print(f"  Latency p95:           {results.get('latency_p95', 0)*1000:.1f}ms")
    for hop_key, stats in results.get("per_hop", {}).items():
        print(f"  {hop_key} ({stats['count']}): F1={stats['avg_f1']:.3f}  SupportRecall={stats['avg_support_recall']:.3f}  AllFound={stats['all_found_rate']:.3f}")

    if args.save:
        path = save_results("musique", results)
        print(f"\nSaved to: {path}")

    return results


if __name__ == "__main__":
    main()
