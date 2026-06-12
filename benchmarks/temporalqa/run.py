"""
temporalqa/run.py — Temporal QA benchmark runner.

Tests time-aware memory retrieval: recency, temporal ranges, ordering.

Usage:
    python -m temporalqa.run
    python -m temporalqa.run --save
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
    make_memory_item,
    retrieve,
    save_results,
    token_f1,
)
from temporalqa.adapter import generate_temporal_scenarios, generate_temporal_scenarios_large

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
LOGGER = logging.getLogger("openclaw.benchmarks.temporalqa")

SOURCE_AGENT_PREFIX = "benchmark_temporalqa"


def run_temporalqa(do_cleanup: bool = True, large: int = 0) -> dict:
    """Run Temporal QA benchmark."""
    if large > 0:
        scenarios = generate_temporal_scenarios_large(count=large)
    else:
        scenarios = generate_temporal_scenarios()
    LOGGER.info("Running %d Temporal QA scenarios", len(scenarios))

    all_results = []
    latencies: list[float] = []
    correct_count = 0
    total_queries = 0
    f1_scores: list[float] = []

    # Per temporal-hint category
    category_scores: dict[str, list[float]] = {}

    for scenario in scenarios:
        sc_id = scenario["id"]
        run_id = f"{SOURCE_AGENT_PREFIX}_{sc_id}_{uuid.uuid4().hex[:6]}"

        LOGGER.info("Scenario: %s", scenario["name"])

        # Ingest items with timestamps
        memory_items = []
        for idx, item_def in enumerate(scenario["items"]):
            item = make_memory_item(
                item_def["text"],
                source_agent=run_id,
                source_session=run_id,
                entity=item_def.get("entity"),
                property=item_def.get("property"),
                value=item_def.get("value", ""),
                memory_type="fact",
                scope="benchmark",
                tags=["temporalqa", sc_id],
                item_id=f"{run_id}_item{idx}",
            )
            # Override timestamps
            if "timestamp" in item_def:
                item["first_seen"] = item_def["timestamp"]
                item["last_confirmed"] = item_def["timestamp"]
            memory_items.append(item)

        ingest_memory_items(memory_items)

        # Query
        for q in scenario["queries"]:
            query_text = q["query"]
            expected = q["expected"]
            hint = q.get("temporal_hint", "unknown")

            retrieved, latency = retrieve(query_text, limit=10, source_agent_prefix=run_id)
            latencies.append(latency)
            total_queries += 1

            # Check if expected answer is found in top results
            all_text = " ".join(
                (r.get("text") or "") + " " + (r.get("value") or "")
                for r in retrieved
            ).lower()

            found = expected.lower() in all_text

            # Compute F1 against best matching result
            best_f1 = 0.0
            for r in retrieved:
                for field in ["value", "text"]:
                    val = (r.get(field) or "")
                    if val:
                        best_f1 = max(best_f1, token_f1(val, expected))

            f1_scores.append(best_f1)

            # For "latest" queries, check if the top-1 result is the most recent
            temporal_correct = found
            if hint == "latest" and retrieved:
                top_value = ((retrieved[0].get("value") or "") + " " + (retrieved[0].get("text") or "")).lower()
                temporal_correct = expected.lower() in top_value

            if temporal_correct:
                correct_count += 1

            category_scores.setdefault(hint, []).append(float(temporal_correct))

            all_results.append({
                "scenario_id": sc_id,
                "query": query_text,
                "expected": expected,
                "temporal_hint": hint,
                "found_in_results": found,
                "temporal_correct": temporal_correct,
                "f1": best_f1,
                "latency": latency,
                "top_result": (retrieved[0].get("value") or "")[:100] if retrieved else "",
            })

        if do_cleanup:
            deleted = cleanup_benchmark_items(run_id)
            LOGGER.info("Cleaned up %d items", deleted)

    # Aggregate
    accuracy = correct_count / total_queries if total_queries else 0.0
    avg_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
    lat = compute_percentiles(latencies)

    per_category = {}
    for cat, scores in sorted(category_scores.items()):
        per_category[cat] = {
            "count": len(scores),
            "accuracy": sum(scores) / len(scores) if scores else 0.0,
        }

    return {
        "benchmark": "TemporalQA",
        "score": accuracy,
        "accuracy": accuracy,
        "avg_f1": avg_f1,
        "total_queries": total_queries,
        "correct": correct_count,
        "latency_p50": lat["p50"],
        "latency_p95": lat["p95"],
        "per_category": per_category,
        "per_question": all_results,
    }


def main():
    parser = argparse.ArgumentParser(description="TemporalQA benchmark")
    parser.add_argument("--no-cleanup", action="store_true")
    parser.add_argument("--large", type=int, default=0, help="Generate N latest queries (0=use hand-crafted only)")
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()

    results = run_temporalqa(do_cleanup=not args.no_cleanup, large=args.large)

    print("\n--- TemporalQA Results ---")
    print(f"  Queries:           {results.get('total_queries', 0)}")
    print(f"  Accuracy:          {results.get('accuracy', 0):.3f}")
    print(f"  Avg F1:            {results.get('avg_f1', 0):.3f}")
    print(f"  Latency p50:       {results.get('latency_p50', 0)*1000:.1f}ms")
    print(f"  Latency p95:       {results.get('latency_p95', 0)*1000:.1f}ms")
    for cat, stats in results.get("per_category", {}).items():
        print(f"  {cat} ({stats['count']}): accuracy={stats['accuracy']:.3f}")

    if args.save:
        path = save_results("temporalqa", results)
        print(f"\nSaved to: {path}")

    return results


if __name__ == "__main__":
    main()
