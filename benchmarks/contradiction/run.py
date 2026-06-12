"""
contradiction/run.py — Contradiction Detection benchmark runner.

Tests memory system's ability to detect and resolve conflicting information.

Usage:
    python -m contradiction.run
    python -m contradiction.run --save
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
from contradiction.adapter import generate_contradiction_scenarios, generate_contradiction_scenarios_large

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
LOGGER = logging.getLogger("openclaw.benchmarks.contradiction")

SOURCE_AGENT_PREFIX = "benchmark_contradiction"


def run_contradiction(do_cleanup: bool = True, large: int = 0) -> dict:
    """Run Contradiction Detection benchmark."""
    if large > 0:
        scenarios = generate_contradiction_scenarios_large(count=large)
    else:
        scenarios = generate_contradiction_scenarios()
    LOGGER.info("Running %d contradiction scenarios", len(scenarios))

    all_results = []
    latencies: list[float] = []
    f1_scores: list[float] = []

    correct_resolution = 0
    incorrect_resolution = 0
    total_queries = 0

    # Per-category tracking
    category_scores: dict[str, list[float]] = {}

    for scenario in scenarios:
        sc_id = scenario["id"]
        category = scenario.get("category", "unknown")
        run_id = f"{SOURCE_AGENT_PREFIX}_{sc_id}_{uuid.uuid4().hex[:6]}"

        LOGGER.info("Scenario: %s (%s)", scenario["name"], category)

        # Ingest items
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
                tags=["contradiction", sc_id, category],
                item_id=f"{run_id}_item{idx}",
            )
            if "timestamp" in item_def:
                item["first_seen"] = item_def["timestamp"]
                item["last_confirmed"] = item_def["timestamp"]
            memory_items.append(item)

        ingest_memory_items(memory_items)

        # Evaluate queries
        for q in scenario["queries"]:
            query_text = q["query"]
            correct_answer = q["correct_answer"]
            contradicted = q.get("contradicted_answer")
            prefer_latest = q.get("should_prefer_latest", True)

            retrieved, latency = retrieve(query_text, limit=10, source_agent_prefix=run_id)
            latencies.append(latency)
            total_queries += 1

            # Check top result
            top_text = ""
            if retrieved:
                top_text = ((retrieved[0].get("value") or "") + " " + (retrieved[0].get("text") or "")).lower()

            all_text = " ".join(
                (r.get("text") or "") + " " + (r.get("value") or "")
                for r in retrieved
            ).lower()

            # Did we find the correct answer?
            correct_found = correct_answer.lower() in all_text
            correct_in_top = correct_answer.lower() in top_text

            # Did we incorrectly surface the contradicted answer in top position?
            contradicted_in_top = False
            if contradicted:
                contradicted_in_top = contradicted.lower() in top_text and correct_answer.lower() not in top_text

            # Resolution scoring
            if prefer_latest:
                resolved_correctly = correct_in_top and not contradicted_in_top
            else:
                resolved_correctly = correct_found  # Just needs to be present

            if resolved_correctly:
                correct_resolution += 1
            else:
                incorrect_resolution += 1

            # F1
            best_f1 = 0.0
            for r in retrieved:
                for field in ["value", "text"]:
                    val = (r.get(field) or "")
                    if val:
                        best_f1 = max(best_f1, token_f1(val, correct_answer))
            f1_scores.append(best_f1)

            score = float(resolved_correctly)
            category_scores.setdefault(category, []).append(score)

            all_results.append({
                "scenario_id": sc_id,
                "category": category,
                "query": query_text,
                "correct_answer": correct_answer,
                "contradicted_answer": contradicted,
                "correct_found": correct_found,
                "correct_in_top": correct_in_top,
                "contradicted_in_top": contradicted_in_top,
                "resolved_correctly": resolved_correctly,
                "f1": best_f1,
                "latency": latency,
                "top_result_value": (retrieved[0].get("value") or "")[:100] if retrieved else "",
            })

        if do_cleanup:
            deleted = cleanup_benchmark_items(run_id)
            LOGGER.info("Cleaned up %d items", deleted)

    # Aggregate
    resolution_accuracy = correct_resolution / total_queries if total_queries else 0.0
    avg_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
    lat = compute_percentiles(latencies)

    per_category = {}
    for cat, scores in sorted(category_scores.items()):
        per_category[cat] = {
            "count": len(scores),
            "accuracy": sum(scores) / len(scores) if scores else 0.0,
        }

    return {
        "benchmark": "Contradiction",
        "score": resolution_accuracy,
        "resolution_accuracy": resolution_accuracy,
        "avg_f1": avg_f1,
        "correct_resolutions": correct_resolution,
        "incorrect_resolutions": incorrect_resolution,
        "total_queries": total_queries,
        "latency_p50": lat["p50"],
        "latency_p95": lat["p95"],
        "per_category": per_category,
        "per_question": all_results,
    }


def main():
    parser = argparse.ArgumentParser(description="Contradiction Detection benchmark")
    parser.add_argument("--no-cleanup", action="store_true")
    parser.add_argument("--large", type=int, default=0, help="Generate N scenarios (0=hand-crafted only)")
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()

    results = run_contradiction(do_cleanup=not args.no_cleanup, large=args.large)

    print("\n--- Contradiction Detection Results ---")
    print(f"  Queries:               {results.get('total_queries', 0)}")
    print(f"  Resolution Accuracy:   {results.get('resolution_accuracy', 0):.3f}")
    print(f"  Avg F1:                {results.get('avg_f1', 0):.3f}")
    print(f"  Correct Resolutions:   {results.get('correct_resolutions', 0)}")
    print(f"  Incorrect Resolutions: {results.get('incorrect_resolutions', 0)}")
    print(f"  Latency p50:           {results.get('latency_p50', 0)*1000:.1f}ms")
    print(f"  Latency p95:           {results.get('latency_p95', 0)*1000:.1f}ms")
    for cat, stats in results.get("per_category", {}).items():
        print(f"  {cat} ({stats['count']}): accuracy={stats['accuracy']:.3f}")

    if args.save:
        path = save_results("contradiction", results)
        print(f"\nSaved to: {path}")

    return results


if __name__ == "__main__":
    main()
