"""
niah/run.py — Needle-in-a-Haystack benchmark runner.

Usage:
    python run.py                     # Default: sizes [100, 1000], 10 needles
    python run.py --sizes 100 1000    # Custom store sizes
    python run.py --num-needles 20    # More needles
    python run.py --no-cleanup        # Keep items after run (for debugging)
    python run.py --dry-run           # Don't ingest, just generate dataset
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

BENCHMARKS_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BENCHMARKS_DIR))
sys.path.insert(0, str(BENCHMARKS_DIR.parent / "scripts"))

from common import (
    cleanup_benchmark_items,
    compute_percentiles,
    count_tokens_list,
    ingest_memory_items,
    make_memory_item,
    mrr,
    recall_at_k,
    retrieve,
    save_results,
)
from niah.generator import NIAHConfig, build_niah_dataset

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
LOGGER = logging.getLogger("openclaw.benchmarks.niah")

SOURCE_AGENT_PREFIX = "benchmark_niah"


def run_niah(
    store_sizes: list[int],
    num_needles: int = 10,
    do_cleanup: bool = True,
    dry_run: bool = False,
    seed: int = 42,
) -> dict:
    """
    Run NIAH benchmark across multiple store sizes.
    Returns combined results dict.
    """
    all_size_results = []
    all_latencies: list[float] = []
    all_token_counts: list[int] = []

    for store_size in store_sizes:
        LOGGER.info("=" * 50)
        LOGGER.info("NIAH: store_size=%d, num_needles=%d", store_size, num_needles)

        run_id = f"{SOURCE_AGENT_PREFIX}_{store_size}_{int(time.time())}"
        source_agent = run_id

        config = NIAHConfig(
            store_size=store_size,
            num_needles=min(num_needles, 32),  # cap to pool size
            seed=seed,
        )
        dataset = build_niah_dataset(config)
        items_data = dataset["items"]
        needles = dataset["needles"]

        LOGGER.info(
            "Generated %d items (%d needles, %d haystack)",
            len(items_data),
            len(needles),
            len(items_data) - len(needles),
        )

        if dry_run:
            LOGGER.info("Dry run — skipping ingestion")
            continue

        # --- Ingest ---
        memory_items = []
        for item in items_data:
            mi = make_memory_item(
                item["text"],
                source_agent=source_agent,
                source_session=run_id,
                memory_type="fact",
                scope="benchmark",
                tags=["niah", f"size_{store_size}"],
                item_id=item["id"],
            )
            memory_items.append(mi)

        LOGGER.info("Ingesting %d items...", len(memory_items))
        t_ingest_start = time.perf_counter()
        ingest_memory_items(memory_items)
        t_ingest = time.perf_counter() - t_ingest_start
        LOGGER.info("Ingestion done in %.1fs", t_ingest)

        # --- Query needles ---
        needle_results = []
        latencies: list[float] = []
        token_counts: list[int] = []

        for needle in needles:
            results, latency = retrieve(needle.query, limit=10, source_agent_prefix=source_agent)
            latencies.append(latency)

            retrieved_ids = [r.get("id") or r.get("item_id", "") for r in results]
            retrieved_texts = [r.get("text", "") for r in results]

            token_counts.append(count_tokens_list(retrieved_texts))

            # Check if needle is in top-k results
            needle_id = needle.needle_id
            found_at = None
            for rank, row in enumerate(results, 1):
                row_text = row.get("text", "")
                # Check by text match (since ID might not be returned directly)
                if needle.value in row_text or needle_id == row.get("item_id", ""):
                    found_at = rank
                    break

            r1 = 1.0 if found_at is not None and found_at <= 1 else 0.0
            r5 = 1.0 if found_at is not None and found_at <= 5 else 0.0
            r10 = 1.0 if found_at is not None and found_at <= 10 else 0.0
            mrr_val = 1.0 / found_at if found_at is not None else 0.0

            needle_results.append({
                "needle_id": needle_id,
                "entity": needle.entity,
                "query": needle.query,
                "found_at_rank": found_at,
                "recall@1": r1,
                "recall@5": r5,
                "recall@10": r10,
                "mrr": mrr_val,
                "latency": latency,
            })

            LOGGER.debug(
                "Needle '%s': found_at=%s, recall@1=%.0f",
                needle.entity,
                found_at,
                r1,
            )

        # --- Aggregate ---
        if needle_results:
            avg_r1 = sum(r["recall@1"] for r in needle_results) / len(needle_results)
            avg_r5 = sum(r["recall@5"] for r in needle_results) / len(needle_results)
            avg_r10 = sum(r["recall@10"] for r in needle_results) / len(needle_results)
            avg_mrr = sum(r["mrr"] for r in needle_results) / len(needle_results)
        else:
            avg_r1 = avg_r5 = avg_r10 = avg_mrr = 0.0

        lat_stats = compute_percentiles(latencies)
        avg_tokens = sum(token_counts) / max(1, len(token_counts))

        all_latencies.extend(latencies)
        all_token_counts.extend(token_counts)

        size_result = {
            "store_size": store_size,
            "num_needles": len(needles),
            "recall@1": avg_r1,
            "recall@5": avg_r5,
            "recall@10": avg_r10,
            "mrr": avg_mrr,
            "latency_p50": lat_stats["p50"],
            "latency_p95": lat_stats["p95"],
            "latency_mean": lat_stats["mean"],
            "tokens_avg": avg_tokens,
            "ingest_time_s": t_ingest,
            "needle_details": needle_results,
        }
        all_size_results.append(size_result)

        LOGGER.info(
            "Results: recall@1=%.3f, recall@5=%.3f, mrr=%.3f, p50=%.0fms",
            avg_r1, avg_r5, avg_mrr, lat_stats["p50"] * 1000,
        )

        # --- Cleanup ---
        if do_cleanup:
            deleted = cleanup_benchmark_items(source_agent)
            LOGGER.info("Cleaned up %d items", deleted)

    # --- Combined summary ---
    if all_size_results:
        combined_r1 = sum(r["recall@1"] for r in all_size_results) / len(all_size_results)
        combined_mrr = sum(r["mrr"] for r in all_size_results) / len(all_size_results)
        combined_r5 = sum(r["recall@5"] for r in all_size_results) / len(all_size_results)
    else:
        combined_r1 = combined_mrr = combined_r5 = 0.0

    overall_lat = compute_percentiles(all_latencies)
    overall_tokens = sum(all_token_counts) / max(1, len(all_token_counts))

    results = {
        "benchmark": "NIAH",
        "score": combined_r1,           # Primary metric: recall@1
        "recall@1": combined_r1,
        "recall@5": combined_r5,
        "mrr": combined_mrr,
        "tokens_avg": overall_tokens,
        "latency_p50": overall_lat["p50"],
        "latency_p95": overall_lat["p95"],
        "store_sizes": store_sizes,
        "by_size": all_size_results,
        "config": {
            "store_sizes": store_sizes,
            "num_needles": num_needles,
            "seed": seed,
        },
    }

    return results


def main():
    parser = argparse.ArgumentParser(description="NIAH benchmark")
    parser.add_argument("--sizes", nargs="+", type=int, default=[100, 1000],
                        help="Store sizes to test (default: 100 1000)")
    parser.add_argument("--num-needles", type=int, default=10,
                        help="Number of needles per store size (default: 10)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-cleanup", action="store_true",
                        help="Don't delete benchmark items after run")
    parser.add_argument("--dry-run", action="store_true",
                        help="Generate dataset only, no ingestion")
    parser.add_argument("--save", action="store_true",
                        help="Save results to JSON file")
    args = parser.parse_args()

    results = run_niah(
        store_sizes=args.sizes,
        num_needles=args.num_needles,
        do_cleanup=not args.no_cleanup,
        dry_run=args.dry_run,
        seed=args.seed,
    )

    print("\n--- NIAH Results ---")
    for size_r in results.get("by_size", []):
        print(
            f"  store_size={size_r['store_size']:6d} | "
            f"recall@1={size_r['recall@1']:.3f} | "
            f"recall@5={size_r['recall@5']:.3f} | "
            f"mrr={size_r['mrr']:.3f} | "
            f"p50={size_r['latency_p50']*1000:.1f}ms"
        )
    print(f"\nOverall recall@1: {results['recall@1']:.3f}")
    print(f"Overall MRR:      {results['mrr']:.3f}")
    print(f"Latency p50:      {results['latency_p50']*1000:.1f}ms")
    print(f"Latency p95:      {results['latency_p95']*1000:.1f}ms")

    if args.save:
        path = save_results("niah", results)
        print(f"Saved to: {path}")

    return results


if __name__ == "__main__":
    main()
