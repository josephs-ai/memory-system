"""
crud_rag/run.py — CRUD-RAG benchmark runner.

Tests Create, Read, Update, Delete operations on the memory store.

Usage:
    python -m crud_rag.run
    python -m crud_rag.run --save
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
import uuid
from datetime import datetime, timezone
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
from crud_rag.adapter import generate_crud_scenarios, generate_crud_scenarios_large

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
LOGGER = logging.getLogger("openclaw.benchmarks.crud_rag")

SOURCE_AGENT_PREFIX = "benchmark_crud"


def execute_op(op: dict, *, run_id: str, item_registry: dict) -> None:
    """Execute a single CRUD operation."""
    op_type = op["op"]
    entity = op.get("entity", "unknown")
    prop = op.get("property", "unknown")
    key = f"{entity}::{prop}"

    if op_type == "create":
        item = make_memory_item(
            op["text"],
            source_agent=run_id,
            source_session=run_id,
            entity=entity,
            property=prop,
            value=op.get("value", ""),
            memory_type="fact",
            scope="benchmark",
            tags=["crud_rag", op_type],
            item_id=f"{run_id}_{key}_{uuid.uuid4().hex[:6]}",
        )
        ingest_memory_items([item])
        item_registry[key] = item["id"]
        LOGGER.debug("CREATE %s -> %s", key, item["id"])

    elif op_type == "update":
        # Ingest new item first, then mark old as superseded
        # (order matters — upsert would overwrite status if done before ingest)
        old_id = item_registry.get(key)

        new_item = make_memory_item(
            op["text"],
            source_agent=run_id,
            source_session=run_id,
            entity=entity,
            property=prop,
            value=op.get("value", ""),
            memory_type="fact",
            scope="benchmark",
            tags=["crud_rag", op_type],
            item_id=f"{run_id}_{key}_{uuid.uuid4().hex[:6]}",
        )
        if old_id:
            new_item["supersedes"] = old_id
        ingest_memory_items([new_item])
        item_registry[key] = new_item["id"]

        # Mark old item superseded AFTER new item is safely ingested
        if old_id:
            from memory_db import POOL
            with POOL.connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "UPDATE memory_items SET status = 'superseded' WHERE id = %s",
                        (old_id,),
                    )
                conn.commit()
        LOGGER.debug("UPDATE %s: %s -> %s", key, old_id, new_item["id"])

    elif op_type == "delete":
        old_id = item_registry.get(key)
        if old_id:
            from memory_db import POOL
            with POOL.connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "UPDATE memory_items SET status = 'deleted' WHERE id = %s",
                        (old_id,),
                    )
                    conn.commit()
            del item_registry[key]
            LOGGER.debug("DELETE %s: %s", key, old_id)


def evaluate_query(query: dict, *, run_id: str) -> dict:
    """Evaluate a single query against the current memory state."""
    q_text = query["query"]
    expected = query["expected_answer"]
    should_find = query["should_find"]
    should_not_find = query.get("should_not_find", None)

    retrieved, latency = retrieve(q_text, limit=10, source_agent_prefix=run_id)

    # Only evaluate results that belong to this benchmark run (same source_agent prefix).
    # Cross-benchmark contamination (e.g. LoCoMo items about "blue" paint) must not
    # trigger false negative_violations.
    owned = [r for r in retrieved if (r.get("source_agent") or "").startswith(run_id)]
    # Fall back to all results if source_agent filter yields nothing (Qdrant payload gap)
    eval_set = owned if owned else retrieved

    retrieved_texts = " ".join((r.get("text") or "") + " " + (r.get("value") or "") for r in eval_set).lower()
    all_retrieved_texts = " ".join((r.get("text") or "") + " " + (r.get("value") or "") for r in retrieved).lower()

    found = expected.lower() in all_retrieved_texts if expected else False

    # Score correctness
    if should_find:
        correct = found
    else:
        # Should NOT find the answer (deleted/nonexistent)
        correct = not found

    # Check negative constraint — only against owned results to avoid cross-contamination
    negative_violation = False
    if should_not_find and should_not_find.lower() in retrieved_texts:
        negative_violation = True
        correct = False

    # F1 against expected
    best_f1 = 0.0
    if expected and retrieved:
        for r in retrieved:
            for field in ["value", "text"]:
                val = r.get(field) or ""
                if val:
                    f1 = token_f1(val, expected)
                    best_f1 = max(best_f1, f1)

    return {
        "query": q_text,
        "expected_answer": expected,
        "should_find": should_find,
        "found": found,
        "correct": correct,
        "negative_violation": negative_violation,
        "f1": best_f1 if should_find else (1.0 if correct else 0.0),
        "latency": latency,
        "num_retrieved": len(retrieved),
    }


def run_crud_rag(do_cleanup: bool = True, large: int = 0) -> dict:
    """Run CRUD-RAG benchmark."""
    if large > 0:
        scenarios = generate_crud_scenarios_large(count=large)
    else:
        scenarios = generate_crud_scenarios()
    LOGGER.info("Running %d CRUD-RAG scenarios", len(scenarios))

    all_query_results = []
    scenario_results = []
    latencies: list[float] = []

    create_scores: list[float] = []
    update_scores: list[float] = []
    delete_scores: list[float] = []
    read_scores: list[float] = []
    conflict_scores: list[float] = []

    for scenario in scenarios:
        sc_id = scenario["id"]
        sc_name = scenario["name"]
        run_id = f"{SOURCE_AGENT_PREFIX}_{sc_id}_{uuid.uuid4().hex[:6]}"

        LOGGER.info("Scenario: %s", sc_name)

        # Execute operations
        item_registry: dict[str, str] = {}
        for op in scenario["ops"]:
            execute_op(op, run_id=run_id, item_registry=item_registry)

        # Small delay to let indexes settle
        time.sleep(0.1)

        # Evaluate queries
        query_results = []
        for q in scenario["queries"]:
            result = evaluate_query(q, run_id=run_id)
            result["scenario_id"] = sc_id
            query_results.append(result)
            all_query_results.append(result)
            latencies.append(result["latency"])

        sc_correct = sum(1 for r in query_results if r["correct"])
        sc_total = len(query_results)
        sc_score = sc_correct / sc_total if sc_total else 0.0

        # Categorize by operation type
        category = sc_id.split("_")[0]
        score_list = {
            "create": create_scores,
            "update": update_scores,
            "delete": delete_scores,
            "read": read_scores,
            "conflict": conflict_scores,
        }.get(category, read_scores)
        score_list.append(sc_score)

        scenario_results.append({
            "id": sc_id,
            "name": sc_name,
            "score": sc_score,
            "correct": sc_correct,
            "total": sc_total,
            "queries": query_results,
        })

        # Cleanup
        if do_cleanup:
            deleted = cleanup_benchmark_items(run_id)
            LOGGER.info("Cleaned up %d items for %s", deleted, sc_id)

    # Aggregate
    total_correct = sum(1 for r in all_query_results if r["correct"])
    total_queries = len(all_query_results)
    overall_accuracy = total_correct / total_queries if total_queries else 0.0
    avg_f1 = sum(r["f1"] for r in all_query_results) / total_queries if total_queries else 0.0
    lat = compute_percentiles(latencies)

    def safe_avg(lst):
        return sum(lst) / len(lst) if lst else 0.0

    return {
        "benchmark": "CRUD-RAG",
        "score": overall_accuracy,
        "overall_accuracy": overall_accuracy,
        "avg_f1": avg_f1,
        "create_accuracy": safe_avg(create_scores),
        "read_accuracy": safe_avg(read_scores),
        "update_accuracy": safe_avg(update_scores),
        "delete_accuracy": safe_avg(delete_scores),
        "conflict_accuracy": safe_avg(conflict_scores),
        "total_scenarios": len(scenarios),
        "total_queries": total_queries,
        "total_correct": total_correct,
        "latency_p50": lat["p50"],
        "latency_p95": lat["p95"],
        "scenarios": scenario_results,
        "per_question": all_query_results,
    }


def main():
    parser = argparse.ArgumentParser(description="CRUD-RAG benchmark")
    parser.add_argument("--no-cleanup", action="store_true")
    parser.add_argument("--large", type=int, default=0, help="Generate N scenarios (0=hand-crafted only)")
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()

    results = run_crud_rag(do_cleanup=not args.no_cleanup, large=args.large)

    print("\n--- CRUD-RAG Results ---")
    print(f"  Scenarios:         {results.get('total_scenarios', 0)}")
    print(f"  Queries:           {results.get('total_queries', 0)}")
    print(f"  Overall Accuracy:  {results.get('overall_accuracy', 0):.3f}")
    print(f"  Avg F1:            {results.get('avg_f1', 0):.3f}")
    print(f"  Create Accuracy:   {results.get('create_accuracy', 0):.3f}")
    print(f"  Read Accuracy:     {results.get('read_accuracy', 0):.3f}")
    print(f"  Update Accuracy:   {results.get('update_accuracy', 0):.3f}")
    print(f"  Delete Accuracy:   {results.get('delete_accuracy', 0):.3f}")
    print(f"  Conflict Accuracy: {results.get('conflict_accuracy', 0):.3f}")
    print(f"  Latency p50:       {results.get('latency_p50', 0)*1000:.1f}ms")
    print(f"  Latency p95:       {results.get('latency_p95', 0)*1000:.1f}ms")

    if args.save:
        path = save_results("crud_rag", results)
        print(f"\nSaved to: {path}")

    return results


if __name__ == "__main__":
    main()
