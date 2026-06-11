"""
msc/run.py — Multi-Session Chat (MSC) benchmark runner.

Tests whether the system can recall persona facts from previous sessions.

Usage:
    python run.py                    # Run on 50 examples
    python run.py --max-examples 100
    python run.py --no-cleanup
    python run.py --download
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
)
from msc.adapter import (
    download_dataset,
    generate_persona_queries,
    load_dataset,
    prev_dialogs_to_memory_items,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
LOGGER = logging.getLogger("openclaw.benchmarks.msc")

SOURCE_AGENT_PREFIX = "benchmark_msc"


def extract_answer_from_results(results: list[dict]) -> str:
    """Extract best answer from retrieved items."""
    if not results:
        return ""

    # Prefer fact/persona memory items
    for r in results:
        if r.get("memory_type") == "fact":
            val = r.get("value", "")
            if val and len(val) > 5:
                return val
            text = r.get("text", "")
            if text:
                return text[:300]

    # Fall back to value or text
    for r in results:
        val = r.get("value", "")
        if val and len(val) > 5:
            return val

    return results[0].get("text", "")[:300]


def run_msc(
    max_examples: int | None = 50,
    do_cleanup: bool = True,
    force_download: bool = False,
    pooled: bool = True,
) -> dict:
    """Run MSC benchmark. Returns results dict.

    When pooled=True (default), all examples are ingested into a single shared
    store before querying.  This makes precision meaningful because each query
    must find the right needle among hundreds of distractors.
    """

    if force_download:
        download_dataset(force=True)

    examples = load_dataset(max_examples=max_examples)
    LOGGER.info("Evaluating %d MSC examples (pooled=%s)", len(examples), pooled)

    all_results = []
    latencies: list[float] = []
    token_counts: list[int] = []
    f1_scores: list[float] = []
    em_scores: list[float] = []
    persona_recall_hits = 0
    total_persona_queries = 0
    precision_scores: list[float] = []
    recall_scores: list[float] = []

    # Shared run_id for pooled mode so all items share a single source_agent prefix
    pooled_run_id = f"{SOURCE_AGENT_PREFIX}_pooled_{uuid.uuid4().hex[:6]}" if pooled else None

    # --- Phase 1: Ingest all items ---
    all_memory_items = []
    per_example_run_ids: list[str] = []

    for i, example in enumerate(examples):
        session_id = example.get("session_id", f"sess_{i}")
        previous_dialogs = example.get("previous_dialogs", [])
        personas = example.get("personas", [])

        if not previous_dialogs and not personas:
            LOGGER.warning("Skipping example %s: no previous dialogs or personas", session_id)
            per_example_run_ids.append("")
            continue

        run_id = pooled_run_id if pooled else f"{SOURCE_AGENT_PREFIX}_{session_id}_{uuid.uuid4().hex[:6]}"
        per_example_run_ids.append(run_id)

        memory_items = prev_dialogs_to_memory_items(
            previous_dialogs,
            source_agent=run_id,
            source_session=f"{run_id}_{session_id}",
        )

        if personas and not previous_dialogs:
            from common import make_memory_item
            for pidx, fact in enumerate(personas):
                if fact.strip():
                    memory_items.append(make_memory_item(
                        fact,
                        source_agent=run_id,
                        source_session=f"{run_id}_{session_id}",
                        entity="speaker",
                        property="persona_fact",
                        value=fact[:200],
                        memory_type="fact",
                        scope="benchmark",
                        tags=["msc", "persona"],
                        item_id=f"{run_id}_{session_id}_persona{pidx}",
                    ))

        if pooled:
            all_memory_items.extend(memory_items)
        else:
            if memory_items:
                ingest_memory_items(memory_items)

    if pooled and all_memory_items:
        LOGGER.info("Pooled ingest: %d total items into shared store", len(all_memory_items))
        # Ingest in batches to avoid overwhelming the API
        BATCH = 50
        for start in range(0, len(all_memory_items), BATCH):
            batch = all_memory_items[start:start + BATCH]
            ingest_memory_items(batch)
        LOGGER.info("Pooled ingest complete")

    # --- Phase 2: Query ---
    for i, example in enumerate(examples):
        session_id = example.get("session_id", f"sess_{i}")
        personas = example.get("personas", [])
        run_id = per_example_run_ids[i]
        if not run_id:
            continue

        # Build the source_session key matching what was used during ingest
        example_source_session = f"{run_id}_{session_id}" if pooled else None

        LOGGER.info(
            "[%d/%d] querying session=%s | personas=%d | session_filter=%s",
            i + 1, len(examples), session_id, len(personas),
            example_source_session[:40] if example_source_session else "none",
        )

        qa_pairs = generate_persona_queries(personas)
        total_persona_queries += len(qa_pairs)

        for qa in qa_pairs:
            question = qa["question"]
            gold_answer = qa["gold_answer"]

            # In pooled mode, scope retrieval to this example's session
            retrieved, latency = retrieve(
                question,
                limit=3,
                source_agent_prefix=run_id,
                session_filter=example_source_session,
            )
            latencies.append(latency)

            retrieved_texts = [r.get("text", "") for r in retrieved]
            token_counts.append(count_tokens_list(retrieved_texts))

            predicted = extract_answer_from_results(retrieved)

            f1 = token_f1(predicted, gold_answer)
            em = exact_match(predicted, gold_answer)
            f1_scores.append(f1)
            em_scores.append(em)

            # Persona recall: did we find any item mentioning the persona fact?
            gold_tokens = set(gold_answer.lower().split())
            recalled = any(
                len(gold_tokens & set(r.get("text", "").lower().split())) > 2
                for r in retrieved
            )
            if recalled:
                persona_recall_hits += 1

            # Precision: of top-k retrieved, how many are relevant to this query?
            relevant_in_topk = sum(
                1 for r in retrieved
                if len(gold_tokens & set(r.get("text", "").lower().split())) > 2
            )
            k = len(retrieved)
            query_precision = relevant_in_topk / k if k > 0 else 0.0
            query_recall = 1.0 if recalled else 0.0
            precision_scores.append(query_precision)
            recall_scores.append(query_recall)

            all_results.append({
                "session_id": session_id,
                "question": question,
                "gold_answer": gold_answer,
                "predicted": predicted,
                "f1": f1,
                "exact_match": em,
                "latency": latency,
                "persona_recalled": recalled,
                "precision": query_precision,
                "recall": query_recall,
                "relevant_in_topk": relevant_in_topk,
                "topk": k,
            })

        # --- Cleanup (per-example mode only) ---
        if do_cleanup and not pooled:
            deleted = cleanup_benchmark_items(run_id)
            LOGGER.info("Cleaned up %d items for session %s", deleted, session_id)

    # --- Pooled cleanup ---
    if do_cleanup and pooled and pooled_run_id:
        deleted = cleanup_benchmark_items(pooled_run_id)
        LOGGER.info("Cleaned up %d pooled items", deleted)

    # --- Aggregate ---
    if not all_results:
        return {"benchmark": "MSC", "score": 0.0, "error": "no results"}

    avg_f1 = sum(f1_scores) / len(f1_scores)
    avg_em = sum(em_scores) / len(em_scores)
    persona_recall = persona_recall_hits / max(1, total_persona_queries)
    lat_stats = compute_percentiles(latencies)
    avg_tokens = sum(token_counts) / max(1, len(token_counts))

    avg_precision = sum(precision_scores) / len(precision_scores) if precision_scores else 0.0
    avg_recall = sum(recall_scores) / len(recall_scores) if recall_scores else 0.0

    return {
        "benchmark": "MSC",
        "score": avg_f1,
        "avg_f1": avg_f1,
        "avg_exact_match": avg_em,
        "avg_precision": avg_precision,         # Avg fraction of top-k that were relevant
        "avg_recall": avg_recall,               # Fraction of needed facts found
        "persona_recall": persona_recall,       # Same as avg_recall (legacy)
        "tokens_avg": avg_tokens,
        "latency_p50": lat_stats["p50"],
        "latency_p95": lat_stats["p95"],
        "num_examples": len(examples),
        "num_qa_pairs": len(all_results),
        "per_question": all_results,
    }


def main():
    parser = argparse.ArgumentParser(description="MSC benchmark")
    parser.add_argument("--max-examples", type=int, default=50,
                        help="Max examples (default: 50)")
    parser.add_argument("--no-cleanup", action="store_true")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()

    results = run_msc(
        max_examples=args.max_examples,
        do_cleanup=not args.no_cleanup,
        force_download=args.download,
    )

    print("\n--- MSC Results ---")
    print(f"  Examples:        {results.get('num_examples', 0)}")
    print(f"  QA Pairs:        {results.get('num_qa_pairs', 0)}")
    print(f"  Avg F1:          {results.get('avg_f1', 0):.3f}")
    print(f"  Avg EM:          {results.get('avg_exact_match', 0):.3f}")
    print(f"  Avg Precision:   {results.get('avg_precision', 0):.3f}  (relevant / top-k)")
    print(f"  Avg Recall:      {results.get('avg_recall', 0):.3f}  (found / needed)")
    print(f"  Persona Recall:  {results.get('persona_recall', 0):.3f}")
    print(f"  Latency p50:     {results.get('latency_p50', 0)*1000:.1f}ms")
    print(f"  Latency p95:     {results.get('latency_p95', 0)*1000:.1f}ms")
    print(f"  Tokens avg:      {results.get('tokens_avg', 0):.0f}")

    if args.save:
        path = save_results("msc", results)
        print(f"\nSaved to: {path}")

    return results


if __name__ == "__main__":
    main()
