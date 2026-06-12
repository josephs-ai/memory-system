"""
beam/run.py — BEAM benchmark runner.

Tests 10 memory ability types from the BEAM benchmark (ICLR 2026).
Retrieval is LLM-free (our pipeline). Scoring uses nugget-based matching:
  - For each question, extract rubric nuggets
  - Check if retrieved memories contain the nuggets (token overlap)
  - Score: proportion of nuggets found in top-K retrieval

Usage:
    python -m beam.run --size 1M --save
    python -m beam.run --size 1M --max-conversations 5 --save
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
import uuid
from collections import defaultdict
from pathlib import Path

BENCHMARKS_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BENCHMARKS_DIR))
sys.path.insert(0, str(BENCHMARKS_DIR.parent / "scripts"))

from common import (
    cleanup_benchmark_items,
    ingest_memory_items,
    retrieve,
    save_results,
    token_f1,
)

from beam.adapter import (
    download_dataset,
    parse_chat_to_turns,
    extract_probing_questions,
    extract_rubric_nuggets,
    chat_turns_to_memory_items,
    QUESTION_TYPES,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
LOGGER = logging.getLogger("openclaw.benchmarks.beam")

SOURCE_AGENT_PREFIX = "benchmark_beam"


def _nugget_recall(nuggets: list[str], retrieved_text: str) -> float:
    """Score what fraction of rubric nuggets are found in retrieved text.

    Uses token-level overlap: a nugget is "found" if >50% of its
    non-stopword tokens appear in the retrieved text.
    """
    if not nuggets:
        return 0.0

    STOP_WORDS = frozenset({
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "shall", "can", "to", "of", "in", "for",
        "on", "with", "at", "by", "from", "that", "this", "it", "its",
        "and", "or", "but", "not", "no", "if", "so", "as", "than",
    })

    retrieved_lower = retrieved_text.lower()
    retrieved_tokens = set(re.findall(r'\w+', retrieved_lower))

    found = 0
    for nugget in nuggets:
        nugget_tokens = [
            t for t in re.findall(r'\w+', nugget.lower())
            if t not in STOP_WORDS and len(t) > 2
        ]
        if not nugget_tokens:
            found += 1  # empty nugget counts as found
            continue

        overlap = sum(1 for t in nugget_tokens if t in retrieved_tokens)
        if overlap / len(nugget_tokens) > 0.5:
            found += 1

    return found / len(nuggets)


def run_beam(
    chat_size: str = "1M",
    max_conversations: int | None = None,
    do_cleanup: bool = True,
    top_k: int = 20,
) -> dict:
    """Run BEAM benchmark."""
    conversations = download_dataset(chat_size)
    if max_conversations:
        conversations = conversations[:max_conversations]

    LOGGER.info(
        "Running BEAM %s: %d conversations, %d questions each",
        chat_size, len(conversations), 20,
    )

    all_results = []
    type_scores: dict[str, list[float]] = defaultdict(list)
    latencies: list[float] = []

    for conv_idx, conv in enumerate(conversations):
        conv_id = conv.get("conversation_id", conv_idx)
        run_id = f"{SOURCE_AGENT_PREFIX}_{chat_size}_{conv_id}_{uuid.uuid4().hex[:6]}"

        # Parse and ingest conversation
        turns = parse_chat_to_turns(conv["chat"])
        if not turns:
            LOGGER.warning("[%d] No turns found, skipping", conv_idx)
            continue

        memory_items = chat_turns_to_memory_items(
            turns,
            source_agent=run_id,
            source_session=run_id,
            conversation_id=str(conv_id),
        )

        LOGGER.info(
            "[%d/%d] conv=%s | turns=%d | items=%d",
            conv_idx + 1, len(conversations), conv_id, len(turns), len(memory_items),
        )

        ingest_memory_items(memory_items)

        # Extract and evaluate probing questions
        questions = extract_probing_questions(conv)
        for q in questions:
            q_type = q.get("question_type", "unknown")
            q_text = q.get("question_text", q.get("question", ""))
            if not q_text:
                continue

            nuggets = extract_rubric_nuggets(q)

            # Retrieve using our pipeline (no LLM)
            results, latency = retrieve(q_text, limit=top_k, source_agent_prefix=run_id)
            latencies.append(latency)

            # Build retrieved text blob
            retrieved_text = " ".join(
                (r.get("text", "") + " " + (r.get("value", "") or ""))
                for r in results
            )

            # Score nugget recall
            score = _nugget_recall(nuggets, retrieved_text)
            type_scores[q_type].append(score)

            all_results.append({
                "conversation_id": str(conv_id),
                "question_type": q_type,
                "question": q_text,
                "nuggets": nuggets,
                "nugget_recall": score,
                "num_results": len(results),
                "latency_ms": latency,
            })

        # Cleanup this conversation's items
        if do_cleanup:
            cleanup_benchmark_items(run_id)

    # Compute metrics
    all_scores = [r["nugget_recall"] for r in all_results]
    overall = sum(all_scores) / len(all_scores) if all_scores else 0.0

    per_type = {}
    for q_type in QUESTION_TYPES:
        scores = type_scores.get(q_type, [])
        if scores:
            per_type[q_type] = {
                "accuracy": sum(scores) / len(scores),
                "count": len(scores),
            }

    result = {
        "benchmark": "beam",
        "chat_size": chat_size,
        "total_conversations": len(conversations),
        "total_questions": len(all_results),
        "overall_accuracy": round(overall * 100, 1),  # Match Mem0's percentage format
        "nugget_recall": round(overall, 4),
        "per_type": per_type,
        "latency_p50_ms": sorted(latencies)[len(latencies) // 2] if latencies else 0,
        "latency_p95_ms": sorted(latencies)[int(len(latencies) * 0.95)] if latencies else 0,
        "per_question": all_results,
    }

    return result


def main():
    parser = argparse.ArgumentParser(description="BEAM benchmark")
    parser.add_argument("--size", choices=["100K", "500K", "1M", "10M"], default="1M")
    parser.add_argument("--max-conversations", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--no-cleanup", action="store_true")
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()

    results = run_beam(
        chat_size=args.size,
        max_conversations=args.max_conversations,
        do_cleanup=not args.no_cleanup,
        top_k=args.top_k,
    )

    print(f"\n--- BEAM {args.size} Results ---")
    print(f"  Conversations:     {results['total_conversations']}")
    print(f"  Questions:         {results['total_questions']}")
    print(f"  Overall Score:     {results['overall_accuracy']}")
    print(f"  Nugget Recall:     {results['nugget_recall']:.3f}")
    print(f"  Latency p50:       {results['latency_p50_ms']:.0f}ms")
    print(f"  Latency p95:       {results['latency_p95_ms']:.0f}ms")
    print()
    for q_type, data in sorted(results["per_type"].items()):
        print(f"  {q_type:30s}: {data['accuracy']:.3f} ({data['count']})")

    if args.save:
        path = save_results(results, f"beam_{args.size}")
        print(f"\nSaved to: {path}")


if __name__ == "__main__":
    main()
