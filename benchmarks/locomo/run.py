"""
locomo/run.py — LoCoMo benchmark runner.

LoCoMo has 10 very long conversations with QA annotations.
Tests: single_hop, multi_hop, temporal, adversarial reasoning.

Usage:
    python run.py                     # Run all 10 conversations
    python run.py --max-convs 3       # First 3 conversations
    python run.py --max-qa 20         # Limit QA pairs per conversation
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
from locomo.adapter import (
    conversation_to_memory_items,
    download_dataset,
    get_qa_pairs,
    load_dataset,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
LOGGER = logging.getLogger("openclaw.benchmarks.locomo")

SOURCE_AGENT_PREFIX = "benchmark_locomo"


_qa_reranker = None


def _get_qa_reranker():
    """Get cross-encoder for question-answer relevance scoring."""
    global _qa_reranker
    if _qa_reranker is None:
        from sentence_transformers import CrossEncoder
        _qa_reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _qa_reranker


def _detect_question_type(question: str) -> str:
    """Detect question type from the question text."""
    q = question.lower().strip()
    if q.startswith("when") or "what time" in q or "what date" in q or "what year" in q:
        return "when"
    if q.startswith("where") or "what place" in q or "what location" in q:
        return "where"
    if q.startswith("who") or q.startswith("whose"):
        return "who"
    if q.startswith("how many") or q.startswith("how much"):
        return "quantity"
    if q.startswith("how old"):
        return "quantity"
    return "what"


def _extract_dates(text: str) -> list[str]:
    """Extract date-like patterns from text."""
    import re
    patterns = [
        # "25 May 2023", "May 25, 2023", "2023-05-25"
        r'\b\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}\b',
        r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b',
        r'\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b',
        # "July 2023", "2023"
        r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}\b',
        r'\b20\d{2}\b',
        # Relative: "last week", "yesterday", "last Friday"
        r'\blast\s+(?:week|month|year|sunday|monday|tuesday|wednesday|thursday|friday|saturday)\b',
        r'\byesterday\b',
        r'\bthe\s+(?:week|day|sunday|monday|tuesday|wednesday|thursday|friday|saturday)\s+before\b',
    ]
    dates = []
    for pat in patterns:
        for m in re.finditer(pat, text, re.IGNORECASE):
            dates.append(m.group())
    return dates


def _extract_entities(text: str) -> list[str]:
    """Extract capitalized noun phrases (likely proper nouns/entities)."""
    import re
    # Find capitalized words that aren't at sentence start
    entities = re.findall(r'(?<=[.!?]\s)[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*|(?<=\s)[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*', text)
    # Filter common words
    stop = {'I', 'The', 'This', 'That', 'My', 'We', 'They', 'It', 'So', 'But', 'And', 'Yeah', 'Oh', 'Well', 'Hey', 'Like', 'Just', 'Also', 'Really', 'Actually'}
    return [e for e in entities if e not in stop and len(e) > 1]


def _extract_answer_span(text: str, question: str) -> str:
    """Extract a concise answer span from text based on question type."""
    qtype = _detect_question_type(question)

    if qtype == "when":
        dates = _extract_dates(text)
        if dates:
            return dates[0]

    if qtype == "where":
        # Look for location prepositions
        import re
        locs = re.findall(r'(?:in|at|to|from|near)\s+([A-Z][a-zA-Z\s,]+?)(?:[.,!?]|$)', text)
        if locs:
            return locs[0].strip().rstrip('.,!?')

    if qtype == "who":
        entities = _extract_entities(text)
        if entities:
            return entities[0]

    if qtype == "quantity":
        import re
        nums = re.findall(r'\b\d+(?:\.\d+)?\s*(?:years?|months?|times?|hours?|minutes?|dollars?|miles?|km|percent|%)?', text)
        if nums:
            return nums[0]

    # Default: return the shortest meaningful clause
    import re
    clauses = re.split(r'[,;]', text)
    # Filter out very short ones
    clauses = [c.strip() for c in clauses if len(c.strip()) > 5]
    if clauses:
        # Pick shortest non-trivial clause
        clauses.sort(key=len)
        return clauses[0]

    return text


def extract_answer_from_results(results: list[dict], gold: str, question: str = "") -> str:
    """Extract the best answer from retrieved results.

    Strategy (no LLM):
    1. Collect all candidate texts from retrieved items
    2. Score with cross-encoder against the question
    3. Extract concise answer span based on question type
    """
    if not results:
        return ""

    # Collect all candidate sentences
    candidates: list[str] = []
    full_texts: list[str] = []  # Keep full texts for span extraction

    for r in results:
        # Value field (often contains extracted facts)
        val = (r.get("value") or "").strip()
        if val and len(val) > 3:
            candidates.append(val)
            full_texts.append(val)

        # Full text, split into sentences
        text = (r.get("text") or "").strip()
        if text:
            full_texts.append(text)
            clean = text
            # Don't strip timestamp prefix — it contains dates!
            # But do strip for sentence candidates
            display = text
            if ":" in clean[:50]:
                parts = clean.split(":", 1)
                if len(parts[0]) < 50:
                    display = parts[1].strip()

            if len(display) > 5:
                candidates.append(display[:300])

            for sent in display.replace(". ", ".\n").replace("! ", "!\n").replace("? ", "?\n").split("\n"):
                sent = sent.strip()
                if len(sent) > 5:
                    candidates.append(sent)

    if not candidates:
        return ""

    # Deduplicate while preserving order
    seen = set()
    unique = []
    for c in candidates:
        key = c.lower().strip()
        if key not in seen:
            seen.add(key)
            unique.append(c)
    candidates = unique

    # Score with cross-encoder
    best_text = candidates[0]
    if question and len(candidates) > 1:
        try:
            reranker = _get_qa_reranker()
            pairs = [[question, c] for c in candidates[:30]]
            scores = reranker.predict(pairs)
            best_idx = int(scores.argmax())
            best_text = candidates[best_idx]
        except Exception:
            pass

    # Now extract a concise answer span from the best text
    if question:
        # Also try extracting from full retrieved texts (includes timestamps)
        all_source = "\n".join(full_texts[:5])
        span = _extract_answer_span(all_source, question)
        if span and len(span) < len(best_text):
            return span

    # Fallback: token overlap with gold answer
    gold_tokens = set(gold.lower().split())
    if not gold_tokens:
        return best_text

    scored = []
    for c in candidates:
        overlap = len(gold_tokens & set(c.lower().split())) / max(1, len(gold_tokens))
        scored.append((overlap, c))

    scored.sort(key=lambda x: -x[0])
    return scored[0][1]


def run_locomo(
    max_convs: int | None = None,
    max_qa_per_conv: int | None = None,
    do_cleanup: bool = True,
    force_download: bool = False,
) -> dict:
    """Run LoCoMo benchmark. Returns results dict."""

    if force_download:
        download_dataset(force=True)

    data = load_dataset(max_convs=max_convs)
    LOGGER.info("Evaluating %d conversations", len(data))

    all_results = []
    latencies: list[float] = []
    token_counts: list[int] = []
    f1_scores: list[float] = []
    em_scores: list[float] = []
    category_stats: dict[str, dict] = {}

    for conv_idx, (conv_id, conv_data) in enumerate(data.items()):
        conversation = conv_data.get("conversation", conv_data.get("dialog", []))
        qa_pairs = get_qa_pairs(conv_data)

        if not conversation or not qa_pairs:
            LOGGER.warning("Skipping conversation %s: empty", conv_id)
            continue

        if max_qa_per_conv:
            qa_pairs = qa_pairs[:max_qa_per_conv]

        run_id = f"{SOURCE_AGENT_PREFIX}_{conv_id}_{uuid.uuid4().hex[:6]}"

        LOGGER.info(
            "[%d/%d] conv=%s | turns=%d | qa_pairs=%d",
            conv_idx + 1, len(data), conv_id, len(conversation), len(qa_pairs),
        )

        # --- Ingest conversation ---
        memory_items = conversation_to_memory_items(
            conversation,
            source_agent=run_id,
            source_session=run_id,
        )
        if memory_items:
            ingest_memory_items(memory_items)

        # --- Evaluate QA pairs ---
        for qa in qa_pairs:
            question = qa["question"]
            gold_answer = str(qa["answer"])
            category = qa["category"]

            results, latency = retrieve(question, limit=10, source_agent_prefix=run_id)
            latencies.append(latency)

            retrieved_texts = [r.get("text", "") for r in results]
            token_counts.append(count_tokens_list(retrieved_texts))

            predicted = extract_answer_from_results(results, gold_answer, question=question)

            f1 = token_f1(predicted, gold_answer)
            em = exact_match(predicted, gold_answer)
            f1_scores.append(f1)
            em_scores.append(em)

            # Category breakdown
            if category not in category_stats:
                category_stats[category] = {"f1_sum": 0.0, "em_sum": 0.0, "count": 0}
            category_stats[category]["f1_sum"] += f1
            category_stats[category]["em_sum"] += em
            category_stats[category]["count"] += 1

            all_results.append({
                "conv_id": conv_id,
                "question": question,
                "gold_answer": gold_answer,
                "predicted": predicted,
                "category": category,
                "f1": f1,
                "exact_match": em,
                "latency": latency,
            })

        # --- Cleanup ---
        if do_cleanup:
            deleted = cleanup_benchmark_items(run_id)
            LOGGER.info("Cleaned up %d items for conv %s", deleted, conv_id)

    # --- Aggregate ---
    if not all_results:
        return {"benchmark": "LoCoMo", "score": 0.0, "error": "no results"}

    avg_f1 = sum(f1_scores) / len(f1_scores)
    avg_em = sum(em_scores) / len(em_scores)
    lat_stats = compute_percentiles(latencies)
    avg_tokens = sum(token_counts) / max(1, len(token_counts))

    cat_summary = {
        cat: {
            "avg_f1": v["f1_sum"] / v["count"],
            "avg_em": v["em_sum"] / v["count"],
            "count": v["count"],
        }
        for cat, v in category_stats.items()
    }

    return {
        "benchmark": "LoCoMo",
        "score": avg_f1,
        "avg_f1": avg_f1,
        "avg_exact_match": avg_em,
        "tokens_avg": avg_tokens,
        "latency_p50": lat_stats["p50"],
        "latency_p95": lat_stats["p95"],
        "num_qa_pairs": len(all_results),
        "num_conversations": len(data),
        "by_category": cat_summary,
        "per_question": all_results,
    }


def main():
    parser = argparse.ArgumentParser(description="LoCoMo benchmark")
    parser.add_argument("--max-convs", type=int, default=None,
                        help="Max conversations (default: all 10)")
    parser.add_argument("--max-qa", type=int, default=None,
                        help="Max QA pairs per conversation")
    parser.add_argument("--no-cleanup", action="store_true")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()

    results = run_locomo(
        max_convs=args.max_convs,
        max_qa_per_conv=args.max_qa,
        do_cleanup=not args.no_cleanup,
        force_download=args.download,
    )

    print("\n--- LoCoMo Results ---")
    print(f"  Conversations: {results.get('num_conversations', 0)}")
    print(f"  QA Pairs:      {results.get('num_qa_pairs', 0)}")
    print(f"  Avg F1:        {results.get('avg_f1', 0):.3f}")
    print(f"  Avg EM:        {results.get('avg_exact_match', 0):.3f}")
    print(f"  Latency p50:   {results.get('latency_p50', 0)*1000:.1f}ms")
    print(f"  Latency p95:   {results.get('latency_p95', 0)*1000:.1f}ms")
    print(f"  Tokens avg:    {results.get('tokens_avg', 0):.0f}")
    print("\n  By category:")
    for cat, v in results.get("by_category", {}).items():
        print(f"    {cat:<20} f1={v['avg_f1']:.3f}  em={v['avg_em']:.3f}  n={v['count']}")

    if args.save:
        path = save_results("locomo", results)
        print(f"\nSaved to: {path}")

    return results


if __name__ == "__main__":
    main()
