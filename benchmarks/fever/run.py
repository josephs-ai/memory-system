"""
fever/run.py — FEVER benchmark runner.

Tests fact verification: can the system retrieve evidence that supports
or refutes a given claim?

Usage:
    python -m fever.run
    python -m fever.run --save
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
)
from fever.adapter import (
    download_dataset,
    load_dataset,
    evidence_to_memory_items,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
LOGGER = logging.getLogger("openclaw.benchmarks.fever")

SOURCE_AGENT_PREFIX = "benchmark_fever"


_nli_model = None


def _get_nli_model():
    """Load the NLI cross-encoder (small model, no LLM needed)."""
    global _nli_model
    if _nli_model is None:
        from sentence_transformers import CrossEncoder
        _nli_model = CrossEncoder("cross-encoder/nli-deberta-v3-base")
        LOGGER.info("Loaded NLI model: cross-encoder/nli-deberta-v3-base")
    return _nli_model


def classify_claim(claim: str, retrieved: list[dict]) -> str:
    """
    Classify claim using NLI cross-encoder (no LLM — same tier as reranker).

    NLI model outputs [contradiction, entailment, neutral] scores.
    """
    if not retrieved:
        return "NOT ENOUGH INFO"

    evidence_pieces = []
    for r in retrieved[:5]:
        text = ((r.get("text") or "") + " " + (r.get("value") or "")).strip()
        if text:
            evidence_pieces.append(text)

    if not evidence_pieces:
        return "NOT ENOUGH INFO"

    # Relevance check
    claim_words = set(claim.lower().split())
    all_evidence = " ".join(evidence_pieces).lower()
    evidence_words = set(all_evidence.split())
    overlap = len(claim_words & evidence_words) / max(1, len(claim_words))

    if overlap < 0.15:
        return "NOT ENOUGH INFO"

    # NLI classification — dual-level scoring.
    # Score at both paragraph level AND sentence level, then combine.
    # Sentence-level catches specific contradictions; paragraph-level
    # prevents false positives from paraphrase fragments.
    try:
        nli = _get_nli_model()

        # --- Paragraph-level scores ---
        para_pairs = [[claim, ev] for ev in evidence_pieces]
        para_scores = nli.predict(para_pairs)
        para_max_c = max(float(s[0]) for s in para_scores)
        para_max_e = max(float(s[1]) for s in para_scores)

        # --- Sentence-level scores ---
        all_sentences = []
        for ev in evidence_pieces:
            clean = ev
            if ":" in clean[:40]:
                clean = clean.split(":", 1)[1].strip()
            for sent in clean.replace(". ", ".\n").split("\n"):
                sent = sent.strip()
                if len(sent) > 10:
                    all_sentences.append(sent)

        if all_sentences:
            sent_pairs = [[claim, sent] for sent in all_sentences]
            sent_scores = nli.predict(sent_pairs)
            sent_max_c = max(float(s[0]) for s in sent_scores)
            sent_max_e = max(float(s[1]) for s in sent_scores)
        else:
            sent_max_c = para_max_c
            sent_max_e = para_max_e

        # --- Decision logic ---
        # REFUTES: both levels must agree on strong contradiction
        # (prevents false positives from paraphrase fragments)
        if sent_max_c > 3.0 and para_max_c > 1.0:
            return "REFUTES"

        # SUPPORTS: entailment at either level, OR high overlap + no contradiction
        if para_max_e > 0.5 or sent_max_e > 1.0:
            return "SUPPORTS"

        # Moderate contradiction — sentence-level strong + paragraph weak → still REFUTES
        if sent_max_c > 4.0:
            return "REFUTES"

        # High relevance + no contradiction signal → SUPPORTS
        if overlap > 0.25 and para_max_c < 1.0 and sent_max_c < 2.0:
            return "SUPPORTS"

        return "NOT ENOUGH INFO"

    except Exception as e:
        LOGGER.warning("NLI failed: %s, using heuristic", e)
        refutation_signals = ["myth", "not true", "false", "incorrect", "contrary to",
                              "is not", "was not", "debunked", "misconception"]
        if any(sig in all_evidence for sig in refutation_signals):
            return "REFUTES"
        return "SUPPORTS" if overlap > 0.3 else "NOT ENOUGH INFO"


def run_fever(
    max_examples: int | None = None,
    do_cleanup: bool = True,
    force_download: bool = False,
) -> dict:
    """Run FEVER benchmark."""

    if force_download:
        download_dataset(force=True)

    examples = load_dataset(max_examples=max_examples)
    LOGGER.info("Evaluating %d FEVER examples", len(examples))

    all_results = []
    latencies: list[float] = []
    correct_label = 0
    total = 0

    # Evidence retrieval metrics
    evidence_found_count = 0

    # Per-label tracking
    label_stats: dict[str, dict] = {}

    for i, example in enumerate(examples):
        ex_id = example.get("id", i)
        claim = example.get("claim", "")
        gold_label = example.get("label", "")

        run_id = f"{SOURCE_AGENT_PREFIX}_{ex_id}_{uuid.uuid4().hex[:6]}"

        LOGGER.info("[%d/%d] claim='%s' | label=%s", i + 1, len(examples), claim[:60], gold_label)

        # Ingest evidence + distractors
        memory_items = evidence_to_memory_items(
            example, source_agent=run_id, source_session=run_id,
        )
        if memory_items:
            ingest_memory_items(memory_items)

        # Retrieve evidence for the claim
        retrieved, latency = retrieve(claim, limit=5, source_agent_prefix=run_id)
        latencies.append(latency)
        total += 1

        # Check if actual evidence (not distractor) was retrieved
        evidence_retrieved = any(
            "evidence" in (r.get("tags", []) if isinstance(r.get("tags"), list) else [])
            for r in retrieved
        )
        if evidence_retrieved:
            evidence_found_count += 1

        # Classify the claim
        predicted_label = classify_claim(claim, retrieved)
        label_correct = predicted_label == gold_label
        if label_correct:
            correct_label += 1

        # Track per gold label
        if gold_label not in label_stats:
            label_stats[gold_label] = {"correct": 0, "total": 0}
        label_stats[gold_label]["total"] += 1
        if label_correct:
            label_stats[gold_label]["correct"] += 1

        all_results.append({
            "id": ex_id,
            "claim": claim,
            "gold_label": gold_label,
            "predicted_label": predicted_label,
            "label_correct": label_correct,
            "evidence_retrieved": evidence_retrieved,
            "latency": latency,
            "num_retrieved": len(retrieved),
        })

        if do_cleanup:
            deleted = cleanup_benchmark_items(run_id)
            LOGGER.info("Cleaned up %d items", deleted)

    # Aggregate
    accuracy = correct_label / total if total else 0.0
    evidence_recall = evidence_found_count / total if total else 0.0
    lat = compute_percentiles(latencies)

    per_label = {}
    for label, stats in sorted(label_stats.items()):
        per_label[label] = {
            "count": stats["total"],
            "accuracy": stats["correct"] / stats["total"] if stats["total"] else 0.0,
        }

    return {
        "benchmark": "FEVER",
        "score": accuracy,
        "label_accuracy": accuracy,
        "evidence_recall": evidence_recall,
        "total_claims": total,
        "correct_labels": correct_label,
        "latency_p50": lat["p50"],
        "latency_p95": lat["p95"],
        "per_label": per_label,
        "per_question": all_results,
    }


def main():
    parser = argparse.ArgumentParser(description="FEVER benchmark")
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--no-cleanup", action="store_true")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()

    results = run_fever(
        max_examples=args.max_examples,
        do_cleanup=not args.no_cleanup,
        force_download=args.download,
    )

    print("\n--- FEVER Results ---")
    print(f"  Claims:            {results.get('total_claims', 0)}")
    print(f"  Label Accuracy:    {results.get('label_accuracy', 0):.3f}")
    print(f"  Evidence Recall:   {results.get('evidence_recall', 0):.3f}")
    print(f"  Latency p50:       {results.get('latency_p50', 0)*1000:.1f}ms")
    print(f"  Latency p95:       {results.get('latency_p95', 0)*1000:.1f}ms")
    for label, stats in results.get("per_label", {}).items():
        print(f"  {label} ({stats['count']}): accuracy={stats['accuracy']:.3f}")

    if args.save:
        path = save_results("fever", results)
        print(f"\nSaved to: {path}")

    return results


if __name__ == "__main__":
    main()
