#!/usr/bin/env python3
"""
ablation_runner.py — Systematic ablation studies for Phase 1.

Runs the benchmark suite with different feature configurations to measure
the contribution of each component:

Ablation configurations:
1. full          — All features enabled (baseline)
2. vector_only   — Vector search only, no FTS, no rerank
3. fts_only      — FTS/BM25 only, no vector, no rerank
4. no_rerank     — Hybrid search without cross-encoder reranking
5. no_temporal   — Hybrid + rerank but no temporal scoring
6. no_fts        — Vector + rerank, no FTS
7. no_vector     — FTS + rerank, no vector (same as fts_only + rerank)

Usage:
    python ablation_runner.py run                     # All ablations
    python ablation_runner.py run --ablations full,vector_only,fts_only
    python ablation_runner.py run --limit 50          # Quick test
    python ablation_runner.py run --skip-answer       # Retrieval-only (fast)
    python ablation_runner.py compare                 # Compare all saved results
    python ablation_runner.py report                  # Generate ablation report
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

PHASE1_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PHASE1_DIR))

from benchmark_suite import PipelineConfig, run_benchmark, LONGMEMEVAL_DIR, RESULTS_DIR
from statistical_utils import bootstrap_ci, permutation_test, cohens_d


# ---------------------------------------------------------------------------
# Ablation configurations
# ---------------------------------------------------------------------------

ABLATION_CONFIGS: dict[str, PipelineConfig] = {
    "full": PipelineConfig(
        name="full",
        use_vector=True,
        use_fts=True,
        use_rerank=True,
        use_temporal=True,
    ),
    "vector_only": PipelineConfig(
        name="vector_only",
        use_vector=True,
        use_fts=False,
        use_rerank=False,
        use_temporal=False,
    ),
    "fts_only": PipelineConfig(
        name="fts_only",
        use_vector=False,
        use_fts=True,
        use_rerank=False,
        use_temporal=False,
    ),
    "hybrid_no_rerank": PipelineConfig(
        name="hybrid_no_rerank",
        use_vector=True,
        use_fts=True,
        use_rerank=False,
        use_temporal=True,
    ),
    "no_temporal": PipelineConfig(
        name="no_temporal",
        use_vector=True,
        use_fts=True,
        use_rerank=True,
        use_temporal=False,
    ),
    "no_fts": PipelineConfig(
        name="no_fts",
        use_vector=True,
        use_fts=False,
        use_rerank=True,
        use_temporal=True,
    ),
    "fts_plus_rerank": PipelineConfig(
        name="fts_plus_rerank",
        use_vector=False,
        use_fts=True,
        use_rerank=True,
        use_temporal=False,
    ),
}

# Ordered for comparison: full first, then ablations
ABLATION_ORDER = [
    "full",
    "vector_only",
    "fts_only",
    "hybrid_no_rerank",
    "no_temporal",
    "no_fts",
    "fts_plus_rerank",
]


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_ablations(
    dataset_path: str,
    ablation_names: list[str] | None = None,
    limit: int | None = None,
    skip_answer: bool = False,
) -> dict[str, dict]:
    """Run specified ablations and return results keyed by ablation name."""
    names = ablation_names or ABLATION_ORDER

    all_results = {}
    for name in names:
        if name not in ABLATION_CONFIGS:
            print(f"WARNING: Unknown ablation '{name}', skipping", flush=True)
            continue

        config = ABLATION_CONFIGS[name]
        result = run_benchmark(
            dataset_path=dataset_path,
            config=config,
            limit=limit,
            skip_answer=skip_answer,
        )
        all_results[name] = result

    return all_results


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

def compare_ablations(results: dict[str, dict], reference: str = "full") -> dict:
    """
    Compare all ablation results against a reference (typically 'full').
    Returns pairwise significance tests and effect sizes.
    """
    if reference not in results:
        print(f"WARNING: Reference '{reference}' not in results", flush=True)
        return {}

    ref = results[reference]
    ref_per_q = {r["question_id"]: r for r in ref["per_question"]}

    comparisons = {}
    metric_keys = ["f1", "em", "contains", "retrieval_hit"]

    for name, ablation in results.items():
        if name == reference:
            continue

        abl_per_q = {r["question_id"]: r for r in ablation["per_question"]}

        # Align by question_id
        common_qids = sorted(set(ref_per_q.keys()) & set(abl_per_q.keys()))
        if not common_qids:
            comparisons[name] = {"error": "no common questions"}
            continue

        comparison = {"n_common": len(common_qids), "metrics": {}}

        for mk in metric_keys:
            ref_scores = [ref_per_q[qid][mk] for qid in common_qids]
            abl_scores = [abl_per_q[qid][mk] for qid in common_qids]

            ref_summary = bootstrap_ci(ref_scores)
            abl_summary = bootstrap_ci(abl_scores)
            sig = permutation_test(ref_scores, abl_scores)
            effect = cohens_d(ref_scores, abl_scores)

            comparison["metrics"][mk] = {
                "reference_mean": ref_summary.mean,
                "ablation_mean": abl_summary.mean,
                "diff": round(ref_summary.mean - abl_summary.mean, 4),
                "diff_pct": round((ref_summary.mean - abl_summary.mean) / max(ref_summary.mean, 1e-9) * 100, 2),
                "p_value": sig["p_value"],
                "significant_005": sig["significant_005"],
                "significant_001": sig["significant_001"],
                "cohens_d": effect,
            }

        comparisons[name] = comparison

    return comparisons


def print_comparison_table(comparisons: dict, reference: str = "full"):
    """Print a formatted comparison table."""
    print(f"\n{'='*90}", flush=True)
    print(f"ABLATION COMPARISON (reference: {reference})", flush=True)
    print(f"{'='*90}", flush=True)

    header = f"{'Ablation':<25} {'Metric':<15} {'Ref':>8} {'Abl':>8} {'Diff':>8} {'%':>7} {'p':>7} {'Sig':>5} {'d':>7}"
    print(header, flush=True)
    print("-" * 90, flush=True)

    for name, comp in comparisons.items():
        if "error" in comp:
            print(f"{name:<25} ERROR: {comp['error']}", flush=True)
            continue

        for mk, stats in comp["metrics"].items():
            sig_marker = "**" if stats["significant_001"] else ("*" if stats["significant_005"] else "")
            print(
                f"{name:<25} {mk:<15} {stats['reference_mean']:>8.4f} {stats['ablation_mean']:>8.4f} "
                f"{stats['diff']:>+8.4f} {stats['diff_pct']:>+6.1f}% {stats['p_value']:>7.4f} {sig_marker:>5} {stats['cohens_d']:>+7.3f}",
                flush=True,
            )
        print(flush=True)


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def generate_ablation_report(results: dict[str, dict], comparisons: dict) -> str:
    """Generate a markdown ablation report."""
    lines = [
        "# Phase 1 — Ablation Study Report",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Summary",
        "",
        "| Ablation | F1 | EM | Contains | Retrieval Hit |",
        "|----------|-----|-----|----------|---------------|",
    ]

    for name in ABLATION_ORDER:
        if name not in results:
            continue
        agg = results[name].get("aggregate", {}).get("ALL", {})
        f1 = agg.get("f1", {}).get("mean", 0)
        em = agg.get("em", {}).get("mean", 0)
        cont = agg.get("contains", {}).get("mean", 0)
        hit = agg.get("retrieval_hit", {}).get("mean", 0)
        lines.append(f"| {name} | {f1:.4f} | {em:.4f} | {cont:.4f} | {hit:.4f} |")

    lines.extend([
        "",
        "## Ablation Comparisons (vs full)",
        "",
        "| Ablation | Metric | Diff | % Change | p-value | Significant | Cohen's d |",
        "|----------|--------|------|----------|---------|-------------|-----------|",
    ])

    for name, comp in comparisons.items():
        if "error" in comp:
            continue
        for mk, stats in comp["metrics"].items():
            sig = "✓✓" if stats["significant_001"] else ("✓" if stats["significant_005"] else "—")
            lines.append(
                f"| {name} | {mk} | {stats['diff']:+.4f} | {stats['diff_pct']:+.1f}% "
                f"| {stats['p_value']:.4f} | {sig} | {stats['cohens_d']:+.3f} |"
            )

    lines.extend([
        "",
        "## Per-Class Breakdown (full pipeline)",
        "",
    ])

    if "full" in results:
        agg = results["full"].get("aggregate", {})
        for qtype in sorted(agg.keys()):
            if qtype == "ALL":
                continue
            metrics = agg[qtype]
            f1 = metrics.get("f1", {}).get("mean", 0)
            em = metrics.get("em", {}).get("mean", 0)
            cont = metrics.get("contains", {}).get("mean", 0)
            hit = metrics.get("retrieval_hit", {}).get("mean", 0)
            n = metrics.get("f1", {}).get("n", 0)
            lines.append(f"### {qtype} (n={n})")
            lines.append(f"- F1: {f1:.4f}")
            lines.append(f"- EM: {em:.4f}")
            lines.append(f"- Contains: {cont:.4f}")
            lines.append(f"- Retrieval Hit: {hit:.4f}")
            lines.append("")

    lines.extend([
        "## Interpretation Guide",
        "",
        "- **Diff > 0**: full pipeline is better than the ablation (removing the feature hurts)",
        "- **Diff < 0**: removing the feature actually helps (possible over-engineering)",
        "- **p < 0.05**: statistically significant difference (*)",
        "- **p < 0.01**: highly significant difference (**)",
        "- **Cohen's d > 0.5**: medium effect; > 0.8: large effect",
        "",
        "## Key Questions Answered",
        "",
        "1. Does hybrid (vector + FTS) beat single-modality search?",
        "2. Does cross-encoder reranking improve retrieval quality?",
        "3. Does temporal scoring contribute meaningfully?",
        "4. Which component has the largest marginal contribution?",
        "5. Is there a simpler configuration that performs nearly as well?",
    ])

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Phase 1 Ablation Runner")
    sub = parser.add_subparsers(dest="command")

    run_p = sub.add_parser("run", help="Run ablation studies")
    run_p.add_argument("--dataset", default=str(LONGMEMEVAL_DIR / "longmemeval_s.json"))
    run_p.add_argument("--limit", type=int, default=None)
    run_p.add_argument("--skip-answer", action="store_true")
    run_p.add_argument("--ablations", type=str, default=None,
                       help="Comma-separated ablation names (default: all)")

    compare_p = sub.add_parser("compare", help="Compare saved results")
    compare_p.add_argument("--results-dir", default=str(RESULTS_DIR))

    report_p = sub.add_parser("report", help="Generate ablation report")
    report_p.add_argument("--results-dir", default=str(RESULTS_DIR))

    args = parser.parse_args()

    if args.command == "run":
        ablation_names = args.ablations.split(",") if args.ablations else None
        results = run_ablations(
            dataset_path=args.dataset,
            ablation_names=ablation_names,
            limit=args.limit,
            skip_answer=args.skip_answer,
        )

        # Compare and report
        if len(results) > 1 and "full" in results:
            comparisons = compare_ablations(results)
            print_comparison_table(comparisons)

            report = generate_ablation_report(results, comparisons)
            report_path = RESULTS_DIR / f"ablation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
            report_path.write_text(report)
            print(f"\nReport saved to {report_path}", flush=True)

    elif args.command == "compare":
        # Load most recent results for each ablation
        results_dir = Path(args.results_dir)
        results = {}
        for name in ABLATION_ORDER:
            files = sorted(results_dir.glob(f"{name}_run*.json"), reverse=True)
            if files:
                with open(files[0]) as f:
                    results[name] = json.load(f)

        if results:
            comparisons = compare_ablations(results)
            print_comparison_table(comparisons)
        else:
            print("No results found. Run ablations first.", flush=True)

    elif args.command == "report":
        results_dir = Path(args.results_dir)
        results = {}
        for name in ABLATION_ORDER:
            files = sorted(results_dir.glob(f"{name}_run*.json"), reverse=True)
            if files:
                with open(files[0]) as f:
                    results[name] = json.load(f)

        if results and "full" in results:
            comparisons = compare_ablations(results)
            report = generate_ablation_report(results, comparisons)
            report_path = RESULTS_DIR / f"ablation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
            report_path.write_text(report)
            print(f"Report saved to {report_path}", flush=True)
            print(report, flush=True)
        else:
            print("Need at least 'full' results. Run ablations first.", flush=True)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
