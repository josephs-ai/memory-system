"""
run_all.py — Run all memory system benchmarks and produce a combined results table.

Usage:
    python run_all.py                     # Run all benchmarks
    python run_all.py --benchmarks niah   # Run only NIAH
    python run_all.py --benchmarks niah longmemeval
    python run_all.py --quick             # Quick mode: smaller sizes/samples
    python run_all.py --save              # Save JSON + markdown results
    python run_all.py --no-cleanup        # Keep benchmark items (debug)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

BENCHMARKS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BENCHMARKS_DIR))
sys.path.insert(0, str(BENCHMARKS_DIR.parent / "scripts"))

from common import format_results_table, print_results_table, save_results

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
LOGGER = logging.getLogger("openclaw.benchmarks.run_all")

RESULTS_DIR = BENCHMARKS_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

ALL_BENCHMARK_NAMES = ["niah", "longmemeval", "locomo", "msc", "hotpotqa", "musique", "crud_rag", "temporalqa", "contradiction", "fever"]


def run_niah_benchmark(quick: bool = False, cleanup: bool = True) -> dict:
    from niah.run import run_niah
    sizes = [100, 500] if quick else [100, 1000]
    num_needles = 5 if quick else 10
    LOGGER.info("Running NIAH (sizes=%s, needles=%d)", sizes, num_needles)
    return run_niah(store_sizes=sizes, num_needles=num_needles, do_cleanup=cleanup)


def run_longmemeval_benchmark(quick: bool = False, cleanup: bool = True) -> dict:
    from longmemeval.run import run_longmemeval
    max_entries = 10 if quick else 50
    LOGGER.info("Running LongMemEval (max_entries=%d)", max_entries)
    return run_longmemeval(
        split="s",
        max_entries=max_entries,
        do_cleanup=cleanup,
        use_llm_judge=False,
    )


def run_locomo_benchmark(quick: bool = False, cleanup: bool = True) -> dict:
    from locomo.run import run_locomo
    max_convs = 2 if quick else None
    max_qa = 10 if quick else 30
    LOGGER.info("Running LoCoMo (max_convs=%s, max_qa=%s)", max_convs, max_qa)
    return run_locomo(
        max_convs=max_convs,
        max_qa_per_conv=max_qa,
        do_cleanup=cleanup,
    )


def run_msc_benchmark(quick: bool = False, cleanup: bool = True) -> dict:
    from msc.run import run_msc
    max_examples = 10 if quick else 50
    LOGGER.info("Running MSC (max_examples=%d)", max_examples)
    return run_msc(
        max_examples=max_examples,
        do_cleanup=cleanup,
    )


def run_hotpotqa_benchmark(quick: bool = False, cleanup: bool = True) -> dict:
    from hotpotqa.run import run_hotpotqa
    max_examples = 5 if quick else 50
    LOGGER.info("Running HotpotQA (max_examples=%d)", max_examples)
    return run_hotpotqa(max_examples=max_examples, do_cleanup=cleanup)


def run_musique_benchmark(quick: bool = False, cleanup: bool = True) -> dict:
    from musique.run import run_musique
    max_examples = 5 if quick else 50
    LOGGER.info("Running MuSiQue (max_examples=%d)", max_examples)
    return run_musique(max_examples=max_examples, do_cleanup=cleanup)


def run_crud_rag_benchmark(quick: bool = False, cleanup: bool = True) -> dict:
    from crud_rag.run import run_crud_rag
    LOGGER.info("Running CRUD-RAG")
    return run_crud_rag(do_cleanup=cleanup)


def run_temporalqa_benchmark(quick: bool = False, cleanup: bool = True) -> dict:
    from temporalqa.run import run_temporalqa
    LOGGER.info("Running TemporalQA")
    return run_temporalqa(do_cleanup=cleanup)


def run_contradiction_benchmark(quick: bool = False, cleanup: bool = True) -> dict:
    from contradiction.run import run_contradiction
    LOGGER.info("Running Contradiction Detection")
    return run_contradiction(do_cleanup=cleanup)


def run_fever_benchmark(quick: bool = False, cleanup: bool = True) -> dict:
    from fever.run import run_fever
    max_examples = 5 if quick else None
    LOGGER.info("Running FEVER (max_examples=%s)", max_examples)
    return run_fever(max_examples=max_examples, do_cleanup=cleanup)


BENCHMARK_RUNNERS = {
    "niah": run_niah_benchmark,
    "longmemeval": run_longmemeval_benchmark,
    "locomo": run_locomo_benchmark,
    "msc": run_msc_benchmark,
    "hotpotqa": run_hotpotqa_benchmark,
    "musique": run_musique_benchmark,
    "crud_rag": run_crud_rag_benchmark,
    "temporalqa": run_temporalqa_benchmark,
    "contradiction": run_contradiction_benchmark,
    "fever": run_fever_benchmark,
}


def run_all(
    benchmark_names: list[str] | None = None,
    quick: bool = False,
    do_cleanup: bool = True,
    save: bool = False,
) -> dict:
    """Run all specified benchmarks and return combined results."""
    if benchmark_names is None:
        benchmark_names = ALL_BENCHMARK_NAMES

    all_results = []
    errors = {}
    start_time = time.perf_counter()

    for name in benchmark_names:
        if name not in BENCHMARK_RUNNERS:
            LOGGER.warning("Unknown benchmark: %s", name)
            continue

        LOGGER.info("=" * 60)
        LOGGER.info("Starting benchmark: %s", name.upper())
        LOGGER.info("=" * 60)

        try:
            t0 = time.perf_counter()
            result = BENCHMARK_RUNNERS[name](quick=quick, cleanup=do_cleanup)
            elapsed = time.perf_counter() - t0
            result["wall_time_s"] = elapsed
            all_results.append(result)
            LOGGER.info(
                "Benchmark %s done in %.1fs: score=%.3f",
                name, elapsed, result.get("score", 0),
            )
        except Exception as e:
            LOGGER.error("Benchmark %s FAILED: %s", name, e, exc_info=True)
            errors[name] = str(e)
            all_results.append({
                "benchmark": name.upper(),
                "score": 0.0,
                "error": str(e),
                "tokens_avg": 0,
                "latency_p50": 0.0,
                "latency_p95": 0.0,
            })

    total_time = time.perf_counter() - start_time

    combined = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_wall_time_s": total_time,
        "benchmarks_run": benchmark_names,
        "quick_mode": quick,
        "results": all_results,
        "errors": errors,
    }

    # Print summary table
    print_results_table(all_results)

    if errors:
        print("Benchmark errors:")
        for name, err in errors.items():
            print(f"  {name}: {err}")

    print(f"\nTotal wall time: {total_time:.1f}s")

    # Save combined results
    if save:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        combined_path = RESULTS_DIR / f"combined_{ts}.json"
        with open(combined_path, "w") as f:
            json.dump(combined, f, indent=2, default=str)
        LOGGER.info("Saved combined results to %s", combined_path)

        # Also save markdown
        md_path = RESULTS_DIR / f"results_{ts}.md"
        md_content = _format_markdown_report(combined)
        md_path.write_text(md_content)
        LOGGER.info("Saved markdown report to %s", md_path)
        print(f"Results saved to:\n  {combined_path}\n  {md_path}")

    return combined


def _format_markdown_report(combined: dict) -> str:
    """Format full markdown report."""
    ts = combined.get("timestamp", "")
    quick = combined.get("quick_mode", False)
    total_time = combined.get("total_wall_time_s", 0)
    results = combined.get("results", [])

    lines = [
        "# Memory System Benchmark Results",
        "",
        f"**Timestamp:** {ts}",
        f"**Mode:** {'Quick' if quick else 'Full'}",
        f"**Total time:** {total_time:.1f}s",
        "",
        "## Summary",
        "",
        format_results_table(results),
        "",
        "## Detailed Results",
        "",
    ]

    for r in results:
        name = r.get("benchmark", "?")
        lines.append(f"### {name}")
        lines.append("")

        if "error" in r and r["error"]:
            lines.append(f"**Error:** {r['error']}")
        else:
            lines.append(f"- **Score (primary):** {r.get('score', 0):.3f}")
            lines.append(f"- **Avg F1:** {r.get('avg_f1', r.get('recall@1', 0)):.3f}")
            lines.append(f"- **Latency p50:** {r.get('latency_p50', 0)*1000:.1f}ms")
            lines.append(f"- **Latency p95:** {r.get('latency_p95', 0)*1000:.1f}ms")
            lines.append(f"- **Avg tokens:** {r.get('tokens_avg', 0):.0f}")

            # Benchmark-specific details
            if name == "NIAH":
                lines.append(f"- **Recall@1:** {r.get('recall@1', 0):.3f}")
                lines.append(f"- **Recall@5:** {r.get('recall@5', 0):.3f}")
                lines.append(f"- **MRR:** {r.get('mrr', 0):.3f}")
                lines.append("")
                lines.append("| Store Size | Recall@1 | Recall@5 | MRR | p50 |")
                lines.append("|------------|----------|----------|-----|-----|")
                for sz in r.get("by_size", []):
                    lines.append(
                        f"| {sz['store_size']} | {sz['recall@1']:.3f} | "
                        f"{sz['recall@5']:.3f} | {sz['mrr']:.3f} | "
                        f"{sz['latency_p50']*1000:.1f}ms |"
                    )

            elif name in ("LongMemEval", "LoCoMo"):
                lines.append(f"- **Exact Match:** {r.get('avg_exact_match', 0):.3f}")
                breakdown_key = "by_question_type" if name == "LongMemEval" else "by_category"
                breakdown = r.get(breakdown_key, {})
                if breakdown:
                    lines.append("")
                    lines.append("| Type | F1 | EM | N |")
                    lines.append("|------|-----|-----|---|")
                    for qt, v in breakdown.items():
                        lines.append(
                            f"| {qt} | {v['avg_f1']:.3f} | {v['avg_em']:.3f} | {v['count']} |"
                        )

            elif name == "MSC":
                lines.append(f"- **Exact Match:** {r.get('avg_exact_match', 0):.3f}")
                lines.append(f"- **Persona Recall:** {r.get('persona_recall', 0):.3f}")

        lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Run all memory system benchmarks")
    parser.add_argument(
        "--benchmarks", nargs="+",
        choices=ALL_BENCHMARK_NAMES + ["all"],
        default=["all"],
        help="Which benchmarks to run (default: all)",
    )
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: smaller dataset sizes for fast testing")
    parser.add_argument("--no-cleanup", action="store_true",
                        help="Don't clean up benchmark items after run")
    parser.add_argument("--save", action="store_true",
                        help="Save results to JSON + markdown")
    args = parser.parse_args()

    benchmark_names = args.benchmarks
    if "all" in benchmark_names:
        benchmark_names = ALL_BENCHMARK_NAMES

    run_all(
        benchmark_names=benchmark_names,
        quick=args.quick,
        do_cleanup=not args.no_cleanup,
        save=args.save,
    )


if __name__ == "__main__":
    main()
