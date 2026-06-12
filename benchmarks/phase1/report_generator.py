#!/usr/bin/env python3
"""
report_generator.py — Consolidated Phase 1 reporting.

Generates a single markdown report combining:
1. Full benchmark results (all query classes)
2. Ablation study results
3. Baseline comparison results
4. Task success eval results (if available)

Usage:
    python report_generator.py              # Auto-find latest results
    python report_generator.py --format md  # Markdown (default)
    python report_generator.py --format json  # JSON output
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PHASE1_DIR = Path(__file__).resolve().parent
RESULTS_DIR = PHASE1_DIR / "results"


def load_latest(pattern: str) -> dict | None:
    """Load the most recent result file matching a glob pattern."""
    files = sorted(RESULTS_DIR.glob(pattern), reverse=True)
    if not files:
        return None
    with open(files[0]) as f:
        return json.load(f)


def load_all_ablations() -> dict[str, dict]:
    """Load the latest result for each ablation."""
    from ablation_runner import ABLATION_ORDER

    results = {}
    for name in ABLATION_ORDER:
        data = load_latest(f"{name}_run*.json")
        if data:
            results[name] = data
    return results


def generate_report() -> str:
    """Generate consolidated Phase 1 report."""
    lines = [
        "# Phase 1 — Performance Report: Make It Undeniable",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
    ]

    # --- Section 1: Full benchmark ---
    full = load_latest("full_run*.json")
    if full:
        lines.extend([
            "## 1. Full Pipeline Results (All Query Classes)",
            "",
            f"Pipeline: {full.get('pipeline', 'full')}",
            f"Questions: {len(full.get('per_question', []))}",
            "",
            "| Query Class | N | F1 | EM | Contains | Retrieval Hit |",
            "|-------------|---|-----|-----|----------|---------------|",
        ])

        agg = full.get("aggregate", {})
        for qtype in sorted(agg.keys()):
            metrics = agg[qtype]
            n = metrics.get("f1", {}).get("n", 0)
            f1 = metrics.get("f1", {}).get("mean", 0)
            em = metrics.get("em", {}).get("mean", 0)
            cont = metrics.get("contains", {}).get("mean", 0)
            hit = metrics.get("retrieval_hit", {}).get("mean", 0)
            lines.append(f"| {qtype} | {n} | {f1:.4f} | {em:.4f} | {cont:.4f} | {hit:.4f} |")

        lines.append("")

        # Highlight weak areas
        lines.append("### Weak areas (F1 < 0.8):")
        for qtype in sorted(agg.keys()):
            if qtype == "ALL":
                continue
            f1 = agg[qtype].get("f1", {}).get("mean", 0)
            if f1 < 0.8:
                lines.append(f"- **{qtype}**: F1={f1:.4f} — needs investigation")
        lines.append("")

    else:
        lines.extend(["## 1. Full Pipeline Results", "", "_No results found. Run: `python benchmark_suite.py run`_", ""])

    # --- Section 2: Ablation study ---
    ablations = load_all_ablations()
    if ablations:
        lines.extend([
            "## 2. Ablation Study",
            "",
            "| Configuration | F1 | EM | Contains | Hit | Δ F1 vs full |",
            "|---------------|-----|-----|----------|-----|--------------|",
        ])

        full_f1 = ablations.get("full", {}).get("aggregate", {}).get("ALL", {}).get("f1", {}).get("mean", 0)

        for name in ["full", "vector_only", "fts_only", "hybrid_no_rerank", "no_temporal", "no_fts", "fts_plus_rerank"]:
            if name not in ablations:
                continue
            agg = ablations[name].get("aggregate", {}).get("ALL", {})
            f1 = agg.get("f1", {}).get("mean", 0)
            em = agg.get("em", {}).get("mean", 0)
            cont = agg.get("contains", {}).get("mean", 0)
            hit = agg.get("retrieval_hit", {}).get("mean", 0)
            delta = f1 - full_f1 if name != "full" else 0
            lines.append(f"| {name} | {f1:.4f} | {em:.4f} | {cont:.4f} | {hit:.4f} | {delta:+.4f} |")

        lines.extend([
            "",
            "### Component Contributions (Δ F1 when removed):",
            "",
        ])

        contributions = []
        for name in ["no_fts", "hybrid_no_rerank", "no_temporal", "vector_only", "fts_only"]:
            if name not in ablations:
                continue
            abl_f1 = ablations[name].get("aggregate", {}).get("ALL", {}).get("f1", {}).get("mean", 0)
            removed = {
                "no_fts": "FTS/BM25",
                "hybrid_no_rerank": "Cross-encoder rerank",
                "no_temporal": "Temporal scoring",
                "vector_only": "FTS + rerank + temporal",
                "fts_only": "Vector + rerank + temporal",
            }
            delta = full_f1 - abl_f1
            contributions.append((removed.get(name, name), delta))

        contributions.sort(key=lambda x: x[1], reverse=True)
        for comp, delta in contributions:
            bar = "█" * int(abs(delta) * 100) if delta > 0 else ""
            direction = "↓" if delta > 0 else "↑" if delta < 0 else "="
            lines.append(f"- **{comp}**: {direction} {delta:+.4f} F1 {bar}")
        lines.append("")

    else:
        lines.extend(["## 2. Ablation Study", "", "_No results found. Run: `python ablation_runner.py run`_", ""])

    # --- Section 3: Baseline comparisons ---
    baselines = {}
    for name in ["random", "recency", "fts_only", "vector_only", "hybrid_no_rerank", "full"]:
        data = load_latest(f"baseline_{name}_*.json")
        if data:
            baselines[name] = data

    if baselines:
        lines.extend([
            "## 3. Baseline Comparisons",
            "",
            "| Baseline | F1 | EM | Contains | Hit |",
            "|----------|-----|-----|----------|-----|",
        ])

        for name in ["random", "recency", "fts_only", "vector_only", "hybrid_no_rerank", "full"]:
            if name not in baselines:
                continue
            agg = baselines[name].get("aggregate", {}).get("ALL", {})
            f1 = agg.get("f1", {}).get("mean", 0)
            em = agg.get("em", {}).get("mean", 0)
            cont = agg.get("contains", {}).get("mean", 0)
            hit = agg.get("retrieval_hit", {}).get("mean", 0)
            lines.append(f"| {name} | {f1:.4f} | {em:.4f} | {cont:.4f} | {hit:.4f} |")
        lines.append("")

    else:
        lines.extend(["## 3. Baseline Comparisons", "", "_No results found. Run: `python baseline_comparisons.py run`_", ""])

    # --- Section 4: Task success ---
    task_data = load_latest("task_success_2*.json")
    if task_data and isinstance(task_data, list):
        helps = sum(1 for r in task_data if r.get("memory_helps"))
        total = len(task_data)
        lines.extend([
            "## 4. Downstream Task Success",
            "",
            f"Memory improves task success in **{helps}/{total}** cases ({helps/max(total,1)*100:.0f}%)",
            "",
            "| Task | Category | With Memory | Without | Delta |",
            "|------|----------|-------------|---------|-------|",
        ])

        for r in task_data:
            with_s = r.get("variants", {}).get("with_memory", {}).get("score", 0)
            without_s = r.get("variants", {}).get("without_memory", {}).get("score", 0)
            delta = r.get("memory_delta", 0)
            lines.append(f"| {r['task_id']} | {r['category']} | {with_s:.2f} | {without_s:.2f} | {delta:+.2f} |")
        lines.append("")

    else:
        lines.extend(["## 4. Downstream Task Success", "", "_No results found. Run: `python task_success_eval.py run`_", ""])

    # --- Conclusions ---
    lines.extend([
        "## 5. Key Findings",
        "",
        "_Fill in after running all benchmarks:_",
        "",
        "1. **Hybrid vs single-modality**: [TBD]",
        "2. **Rerank contribution**: [TBD]",
        "3. **Temporal scoring contribution**: [TBD]",
        "4. **Biggest component contribution**: [TBD]",
        "5. **Simplification opportunities**: [TBD]",
        "6. **Weakest query classes**: [TBD]",
        "",
        "## 6. Recommendations",
        "",
        "1. [TBD after analysis]",
    ])

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Phase 1 Report Generator")
    parser.add_argument("--format", choices=["md", "json"], default="md")
    args = parser.parse_args()

    report = generate_report()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.format == "md":
        path = RESULTS_DIR / f"phase1_report_{ts}.md"
        path.write_text(report)
        print(report, flush=True)
        print(f"\nReport saved to {path}", flush=True)
    else:
        # JSON-style output
        path = RESULTS_DIR / f"phase1_report_{ts}.json"
        path.write_text(json.dumps({"report": report, "timestamp": ts}))
        print(f"Saved to {path}", flush=True)


if __name__ == "__main__":
    main()
