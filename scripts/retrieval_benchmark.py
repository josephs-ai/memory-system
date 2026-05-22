"""
retrieval_benchmark.py — Evaluation benchmarks for retrieval quality.

Part of P1: Proving retrieval quality with real numbers.

Provides:
1. Golden dataset schema with pipeline targeting, graded relevance, content hashes
2. IR metrics: precision@k, recall@k, MRR, nDCG
3. Benchmark runner against live search_memory and context_hydrator
4. Regression detection comparing current run to saved baseline
5. JSON/markdown artifact output for CI and monitoring

Usage:
    python retrieval_benchmark.py run                          # Run full benchmark
    python retrieval_benchmark.py run --target search_memory   # Run one pipeline
    python retrieval_benchmark.py run --tags code,routing      # Filter by tags
    python retrieval_benchmark.py run --save-baseline          # Save as baseline
    python retrieval_benchmark.py compare                      # Compare to baseline
    python retrieval_benchmark.py report                       # Generate report
    python retrieval_benchmark.py seed                         # Auto-seed golden dataset
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

LOGGER = logging.getLogger("openclaw.retrieval_benchmark")

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

BENCHMARK_DIR = SCRIPT_DIR.parent / "health"
GOLDEN_PATH = BENCHMARK_DIR / "golden_dataset.json"
BASELINE_PATH = BENCHMARK_DIR / "retrieval_baseline.json"
REPORT_PATH = BENCHMARK_DIR / "retrieval_benchmark_report.json"

DEFAULT_K_VALUES = [1, 3, 5, 10]
REGRESSION_THRESHOLD = 0.05  # 5% degradation triggers a warning


# ---------------------------------------------------------------------------
# Golden Dataset Schema
# ---------------------------------------------------------------------------

GOLDEN_SCHEMA_VERSION = "1.0"

"""
Golden dataset entry schema:
{
    "query_id": str,               # Unique identifier
    "query": str,                  # Natural language query
    "target_pipeline": str,        # "search_memory" | "context_hydrator" | "both"
    "query_type": str,             # "code" | "memory" | "timeline" | "general"
    "tags": [str],                 # For filtering: ["python", "routing", "browser"]
    "expected_items": [
        {
            "item_id": str,        # Memory item or code chunk ID
            "relevance": int,      # 0=irrelevant, 1=partial, 2=relevant, 3=exact
            "content_hash": str,   # SHA-256 of item text (for resilience)
            "content_snippet": str # First 100 chars (for debugging / auto-heal)
        }
    ],
    "context_params": {            # Pipeline-specific parameters
        "project_id": str | None,
        "code_limit": int,
        "memory_limit": int,
    }
}
"""


def content_hash(text: str) -> str:
    """SHA-256 hash of item text for golden dataset resilience."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def make_golden_entry(
    query_id: str,
    query: str,
    expected_items: list[dict],
    *,
    target_pipeline: str = "both",
    query_type: str = "general",
    tags: list[str] | None = None,
    context_params: dict | None = None,
) -> dict:
    """Create a validated golden dataset entry."""
    return {
        "query_id": query_id,
        "query": query,
        "target_pipeline": target_pipeline,
        "query_type": query_type,
        "tags": tags or [],
        "expected_items": expected_items,
        "context_params": context_params or {},
    }


def make_expected_item(
    item_id: str,
    relevance: int,
    text: str = "",
) -> dict:
    """Create an expected item entry with content hash."""
    return {
        "item_id": item_id,
        "relevance": relevance,
        "content_hash": content_hash(text) if text else "",
        "content_snippet": text[:100] if text else "",
    }


# ---------------------------------------------------------------------------
# IR Metrics
# ---------------------------------------------------------------------------

def precision_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Proportion of retrieved items in top-k that are relevant."""
    if k <= 0:
        return 0.0
    top_k = retrieved_ids[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for rid in top_k if rid in relevant_ids)
    return hits / len(top_k)


def recall_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Proportion of relevant items that appear in top-k."""
    if not relevant_ids:
        return 1.0  # No relevant items → trivially recalled
    top_k = set(retrieved_ids[:k])
    hits = len(top_k & relevant_ids)
    return hits / len(relevant_ids)


def mean_reciprocal_rank(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    """Reciprocal of the rank of the first relevant item."""
    for i, rid in enumerate(retrieved_ids):
        if rid in relevant_ids:
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(
    retrieved_ids: list[str],
    relevance_map: dict[str, int],
    k: int,
) -> float:
    """
    Normalized Discounted Cumulative Gain at k.

    relevance_map: item_id → relevance grade (0-3)
    """
    if k <= 0:
        return 0.0

    # DCG
    dcg = 0.0
    for i, rid in enumerate(retrieved_ids[:k]):
        rel = relevance_map.get(rid, 0)
        dcg += (2 ** rel - 1) / math.log2(i + 2)

    # Ideal DCG
    ideal_rels = sorted(relevance_map.values(), reverse=True)[:k]
    idcg = 0.0
    for i, rel in enumerate(ideal_rels):
        idcg += (2 ** rel - 1) / math.log2(i + 2)

    if idcg == 0:
        return 0.0
    return dcg / idcg


def compute_query_metrics(
    retrieved_ids: list[str],
    expected_items: list[dict],
    k_values: list[int] | None = None,
) -> dict:
    """Compute all metrics for a single query result."""
    k_values = k_values or DEFAULT_K_VALUES

    relevant_ids = {e["item_id"] for e in expected_items if e.get("relevance", 0) >= 1}
    relevance_map = {e["item_id"]: e.get("relevance", 0) for e in expected_items}

    metrics = {
        "mrr": mean_reciprocal_rank(retrieved_ids, relevant_ids),
        "retrieved_count": len(retrieved_ids),
        "relevant_count": len(relevant_ids),
    }

    for k in k_values:
        metrics[f"precision@{k}"] = precision_at_k(retrieved_ids, relevant_ids, k)
        metrics[f"recall@{k}"] = recall_at_k(retrieved_ids, relevant_ids, k)
        metrics[f"ndcg@{k}"] = ndcg_at_k(retrieved_ids, relevance_map, k)

    return metrics


def compute_aggregate_metrics(per_query_metrics: list[dict]) -> dict:
    """Compute mean metrics across all queries."""
    if not per_query_metrics:
        return {}

    all_keys = set()
    for m in per_query_metrics:
        all_keys.update(m.keys())

    # Only average numeric keys
    numeric_keys = [k for k in all_keys if isinstance(per_query_metrics[0].get(k), (int, float))]

    agg = {}
    for key in numeric_keys:
        values = [m[key] for m in per_query_metrics if key in m]
        if values:
            agg[f"mean_{key}"] = sum(values) / len(values)
            agg[f"min_{key}"] = min(values)
            agg[f"max_{key}"] = max(values)

    agg["query_count"] = len(per_query_metrics)
    return agg


# ---------------------------------------------------------------------------
# Benchmark Runner
# ---------------------------------------------------------------------------

_EMBED_MODEL = None


def _get_embed_model():
    """Lazy-load the sentence-transformer embedding model (MiniLM, 384-dim)."""
    global _EMBED_MODEL
    if _EMBED_MODEL is None:
        from sentence_transformers import SentenceTransformer
        _EMBED_MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _EMBED_MODEL


def _embed_query(text: str) -> list[float]:
    """Embed a query string using MiniLM (matches existing DB embeddings)."""
    model = _get_embed_model()
    emb = model.encode(text, normalize_embeddings=True)
    return emb.tolist()


def run_search_memory_query(query: str, top_k: int = 10, project_id: str | None = None) -> list[str]:
    """Execute a query against search_memory pipeline, return item IDs in order."""
    from memory_db import hybrid_search_memory_items

    emb = _embed_query(query)

    results = hybrid_search_memory_items(
        emb,
        query_text=query,
        status="active",
        allowed_sensitivities=["public", "internal"],
        limit=top_k,
    )

    return [r.get("id", "") for r in results if r.get("id")]


def run_context_hydrator_query(query: str, memory_limit: int = 10, code_limit: int = 0) -> list[str]:
    """Execute a query against context_hydrator, return item IDs in order."""
    from context_hydrator import build_context_pack

    pack = build_context_pack(
        query,
        memory_limit=memory_limit,
        code_limit=code_limit,
        rerank=True,
        include_graph=False,  # Graph adds latency, not needed for benchmarks
    )

    ids = []
    # Memory context items
    for m in pack.get("memory_context", []):
        if m.get("id"):
            ids.append(m["id"])
    # Code context items (use qualified_name as ID proxy)
    for c in pack.get("code_context", []):
        if c.get("qualified_name"):
            ids.append(c["qualified_name"])
    return ids


def run_benchmark(
    golden_data: list[dict],
    *,
    target: str | None = None,
    tags: list[str] | None = None,
    k_values: list[int] | None = None,
) -> dict:
    """
    Run the full benchmark suite.

    Args:
        golden_data: List of golden dataset entries
        target: Filter by target_pipeline ("search_memory" | "context_hydrator")
        tags: Filter by tags (any match)
        k_values: Which k values to compute metrics for

    Returns:
        Benchmark result dict with per-query and aggregate metrics
    """
    k_values = k_values or DEFAULT_K_VALUES

    # Filter entries
    entries = golden_data
    if target:
        entries = [e for e in entries if e["target_pipeline"] in (target, "both")]
    if tags:
        tag_set = set(tags)
        entries = [e for e in entries if tag_set & set(e.get("tags", []))]

    results = []
    for entry in entries:
        pipeline = entry["target_pipeline"]
        query = entry["query"]
        params = entry.get("context_params", {})
        t0 = time.monotonic()

        try:
            if pipeline in ("search_memory", "both"):
                retrieved = run_search_memory_query(
                    query,
                    top_k=max(k_values) if k_values else 10,
                    project_id=params.get("project_id"),
                )
            else:
                retrieved = run_context_hydrator_query(
                    query,
                    memory_limit=params.get("memory_limit", 10),
                    code_limit=params.get("code_limit", 0),
                )
        except Exception as e:
            LOGGER.error("Query %s failed: %s", entry["query_id"], e)
            retrieved = []

        elapsed_ms = round((time.monotonic() - t0) * 1000, 1)

        metrics = compute_query_metrics(retrieved, entry["expected_items"], k_values)
        metrics["query_id"] = entry["query_id"]
        metrics["query"] = query
        metrics["pipeline"] = pipeline
        metrics["elapsed_ms"] = elapsed_ms
        metrics["retrieved_ids"] = retrieved[:max(k_values) if k_values else 10]

        results.append(metrics)

    aggregate = compute_aggregate_metrics(results)

    return {
        "type": "retrieval_benchmark_result",
        "computed_at": datetime.now(timezone.utc).isoformat(),
        "schema_version": GOLDEN_SCHEMA_VERSION,
        "k_values": k_values,
        "target_filter": target,
        "tag_filter": tags,
        "per_query": results,
        "aggregate": aggregate,
    }


# ---------------------------------------------------------------------------
# Regression Detection
# ---------------------------------------------------------------------------

def compare_to_baseline(
    current: dict,
    baseline: dict,
    *,
    threshold: float = REGRESSION_THRESHOLD,
) -> dict:
    """
    Compare current benchmark results to a saved baseline.

    Returns a comparison dict with per-metric deltas and regression flags.
    """
    curr_agg = current.get("aggregate", {})
    base_agg = baseline.get("aggregate", {})

    comparisons = []
    regressions = []

    for key in sorted(curr_agg.keys()):
        if not key.startswith("mean_"):
            continue
        curr_val = curr_agg.get(key)
        base_val = base_agg.get(key)
        if curr_val is None or base_val is None:
            continue

        delta = curr_val - base_val
        pct_change = delta / base_val if base_val != 0 else 0.0

        is_regression = pct_change < -threshold
        entry = {
            "metric": key,
            "baseline": round(base_val, 4),
            "current": round(curr_val, 4),
            "delta": round(delta, 4),
            "pct_change": round(pct_change * 100, 2),
            "regression": is_regression,
        }
        comparisons.append(entry)
        if is_regression:
            regressions.append(entry)

    return {
        "type": "retrieval_benchmark_comparison",
        "computed_at": datetime.now(timezone.utc).isoformat(),
        "threshold_pct": threshold * 100,
        "comparisons": comparisons,
        "regression_count": len(regressions),
        "regressions": regressions,
        "passed": len(regressions) == 0,
    }


# ---------------------------------------------------------------------------
# Golden Dataset Management
# ---------------------------------------------------------------------------

def load_golden_dataset(path: Path | None = None) -> list[dict]:
    """Load golden dataset from JSON file."""
    path = path or GOLDEN_PATH
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("entries", []) if isinstance(data, dict) else data


def save_golden_dataset(entries: list[dict], path: Path | None = None):
    """Save golden dataset to JSON file."""
    path = path or GOLDEN_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "schema_version": GOLDEN_SCHEMA_VERSION,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "entry_count": len(entries),
        "entries": entries,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def save_baseline(result: dict, path: Path | None = None):
    """Save benchmark result as baseline."""
    path = path or BASELINE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)


def load_baseline(path: Path | None = None) -> dict | None:
    """Load saved baseline."""
    path = path or BASELINE_PATH
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Auto-seed golden dataset from live DB
# ---------------------------------------------------------------------------

SEED_QUERIES = [
    {
        "query_id": "browser-config",
        "query": "browser configuration port settings",
        "target_pipeline": "search_memory",
        "query_type": "memory",
        "tags": ["browser", "config"],
    },
    {
        "query_id": "memory-pipeline",
        "query": "memory pipeline dehydration chunking extraction",
        "target_pipeline": "search_memory",
        "query_type": "code",
        "tags": ["pipeline", "architecture"],
    },
    {
        "query_id": "neo4j-graph",
        "query": "Neo4j code graph schema relationships",
        "target_pipeline": "search_memory",
        "query_type": "code",
        "tags": ["graph", "neo4j", "code"],
    },
    {
        "query_id": "state-machine",
        "query": "orchestrator state machine transitions lifecycle",
        "target_pipeline": "search_memory",
        "query_type": "code",
        "tags": ["orchestrator", "state-machine"],
    },
    {
        "query_id": "feedback-scoring",
        "query": "retrieval feedback scoring useful bad",
        "target_pipeline": "search_memory",
        "query_type": "code",
        "tags": ["feedback", "retrieval"],
    },
    {
        "query_id": "path-watcher",
        "query": "tracked path watcher file monitoring changes",
        "target_pipeline": "search_memory",
        "query_type": "code",
        "tags": ["watcher", "monitoring"],
    },
    {
        "query_id": "project-scope",
        "query": "project scoped memory isolation boundaries",
        "target_pipeline": "search_memory",
        "query_type": "memory",
        "tags": ["project", "scope"],
    },
    {
        "query_id": "reranking",
        "query": "cross-encoder reranking retrieval scoring",
        "target_pipeline": "search_memory",
        "query_type": "code",
        "tags": ["reranking", "retrieval"],
    },
    {
        "query_id": "code-ast-parser",
        "query": "AST parser Python function class extraction",
        "target_pipeline": "context_hydrator",
        "query_type": "code",
        "tags": ["parser", "ast", "code"],
    },
    {
        "query_id": "embedding-pipeline",
        "query": "embed memory items vectors qdrant",
        "target_pipeline": "context_hydrator",
        "query_type": "code",
        "tags": ["embedding", "qdrant"],
    },
]


def seed_golden_dataset():
    """
    Auto-seed the golden dataset by running seed queries against the live DB
    and using the top results as "expected" items with graded relevance.

    Top-1 gets relevance=3 (exact), top 2-3 get relevance=2 (relevant),
    top 4-5 get relevance=1 (partial). Human review recommended after seeding.
    """
    from memory_db import hybrid_search_memory_items, close_pool

    entries = []

    for seed in SEED_QUERIES:
        emb = _embed_query(seed["query"])
        results = hybrid_search_memory_items(
            emb,
            query_text=seed["query"],
            status="active",
            allowed_sensitivities=["public", "internal"],
            limit=5,
        )

        expected = []
        for i, r in enumerate(results):
            text = r.get("text", "")
            if i == 0:
                relevance = 3
            elif i <= 2:
                relevance = 2
            else:
                relevance = 1

            expected.append(make_expected_item(
                item_id=r.get("id", ""),
                relevance=relevance,
                text=text,
            ))

        entry = make_golden_entry(
            query_id=seed["query_id"],
            query=seed["query"],
            expected_items=expected,
            target_pipeline=seed["target_pipeline"],
            query_type=seed["query_type"],
            tags=seed["tags"],
        )
        entries.append(entry)

    save_golden_dataset(entries)
    close_pool()
    return entries


# ---------------------------------------------------------------------------
# Report Generation
# ---------------------------------------------------------------------------

def generate_report(result: dict, comparison: dict | None = None) -> str:
    """Generate a human-readable markdown report."""
    lines = ["# Retrieval Benchmark Report", ""]
    lines.append(f"**Generated:** {result.get('computed_at', 'unknown')}")
    lines.append(f"**Queries:** {result['aggregate'].get('query_count', 0)}")
    lines.append("")

    # Aggregate metrics
    agg = result.get("aggregate", {})
    lines.append("## Aggregate Metrics")
    lines.append("")
    lines.append("| Metric | Mean | Min | Max |")
    lines.append("|--------|------|-----|-----|")
    for k_val in result.get("k_values", DEFAULT_K_VALUES):
        for metric in [f"precision@{k_val}", f"recall@{k_val}", f"ndcg@{k_val}"]:
            mean = agg.get(f"mean_{metric}", "-")
            mn = agg.get(f"min_{metric}", "-")
            mx = agg.get(f"max_{metric}", "-")
            if isinstance(mean, float):
                mean = f"{mean:.4f}"
                mn = f"{mn:.4f}" if isinstance(mn, float) else mn
                mx = f"{mx:.4f}" if isinstance(mx, float) else mx
            lines.append(f"| {metric} | {mean} | {mn} | {mx} |")

    mrr_mean = agg.get("mean_mrr", "-")
    if isinstance(mrr_mean, float):
        mrr_mean = f"{mrr_mean:.4f}"
    lines.append(f"| MRR | {mrr_mean} | - | - |")
    lines.append("")

    # Comparison
    if comparison:
        lines.append("## Regression Check")
        lines.append("")
        if comparison.get("passed"):
            lines.append("✅ **PASSED** — No regressions detected.")
        else:
            lines.append(f"⚠️ **{comparison['regression_count']} REGRESSION(S) DETECTED**")
            for reg in comparison.get("regressions", []):
                lines.append(f"  - {reg['metric']}: {reg['baseline']:.4f} → {reg['current']:.4f} ({reg['pct_change']:+.1f}%)")
        lines.append("")

    # Per-query
    lines.append("## Per-Query Results")
    lines.append("")
    for q in result.get("per_query", []):
        p1 = q.get("precision@1", 0)
        p3 = q.get("precision@3", 0)
        mrr = q.get("mrr", 0)
        lines.append(f"- **{q['query_id']}** ({q.get('pipeline', '?')}): P@1={p1:.2f} P@3={p3:.2f} MRR={mrr:.2f} [{q.get('elapsed_ms', '?')}ms]")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Retrieval benchmark suite")
    sub = parser.add_subparsers(dest="command")

    # seed
    sub.add_parser("seed", help="Auto-seed golden dataset from live DB")

    # run
    run_p = sub.add_parser("run", help="Run benchmark suite")
    run_p.add_argument("--target", choices=["search_memory", "context_hydrator"])
    run_p.add_argument("--tags", type=str, help="Comma-separated tags")
    run_p.add_argument("--save-baseline", action="store_true")
    run_p.add_argument("--output", type=str)

    # compare
    sub.add_parser("compare", help="Compare latest run to baseline")

    # report
    report_p = sub.add_parser("report", help="Generate markdown report")
    report_p.add_argument("--output", type=str)

    args = parser.parse_args()

    if args.command == "seed":
        entries = seed_golden_dataset()
        print(f"Seeded {len(entries)} golden entries to {GOLDEN_PATH}")

    elif args.command == "run":
        golden = load_golden_dataset()
        if not golden:
            print("No golden dataset found. Run 'seed' first.")
            sys.exit(1)

        tags = args.tags.split(",") if args.tags else None
        result = run_benchmark(golden, target=args.target, tags=tags)

        if args.save_baseline:
            save_baseline(result)
            print(f"Baseline saved to {BASELINE_PATH}")

        out_path = args.output or str(REPORT_PATH)
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"Results written to {out_path}")
        print(json.dumps(result["aggregate"], indent=2))

    elif args.command == "compare":
        baseline = load_baseline()
        if not baseline:
            print("No baseline found. Run with --save-baseline first.")
            sys.exit(1)

        golden = load_golden_dataset()
        current = run_benchmark(golden)
        comparison = compare_to_baseline(current, baseline)
        print(json.dumps(comparison, indent=2))

        if not comparison["passed"]:
            sys.exit(1)

    elif args.command == "report":
        if not REPORT_PATH.exists():
            print("No benchmark results found. Run 'run' first.")
            sys.exit(1)
        with open(REPORT_PATH) as f:
            result = json.load(f)

        baseline = load_baseline()
        comparison = compare_to_baseline(result, baseline) if baseline else None

        report_md = generate_report(result, comparison)
        if args.output:
            with open(args.output, "w") as f:
                f.write(report_md)
            print(f"Report written to {args.output}")
        else:
            print(report_md)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
