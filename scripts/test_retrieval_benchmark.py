"""
Tests for retrieval_benchmark.py (P1: Evaluation Benchmarks).

Covers:
- IR metrics: precision@k, recall@k, MRR, nDCG
- Golden dataset schema creation and validation
- Per-query and aggregate metric computation
- Regression detection with configurable thresholds
- Report generation
- Edge cases: empty results, no relevant items, perfect retrieval
"""

import sys
from pathlib import Path

import pytest

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from retrieval_benchmark import (
    precision_at_k,
    recall_at_k,
    mean_reciprocal_rank,
    ndcg_at_k,
    compute_query_metrics,
    compute_aggregate_metrics,
    compare_to_baseline,
    content_hash,
    make_golden_entry,
    make_expected_item,
    generate_report,
)


# ---------------------------------------------------------------------------
# precision@k
# ---------------------------------------------------------------------------

class TestPrecisionAtK:
    def test_perfect_precision(self):
        assert precision_at_k(["a", "b", "c"], {"a", "b", "c"}, 3) == 1.0

    def test_no_relevant(self):
        assert precision_at_k(["a", "b", "c"], set(), 3) == 0.0

    def test_partial_precision(self):
        assert precision_at_k(["a", "b", "c", "d"], {"a", "c"}, 4) == 0.5

    def test_k_greater_than_results(self):
        assert precision_at_k(["a"], {"a", "b"}, 5) == 1.0

    def test_k_zero(self):
        assert precision_at_k(["a"], {"a"}, 0) == 0.0

    def test_empty_retrieved(self):
        assert precision_at_k([], {"a"}, 5) == 0.0

    def test_k_one_hit(self):
        assert precision_at_k(["a", "b"], {"a"}, 1) == 1.0

    def test_k_one_miss(self):
        assert precision_at_k(["b", "a"], {"a"}, 1) == 0.0


# ---------------------------------------------------------------------------
# recall@k
# ---------------------------------------------------------------------------

class TestRecallAtK:
    def test_perfect_recall(self):
        assert recall_at_k(["a", "b", "c"], {"a", "b"}, 3) == 1.0

    def test_partial_recall(self):
        assert recall_at_k(["a", "c", "d"], {"a", "b"}, 3) == 0.5

    def test_no_recall(self):
        assert recall_at_k(["c", "d"], {"a", "b"}, 2) == 0.0

    def test_no_relevant_items(self):
        # Trivially recalled
        assert recall_at_k(["a", "b"], set(), 2) == 1.0

    def test_k_limits_window(self):
        assert recall_at_k(["c", "a"], {"a"}, 1) == 0.0
        assert recall_at_k(["c", "a"], {"a"}, 2) == 1.0


# ---------------------------------------------------------------------------
# MRR
# ---------------------------------------------------------------------------

class TestMRR:
    def test_first_position(self):
        assert mean_reciprocal_rank(["a", "b", "c"], {"a"}) == 1.0

    def test_second_position(self):
        assert mean_reciprocal_rank(["b", "a", "c"], {"a"}) == 0.5

    def test_third_position(self):
        assert mean_reciprocal_rank(["b", "c", "a"], {"a"}) == pytest.approx(1/3)

    def test_not_found(self):
        assert mean_reciprocal_rank(["b", "c", "d"], {"a"}) == 0.0

    def test_multiple_relevant_first_wins(self):
        assert mean_reciprocal_rank(["b", "a", "c"], {"a", "c"}) == 0.5

    def test_empty_retrieved(self):
        assert mean_reciprocal_rank([], {"a"}) == 0.0


# ---------------------------------------------------------------------------
# nDCG@k
# ---------------------------------------------------------------------------

class TestNDCG:
    def test_perfect_ranking(self):
        # Items in ideal order: relevance 3, 2, 1
        retrieved = ["a", "b", "c"]
        rel_map = {"a": 3, "b": 2, "c": 1}
        assert ndcg_at_k(retrieved, rel_map, 3) == pytest.approx(1.0)

    def test_reversed_ranking(self):
        # Items in worst order: 1, 2, 3
        retrieved = ["c", "b", "a"]
        rel_map = {"a": 3, "b": 2, "c": 1}
        ndcg = ndcg_at_k(retrieved, rel_map, 3)
        assert 0.0 < ndcg < 1.0  # Not perfect, not zero

    def test_no_relevant_items(self):
        assert ndcg_at_k(["a", "b"], {}, 2) == 0.0

    def test_all_irrelevant_retrieved(self):
        assert ndcg_at_k(["x", "y"], {"a": 3}, 2) == 0.0

    def test_k_zero(self):
        assert ndcg_at_k(["a"], {"a": 3}, 0) == 0.0

    def test_single_exact_match(self):
        assert ndcg_at_k(["a"], {"a": 3}, 1) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# compute_query_metrics
# ---------------------------------------------------------------------------

class TestComputeQueryMetrics:
    def test_basic_metrics(self):
        retrieved = ["a", "b", "c", "d", "e"]
        expected = [
            {"item_id": "a", "relevance": 3},
            {"item_id": "c", "relevance": 2},
            {"item_id": "z", "relevance": 1},  # Not retrieved
        ]
        metrics = compute_query_metrics(retrieved, expected, k_values=[1, 3, 5])

        assert metrics["mrr"] == 1.0  # "a" is first
        assert metrics["precision@1"] == 1.0  # "a" is relevant
        assert metrics["precision@3"] == pytest.approx(2/3)  # "a" and "c"
        assert metrics["recall@3"] == pytest.approx(2/3)  # 2 of 3 found
        assert metrics["recall@5"] == pytest.approx(2/3)  # "z" never found
        assert "ndcg@3" in metrics

    def test_empty_results(self):
        metrics = compute_query_metrics([], [{"item_id": "a", "relevance": 3}])
        assert metrics["mrr"] == 0.0
        assert metrics["precision@1"] == 0.0
        assert metrics["recall@1"] == 0.0


# ---------------------------------------------------------------------------
# compute_aggregate_metrics
# ---------------------------------------------------------------------------

class TestComputeAggregateMetrics:
    def test_aggregate(self):
        per_query = [
            {"mrr": 1.0, "precision@3": 0.67, "recall@3": 0.67},
            {"mrr": 0.5, "precision@3": 0.33, "recall@3": 0.33},
        ]
        agg = compute_aggregate_metrics(per_query)

        assert agg["mean_mrr"] == 0.75
        assert agg["mean_precision@3"] == 0.5
        assert agg["query_count"] == 2

    def test_empty(self):
        assert compute_aggregate_metrics([]) == {}


# ---------------------------------------------------------------------------
# Regression detection
# ---------------------------------------------------------------------------

class TestRegressionDetection:
    def test_no_regression(self):
        current = {"aggregate": {"mean_mrr": 0.80, "mean_precision@3": 0.70}}
        baseline = {"aggregate": {"mean_mrr": 0.78, "mean_precision@3": 0.68}}
        result = compare_to_baseline(current, baseline)
        assert result["passed"] is True
        assert result["regression_count"] == 0

    def test_regression_detected(self):
        current = {"aggregate": {"mean_mrr": 0.60, "mean_precision@3": 0.70}}
        baseline = {"aggregate": {"mean_mrr": 0.80, "mean_precision@3": 0.70}}
        result = compare_to_baseline(current, baseline, threshold=0.05)
        assert result["passed"] is False
        assert result["regression_count"] >= 1
        assert any(r["metric"] == "mean_mrr" for r in result["regressions"])

    def test_just_under_threshold(self):
        # 4% drop < 5% threshold → not regression
        current = {"aggregate": {"mean_mrr": 0.77}}
        baseline = {"aggregate": {"mean_mrr": 0.80}}
        result = compare_to_baseline(current, baseline, threshold=0.05)
        assert result["passed"] is True

    def test_improvement_not_flagged(self):
        current = {"aggregate": {"mean_mrr": 0.95}}
        baseline = {"aggregate": {"mean_mrr": 0.80}}
        result = compare_to_baseline(current, baseline)
        assert result["passed"] is True


# ---------------------------------------------------------------------------
# Golden dataset helpers
# ---------------------------------------------------------------------------

class TestGoldenDataset:
    def test_content_hash_deterministic(self):
        h1 = content_hash("test text")
        h2 = content_hash("test text")
        assert h1 == h2
        assert len(h1) == 16

    def test_different_text_different_hash(self):
        assert content_hash("text a") != content_hash("text b")

    def test_make_golden_entry(self):
        entry = make_golden_entry(
            "q1", "test query",
            [make_expected_item("item-1", 3, "item text")],
            target_pipeline="search_memory",
            query_type="code",
            tags=["test"],
        )
        assert entry["query_id"] == "q1"
        assert entry["target_pipeline"] == "search_memory"
        assert len(entry["expected_items"]) == 1
        assert entry["expected_items"][0]["relevance"] == 3
        assert entry["expected_items"][0]["content_hash"] != ""

    def test_make_expected_item_no_text(self):
        item = make_expected_item("item-1", 2)
        assert item["content_hash"] == ""
        assert item["content_snippet"] == ""


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

class TestReportGeneration:
    def test_basic_report(self):
        result = {
            "computed_at": "2026-05-13T00:00:00Z",
            "k_values": [1, 3],
            "per_query": [
                {"query_id": "q1", "pipeline": "search_memory",
                 "precision@1": 1.0, "precision@3": 0.67, "mrr": 1.0, "elapsed_ms": 50},
            ],
            "aggregate": {
                "query_count": 1,
                "mean_precision@1": 1.0,
                "min_precision@1": 1.0,
                "max_precision@1": 1.0,
                "mean_precision@3": 0.67,
                "min_precision@3": 0.67,
                "max_precision@3": 0.67,
                "mean_recall@1": 0.5,
                "min_recall@1": 0.5,
                "max_recall@1": 0.5,
                "mean_recall@3": 0.75,
                "min_recall@3": 0.75,
                "max_recall@3": 0.75,
                "mean_ndcg@1": 1.0,
                "min_ndcg@1": 1.0,
                "max_ndcg@1": 1.0,
                "mean_ndcg@3": 0.9,
                "min_ndcg@3": 0.9,
                "max_ndcg@3": 0.9,
                "mean_mrr": 1.0,
                "min_mrr": 1.0,
                "max_mrr": 1.0,
            },
        }
        report = generate_report(result)
        assert "Retrieval Benchmark Report" in report
        assert "precision@1" in report
        assert "q1" in report

    def test_report_with_regression(self):
        result = {"computed_at": "now", "k_values": [1], "per_query": [],
                  "aggregate": {"query_count": 0}}
        comparison = {
            "passed": False,
            "regression_count": 1,
            "regressions": [{"metric": "mean_mrr", "baseline": 0.8, "current": 0.6, "pct_change": -25.0}],
        }
        report = generate_report(result, comparison)
        assert "REGRESSION" in report
