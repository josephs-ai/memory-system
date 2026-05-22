"""
Tests for P0-S3: auto_demote_bad_items.py and retrieval_metrics.py.

Covers:
- Demotion candidate scanning with various feedback distributions
- Threshold behavior (min_bad, min_distinct_queries)
- Shortlist artifact generation
- Metric computation (rates, totals, by_query_type)
- Edge cases: no feedback, all-useful, all-bad, NULL timestamps
"""

import sys
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from auto_demote_bad_items import (
    scan_demotion_candidates,
    generate_shortlist_artifact,
)
from retrieval_metrics import compute_aggregate_metrics


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_conn_with_queries(query_results: dict):
    """
    Create a mock conn where cur.execute() returns different results
    based on the SQL query content. query_results maps a substring → rows.
    """
    mock_cur = MagicMock()
    _call_count = [0]
    _stored_results = [[]]


    def fake_execute(sql, params=None):
        for substring, rows in query_results.items():
            if substring in sql:
                _stored_results[0] = rows
                return
        _stored_results[0] = []

    mock_cur.execute = fake_execute
    mock_cur.fetchall = lambda: _stored_results[0]
    mock_cur.fetchone = lambda: _stored_results[0][0] if _stored_results[0] else (0, 0, 0, 0, 0)
    mock_cur.__enter__ = lambda s: s
    mock_cur.__exit__ = MagicMock(return_value=False)

    mock_conn = MagicMock()
    mock_conn.cursor.return_value = mock_cur
    return mock_conn


# ---------------------------------------------------------------------------
# auto_demote_bad_items tests
# ---------------------------------------------------------------------------

class TestScanDemotionCandidates:
    def test_no_bad_items_returns_empty(self):
        conn = _mock_conn_with_queries({
            "item_stats": [],
            "DISTINCT query_text": [],
        })
        result = scan_demotion_candidates(conn=conn)
        assert result == []

    def test_item_above_threshold_flagged(self):
        now = datetime.now(timezone.utc)
        conn = _mock_conn_with_queries({
            "item_stats": [
                ("mem-bad-1", 5, 1, 3, now),
            ],
            "DISTINCT query_text": [
                ("what is the config",),
                ("how to set up",),
                ("where is the file",),
            ],
        })
        result = scan_demotion_candidates(min_bad=3, min_distinct_queries=2, conn=conn)
        assert len(result) == 1
        assert result[0]["item_id"] == "mem-bad-1"
        assert result[0]["bad_count"] == 5
        assert result[0]["useful_count"] == 1
        assert result[0]["net_score"] == -4
        assert len(result[0]["sample_queries"]) == 3

    def test_net_positive_item_excluded(self):
        """Items with more useful than bad should not appear."""
        datetime.now(timezone.utc)
        # The SQL WHERE clause filters bad_count > useful_count
        # So a net-positive item won't be returned by the query
        conn = _mock_conn_with_queries({
            "item_stats": [],  # SQL filters it out
        })
        result = scan_demotion_candidates(conn=conn)
        assert result == []

    def test_below_min_bad_excluded(self):
        """Items with fewer bad marks than threshold are excluded."""
        conn = _mock_conn_with_queries({
            "item_stats": [],  # SQL HAVING filters it
        })
        result = scan_demotion_candidates(min_bad=5, conn=conn)
        assert result == []

    def test_distinct_query_filter_applied(self):
        """Items bad from only 1 query are excluded if min_distinct_queries=2."""
        now = datetime.now(timezone.utc)
        conn = _mock_conn_with_queries({
            "item_stats": [
                ("mem-spam", 4, 0, 1, now),  # distinct_bad_queries=1
            ],
            "DISTINCT query_text": [
                ("same query every time",),
            ],
        })
        result = scan_demotion_candidates(min_bad=3, min_distinct_queries=2, conn=conn)
        # distinct_bad_queries=1 < min_distinct_queries=2, so filtered out
        assert result == []

    def test_null_query_text_relaxes_distinct_check(self):
        """Pre-S2 feedback (no query_text) has distinct_bad_queries=0, which relaxes the filter."""
        now = datetime.now(timezone.utc)
        conn = _mock_conn_with_queries({
            "item_stats": [
                ("mem-old", 5, 0, 0, now),  # distinct=0 (no query_text stored)
            ],
            "DISTINCT query_text": [],
        })
        result = scan_demotion_candidates(min_bad=3, min_distinct_queries=2, conn=conn)
        # distinct=0 should pass through (backward compat)
        assert len(result) == 1


class TestShortlistArtifact:
    def test_artifact_structure(self):
        candidates = [
            {"item_id": "mem-1", "bad_count": 5, "useful_count": 1,
             "net_score": -4, "distinct_bad_queries": 3,
             "last_bad_at": "2026-05-13T00:00:00+00:00", "sample_queries": []},
        ]
        artifact = generate_shortlist_artifact(candidates)

        assert artifact["type"] == "demotion_shortlist"
        assert "generated_at" in artifact
        assert artifact["candidate_count"] == 1
        assert artifact["candidates"] == candidates
        assert "action_required" in artifact

    def test_empty_artifact(self):
        artifact = generate_shortlist_artifact([])
        assert artifact["candidate_count"] == 0
        assert artifact["candidates"] == []


# ---------------------------------------------------------------------------
# retrieval_metrics tests
# ---------------------------------------------------------------------------

class TestComputeAggregateMetrics:
    """Tests using a sequential mock that returns different results per execute() call."""

    def _mock_sequential_conn(self, call_sequence: list):
        """
        Create a mock conn where each cursor.execute() call returns
        the next item in call_sequence. Each item is a dict with:
            fetchone: tuple or None
            fetchall: list of tuples
        """
        mock_cur = MagicMock()
        _idx = [0]

        def fake_execute(sql, params=None):
            pass  # Just advance on fetch

        def fake_fetchone():
            if _idx[0] < len(call_sequence):
                result = call_sequence[_idx[0]].get("fetchone", (0, 0, 0, 0, 0))
                _idx[0] += 1
                return result
            return (0, 0, 0, 0, 0)

        def fake_fetchall():
            if _idx[0] < len(call_sequence):
                result = call_sequence[_idx[0]].get("fetchall", [])
                _idx[0] += 1
                return result
            return []

        mock_cur.execute = fake_execute
        mock_cur.fetchone = fake_fetchone
        mock_cur.fetchall = fake_fetchall
        mock_cur.__enter__ = lambda s: s
        mock_cur.__exit__ = MagicMock(return_value=False)

        mock_conn = MagicMock()
        mock_conn.cursor.return_value = mock_cur
        return mock_conn

    def test_basic_metrics(self):
        # compute_aggregate_metrics makes 4 queries in order:
        # 1. Overall counts (fetchone)
        # 2. Top useful (fetchall)
        # 3. Top bad (fetchall)
        # 4. By query type (fetchall)
        conn = self._mock_sequential_conn([
            {"fetchone": (29, 4, 1, 24, 3)},
            {"fetchall": [("canon-browser-0001", 3)]},
            {"fetchall": [("canon-browser-0002", 1)]},
            {"fetchall": [("unknown", 25, 4, 1)]},
        ])
        metrics = compute_aggregate_metrics(window_days=30, conn=conn)

        assert metrics["type"] == "retrieval_quality_metrics"
        assert metrics["window"]["days"] == 30
        assert metrics["totals"]["total_feedback"] == 29
        assert metrics["totals"]["useful"] == 4
        assert metrics["totals"]["bad"] == 1
        assert metrics["rates"]["useful_rate"] == 0.8
        assert metrics["rates"]["bad_rate"] == 0.2

    def test_no_feedback_metrics(self):
        conn = self._mock_sequential_conn([
            {"fetchone": (0, 0, 0, 0, 0)},
            {"fetchall": []},
            {"fetchall": []},
            {"fetchall": []},
        ])
        metrics = compute_aggregate_metrics(conn=conn)

        assert metrics["totals"]["total_feedback"] == 0
        assert metrics["rates"]["useful_rate"] is None
        assert metrics["rates"]["feedback_rate"] is None

    def test_all_useful_metrics(self):
        conn = self._mock_sequential_conn([
            {"fetchone": (10, 10, 0, 0, 5)},
            {"fetchall": [("item-a", 5), ("item-b", 3), ("item-c", 2)]},
            {"fetchall": []},
            {"fetchall": [("code", 6, 6, 0), ("memory", 4, 4, 0)]},
        ])
        metrics = compute_aggregate_metrics(conn=conn)

        assert metrics["rates"]["useful_rate"] == 1.0
        assert metrics["rates"]["bad_rate"] == 0.0
        assert len(metrics["top_useful_items"]) == 3
        assert len(metrics["top_bad_items"]) == 0
        assert "code" in metrics["by_query_type"]
        assert metrics["by_query_type"]["code"]["useful_rate"] == 1.0

    def test_by_query_type_breakdown(self):
        conn = self._mock_sequential_conn([
            {"fetchone": (20, 10, 5, 5, 8)},
            {"fetchall": [("item-a", 5)]},
            {"fetchall": [("item-b", 3)]},
            {"fetchall": [
                ("code", 8, 6, 2),
                ("memory", 5, 3, 1),
                ("timeline", 4, 1, 2),
                ("unknown", 3, 0, 0),
            ]},
        ])
        metrics = compute_aggregate_metrics(conn=conn)

        by_qt = metrics["by_query_type"]
        assert "code" in by_qt
        assert by_qt["code"]["useful_rate"] == pytest.approx(0.75)
        assert "timeline" in by_qt
        assert by_qt["timeline"]["useful_rate"] == pytest.approx(1/3, abs=0.01)

    def test_window_parameter(self):
        conn = self._mock_sequential_conn([
            {"fetchone": (5, 2, 1, 2, 3)},
            {"fetchall": []},
            {"fetchall": []},
            {"fetchall": []},
        ])
        metrics = compute_aggregate_metrics(window_days=7, conn=conn)
        assert metrics["window"]["days"] == 7
