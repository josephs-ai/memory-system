"""
Tests for feedback_score_engine.py (P0 S1).

Tests cover:
- Query type classification
- Recency weighting
- Cold-start damping
- Feedback boost calculation (single + batch)
- Edge cases: no feedback, all-useful, all-bad, mixed, NULL timestamps
- Clamping behavior
"""

import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import MagicMock

import pytest

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from feedback_score_engine import (
    classify_query_type,
    _recency_weight,
    _cold_start_factor,
    feedback_boost,
    feedback_boost_batch,
    USEFUL_WEIGHT,
    BAD_WEIGHT,
    MAX_BOOST,
    MIN_BOOST,
    RECENCY_HALF_LIFE_DAYS,
    COLD_START_DAMPING,
)


# ---------------------------------------------------------------------------
# classify_query_type
# ---------------------------------------------------------------------------

class TestClassifyQueryType:
    def test_empty_returns_general(self):
        assert classify_query_type("") == "general"
        assert classify_query_type(None) == "general"

    def test_code_query(self):
        assert classify_query_type("fix the parse_code_ast.py module") == "code"
        assert classify_query_type("what function handles import resolution") == "code"
        assert classify_query_type("refactor the API endpoint schema migration") == "code"

    def test_timeline_query(self):
        assert classify_query_type("what happened yesterday with the deploy") == "timeline"
        assert classify_query_type("events from 2026-04-15") == "timeline"
        assert classify_query_type("what changed last week") == "timeline"

    def test_memory_query(self):
        assert classify_query_type("what was the decision about workflow conventions") == "memory"
        assert classify_query_type("remember the preference for code style") == "memory"

    def test_ambiguous_returns_general(self):
        assert classify_query_type("hello world") == "general"
        assert classify_query_type("what is the status") == "general"

    def test_mixed_prefers_strongest(self):
        # "function" is code, "remember" is memory — code should win if more hits
        result = classify_query_type("function class method implementation")
        assert result == "code"


# ---------------------------------------------------------------------------
# _recency_weight
# ---------------------------------------------------------------------------

class TestRecencyWeight:
    def test_none_returns_moderate(self):
        assert _recency_weight(None) == 0.5

    def test_now_returns_near_one(self):
        now = datetime.now(timezone.utc)
        w = _recency_weight(now)
        assert 0.95 < w <= 1.0

    def test_half_life_returns_half(self):
        past = datetime.now(timezone.utc) - timedelta(days=RECENCY_HALF_LIFE_DAYS)
        w = _recency_weight(past)
        assert 0.45 < w < 0.55

    def test_double_half_life_returns_quarter(self):
        past = datetime.now(timezone.utc) - timedelta(days=RECENCY_HALF_LIFE_DAYS * 2)
        w = _recency_weight(past)
        assert 0.20 < w < 0.30

    def test_naive_datetime_treated_as_utc(self):
        past = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(days=1)
        w = _recency_weight(past)
        assert 0.9 < w < 1.0


# ---------------------------------------------------------------------------
# _cold_start_factor
# ---------------------------------------------------------------------------

class TestColdStartFactor:
    def test_zero_returns_zero(self):
        assert _cold_start_factor(0) == 0.0

    def test_one_returns_fraction(self):
        f = _cold_start_factor(1)
        assert f == pytest.approx(1.0 / COLD_START_DAMPING)

    def test_at_threshold_returns_one(self):
        assert _cold_start_factor(COLD_START_DAMPING) == 1.0

    def test_above_threshold_capped_at_one(self):
        assert _cold_start_factor(COLD_START_DAMPING * 10) == 1.0

    def test_negative_returns_zero(self):
        assert _cold_start_factor(-5) == 0.0


# ---------------------------------------------------------------------------
# feedback_boost (mocked DB)
# ---------------------------------------------------------------------------

class TestFeedbackBoost:
    def _mock_conn(self, rows):
        """Create a mock connection that returns the given rows."""
        mock_cur = MagicMock()
        mock_cur.fetchall.return_value = rows
        mock_cur.__enter__ = lambda s: s
        mock_cur.__exit__ = MagicMock(return_value=False)

        mock_conn = MagicMock()
        mock_conn.cursor.return_value = mock_cur
        return mock_conn

    def test_no_feedback_returns_zero(self):
        conn = self._mock_conn([])
        assert feedback_boost("item-1", conn=conn) == 0.0

    def test_single_useful_recent(self):
        now = datetime.now(timezone.utc)
        conn = self._mock_conn([("useful", now, None)])
        boost = feedback_boost("item-1", conn=conn)
        raw = USEFUL_WEIGHT * 1.0 * (1.0 / COLD_START_DAMPING)
        expected = min(raw, MAX_BOOST)
        assert boost == pytest.approx(expected, abs=0.05)
        assert boost > 0

    def test_single_bad_recent(self):
        now = datetime.now(timezone.utc)
        conn = self._mock_conn([("bad", now, None)])
        boost = feedback_boost("item-1", conn=conn)
        raw = BAD_WEIGHT * 1.0 * (1.0 / COLD_START_DAMPING)
        expected = max(raw, MIN_BOOST)
        assert boost == pytest.approx(expected, abs=0.05)
        assert boost < 0

    def test_mixed_feedback_net_score(self):
        now = datetime.now(timezone.utc)
        rows = [
            ("useful", now, None),
            ("useful", now, None),
            ("bad", now, None),
        ]
        conn = self._mock_conn(rows)
        boost = feedback_boost("item-1", conn=conn)
        raw = (2 * USEFUL_WEIGHT + 1 * BAD_WEIGHT) * _cold_start_factor(3)
        expected = max(MIN_BOOST, min(MAX_BOOST, raw))
        assert boost == pytest.approx(expected, abs=0.05)
        assert boost > 0

    def test_all_bad_clamps_to_min(self):
        now = datetime.now(timezone.utc)
        rows = [("bad", now, None)] * 10
        conn = self._mock_conn(rows)
        boost = feedback_boost("item-1", conn=conn)
        assert boost == MIN_BOOST

    def test_all_useful_clamps_to_max(self):
        now = datetime.now(timezone.utc)
        rows = [("useful", now, None)] * 10
        conn = self._mock_conn(rows)
        boost = feedback_boost("item-1", conn=conn)
        assert boost == MAX_BOOST

    def test_old_feedback_weighted_less(self):
        now = datetime.now(timezone.utc)
        old = now - timedelta(days=RECENCY_HALF_LIFE_DAYS * 3)
        rows_recent = [("useful", now, None)]
        rows_old = [("useful", old, None)]

        conn_recent = self._mock_conn(rows_recent)
        conn_old = self._mock_conn(rows_old)

        boost_recent = feedback_boost("item-1", conn=conn_recent)
        boost_old = feedback_boost("item-1", conn=conn_old)

        assert boost_recent > boost_old
        assert boost_old > 0

    def test_exception_returns_zero(self):
        mock_conn = MagicMock()
        mock_conn.cursor.side_effect = Exception("DB down")
        assert feedback_boost("item-1", conn=mock_conn) == 0.0


# ---------------------------------------------------------------------------
# feedback_boost_batch (mocked DB)
# ---------------------------------------------------------------------------

class TestFeedbackBoostBatch:
    def _mock_conn(self, rows):
        mock_cur = MagicMock()
        mock_cur.fetchall.return_value = rows
        mock_cur.__enter__ = lambda s: s
        mock_cur.__exit__ = MagicMock(return_value=False)

        mock_conn = MagicMock()
        mock_conn.cursor.return_value = mock_cur
        return mock_conn

    def test_empty_ids_returns_empty(self):
        result = feedback_boost_batch([])
        assert result == {}

    def test_batch_with_mixed_items(self):
        now = datetime.now(timezone.utc)
        rows = [
            ("item-a", "useful", now, None),
            ("item-a", "useful", now, None),
            ("item-a", "useful", now, None),
            ("item-b", "bad", now, None),
            ("item-b", "bad", now, None),
            ("item-b", "bad", now, None),
        ]
        conn = self._mock_conn(rows)
        result = feedback_boost_batch(["item-a", "item-b", "item-c"], conn=conn)

        assert result["item-a"] > 0  # useful
        assert result["item-b"] < 0  # bad
        assert result["item-c"] == 0.0  # no feedback

    def test_exception_returns_zeros(self):
        mock_conn = MagicMock()
        mock_conn.cursor.side_effect = Exception("DB down")
        result = feedback_boost_batch(["item-1", "item-2"], conn=mock_conn)
        assert result == {"item-1": 0.0, "item-2": 0.0}
