"""
Tests for P0-S4: Per-query-type preference learning.

Covers:
- Type-specific feedback blending activates above TYPED_COLD_START threshold
- Type-specific feedback stays dormant below threshold (global-only)
- Cross-type isolation: item good for code but bad for memory
- Blend weight correctness
- Batch path handles typed scoring
- "general" query_type falls back to global-only
"""

import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import MagicMock


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from feedback_score_engine import (
    feedback_boost,
    feedback_boost_batch,
    _compute_weighted_sum,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_conn(rows):
    """Create a mock connection that returns rows with (feedback, timestamp, query_type)."""
    mock_cur = MagicMock()
    mock_cur.fetchall.return_value = rows
    mock_cur.__enter__ = lambda s: s
    mock_cur.__exit__ = MagicMock(return_value=False)

    mock_conn = MagicMock()
    mock_conn.cursor.return_value = mock_cur
    return mock_conn


# ---------------------------------------------------------------------------
# _compute_weighted_sum (extracted helper)
# ---------------------------------------------------------------------------

class TestComputeWeightedSum:
    def test_empty(self):
        assert _compute_weighted_sum([]) == 0.0

    def test_single_useful(self):
        now = datetime.now(timezone.utc)
        result = _compute_weighted_sum([("useful", now)])
        assert result > 0

    def test_single_bad(self):
        now = datetime.now(timezone.utc)
        result = _compute_weighted_sum([("bad", now)])
        assert result < 0


# ---------------------------------------------------------------------------
# Per-query-type scoring (single item)
# ---------------------------------------------------------------------------

class TestQueryTypeScoring:
    def test_global_only_when_no_query_type(self):
        """Without query_type, global scoring is used."""
        now = datetime.now(timezone.utc)
        rows = [
            ("useful", now, "code"),
            ("useful", now, "code"),
            ("useful", now, "code"),
        ]
        conn = _mock_conn(rows)
        boost_global = feedback_boost("item-1", query_type=None, conn=conn)

        conn2 = _mock_conn(rows)
        boost_general = feedback_boost("item-1", query_type="general", conn=conn2)

        # Both should be the same (global-only)
        assert boost_global == boost_general

    def test_typed_activates_above_cold_start(self):
        """Type-specific scoring activates when ≥TYPED_COLD_START typed rows exist."""
        # Use older timestamps so raw scores stay below clamp boundaries
        now = datetime.now(timezone.utc)
        mid = now - timedelta(days=20)  # Moderate recency decay

        # 5 rows: 2 useful for code, 3 bad for memory
        # Global: net negative (2 useful + 3 bad → 2*1.0 + 3*(-1.5) = -2.5)
        # Code-typed: 2 useful → positive
        # Memory-typed: 3 bad → very negative
        rows = [
            ("useful", mid, "code"),
            ("useful", mid, "code"),
            ("bad", mid, "memory"),
            ("bad", mid, "memory"),
            ("bad", mid, "memory"),
        ]

        # Query as "code" — typed (2 useful) blended with global (net negative)
        conn_code = _mock_conn(rows)
        boost_code = feedback_boost("item-1", query_type="code", conn=conn_code)

        # Query as "memory" — typed (3 bad) blended with global (net negative)
        conn_mem = _mock_conn(rows)
        boost_memory = feedback_boost("item-1", query_type="memory", conn=conn_mem)

        # Code should be higher than memory (typed code is positive, typed memory is negative)
        assert boost_code > boost_memory

    def test_typed_below_cold_start_uses_global(self):
        """Below TYPED_COLD_START, typed scoring is not applied."""
        now = datetime.now(timezone.utc)
        rows = [
            ("useful", now, "code"),       # Only 1 code-typed row
            ("useful", now, "memory"),
            ("useful", now, "memory"),
        ]
        conn_typed = _mock_conn(rows)
        boost_typed = feedback_boost("item-1", query_type="code", conn=conn_typed)

        conn_global = _mock_conn(rows)
        boost_global = feedback_boost("item-1", query_type=None, conn=conn_global)

        # Should be equal because typed code has only 1 row < TYPED_COLD_START
        assert boost_typed == boost_global

    def test_cross_type_isolation(self):
        """Item consistently useful for code but bad for memory should score differently."""
        now = datetime.now(timezone.utc)
        rows = [
            ("useful", now, "code"),
            ("useful", now, "code"),
            ("useful", now, "code"),
            ("bad", now, "memory"),
            ("bad", now, "memory"),
            ("bad", now, "memory"),
        ]

        conn_code = _mock_conn(rows)
        boost_code = feedback_boost("item-1", query_type="code", conn=conn_code)

        conn_mem = _mock_conn(rows)
        boost_memory = feedback_boost("item-1", query_type="memory", conn=conn_mem)

        # Code should be positive (typed = all useful)
        assert boost_code > 0
        # Memory should be negative (typed = all bad)
        assert boost_memory < 0
        # The two should be clearly different
        assert boost_code > boost_memory

    def test_blend_weight_proportions(self):
        """Verify the blend uses TYPED_BLEND_WEIGHT correctly."""
        now = datetime.now(timezone.utc)
        # All feedback is the same type, so typed == global
        rows = [
            ("useful", now, "code"),
            ("useful", now, "code"),
            ("useful", now, "code"),
        ]
        conn_typed = _mock_conn(rows)
        boost_typed = feedback_boost("item-1", query_type="code", conn=conn_typed)

        conn_global = _mock_conn(rows)
        boost_global = feedback_boost("item-1", query_type=None, conn=conn_global)

        # When all feedback matches query_type, blended == global
        # (because typed_damped == global_damped)
        assert boost_typed == boost_global


# ---------------------------------------------------------------------------
# Per-query-type batch scoring
# ---------------------------------------------------------------------------

class TestQueryTypeBatchScoring:
    def test_batch_with_query_type(self):
        """Batch scoring respects query_type parameter."""
        now = datetime.now(timezone.utc)
        rows = [
            ("item-a", "useful", now, "code"),
            ("item-a", "useful", now, "code"),
            ("item-a", "bad", now, "memory"),
            ("item-a", "bad", now, "memory"),
            ("item-b", "useful", now, "memory"),
            ("item-b", "useful", now, "memory"),
            ("item-b", "useful", now, "memory"),
        ]
        conn = _mock_conn(rows)
        result = feedback_boost_batch(
            ["item-a", "item-b"],
            query_type="code",
            conn=conn,
        )

        # item-a: typed=code (2 useful), global (2 useful + 2 bad) → blended positive
        # item-b: typed=code (0 rows < cold start) → falls back to global (3 useful) → positive
        assert result["item-a"] > 0  # Typed code signal is positive
        assert result["item-b"] > 0  # Global is positive

    def test_batch_general_uses_global_only(self):
        """query_type='general' should use global scoring only."""
        now = datetime.now(timezone.utc)
        rows = [
            ("item-a", "useful", now, "code"),
            ("item-a", "useful", now, "code"),
            ("item-a", "useful", now, "code"),
        ]
        conn1 = _mock_conn(rows)
        result_general = feedback_boost_batch(["item-a"], query_type="general", conn=conn1)

        conn2 = _mock_conn(rows)
        result_none = feedback_boost_batch(["item-a"], query_type=None, conn=conn2)

        assert result_general["item-a"] == result_none["item-a"]

    def test_batch_empty_items(self):
        result = feedback_boost_batch([], query_type="code")
        assert result == {}
