"""
Tests for temporal_scoring.py (P3: Temporal Reasoning).

Covers:
- Temporal query detection
- Time reference extraction (ISO dates, relative, named months)
- Recency scoring (decay curve, staleness penalty)
- Temporal match scoring
- Combined temporal boost logic
- Batch scoring
- Edge cases
"""

import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from temporal_scoring import (
    is_temporal_query,
    extract_time_reference,
    recency_score,
    temporal_match_score,
    compute_temporal_boost,
    compute_temporal_boost_batch,
    RECENCY_WEIGHT,
    STALENESS_PENALTY,
    TEMPORAL_QUERY_BOOST,
)


# ---------------------------------------------------------------------------
# Temporal query detection
# ---------------------------------------------------------------------------

class TestIsTemporalQuery:
    def test_obvious_temporal(self):
        assert is_temporal_query("when did we deploy the fix") is True
        assert is_temporal_query("what happened yesterday with the build") is True
        assert is_temporal_query("what changed last week") is True

    def test_date_reference(self):
        assert is_temporal_query("events from 2026-04-15") is True
        assert is_temporal_query("what happened in March") is True

    def test_non_temporal(self):
        assert is_temporal_query("fix the parsing function") is False
        assert is_temporal_query("what is the config for ports") is False
        assert is_temporal_query("hello world") is False

    def test_empty(self):
        assert is_temporal_query("") is False
        assert is_temporal_query(None) is False

    def test_relative_time(self):
        assert is_temporal_query("what did we do 3 days ago") is True
        assert is_temporal_query("changes since last month") is True


# ---------------------------------------------------------------------------
# Time reference extraction
# ---------------------------------------------------------------------------

class TestExtractTimeReference:
    def test_iso_date(self):
        ref = extract_time_reference("what happened on 2026-04-15")
        assert ref is not None
        assert ref["type"] == "absolute"
        assert ref["start"].day == 15
        assert ref["start"].month == 4

    def test_days_ago(self):
        ref = extract_time_reference("what changed 5 days ago")
        assert ref is not None
        assert ref["type"] == "relative"
        now = datetime.now(timezone.utc)
        expected = now - timedelta(days=5)
        assert abs((ref["start"] - expected).total_seconds()) < 86400 * 2

    def test_weeks_ago(self):
        ref = extract_time_reference("updates from 2 weeks ago")
        assert ref is not None
        assert ref["type"] == "relative"

    def test_last_n_days(self):
        ref = extract_time_reference("what happened in the last 7 days")
        assert ref is not None
        assert ref["type"] == "relative"
        # End should be approximately now
        assert ref["end"] is not None

    def test_named_month(self):
        ref = extract_time_reference("what decisions were made in March 2026")
        assert ref is not None
        assert ref["type"] == "named_month"
        assert ref["start"].month == 3
        assert ref["start"].year == 2026

    def test_named_month_no_year(self):
        ref = extract_time_reference("what happened in April")
        assert ref is not None
        assert ref["start"].month == 4

    def test_no_time_reference(self):
        assert extract_time_reference("fix the code") is None
        assert extract_time_reference("") is None
        assert extract_time_reference(None) is None


# ---------------------------------------------------------------------------
# Recency scoring
# ---------------------------------------------------------------------------

class TestRecencyScore:
    def test_very_recent_gets_max_boost(self):
        now = datetime.now(timezone.utc)
        score = recency_score(now, now=now)
        assert score == RECENCY_WEIGHT

    def test_one_day_old_max_boost(self):
        now = datetime.now(timezone.utc)
        ts = now - timedelta(hours=12)
        score = recency_score(ts, now=now)
        assert score == RECENCY_WEIGHT

    def test_moderate_age_positive(self):
        now = datetime.now(timezone.utc)
        ts = now - timedelta(days=7)
        score = recency_score(ts, now=now)
        assert score > 0
        assert score < RECENCY_WEIGHT

    def test_old_item_gets_penalty(self):
        now = datetime.now(timezone.utc)
        ts = now - timedelta(days=180)
        score = recency_score(ts, now=now)
        assert score < 0

    def test_very_old_clamps_to_min(self):
        now = datetime.now(timezone.utc)
        ts = now - timedelta(days=365)
        score = recency_score(ts, now=now)
        assert score >= STALENESS_PENALTY

    def test_none_timestamp_returns_zero(self):
        assert recency_score(None) == 0.0

    def test_naive_datetime_handled(self):
        now = datetime.now(timezone.utc)
        ts = (now - timedelta(days=5)).replace(tzinfo=None)
        score = recency_score(ts, now=now)
        assert isinstance(score, float)

    def test_monotonic_decay(self):
        """Scores should decrease as items get older."""
        now = datetime.now(timezone.utc)
        scores = [recency_score(now - timedelta(days=d), now=now) for d in [1, 7, 30, 60, 120]]
        for i in range(len(scores) - 1):
            assert scores[i] >= scores[i + 1]


# ---------------------------------------------------------------------------
# Temporal match scoring
# ---------------------------------------------------------------------------

class TestTemporalMatchScore:
    def test_item_in_range_gets_boost(self):
        ts = datetime(2026, 4, 15, 12, 0, tzinfo=timezone.utc)
        ref = {
            "start": datetime(2026, 4, 15, 0, 0, tzinfo=timezone.utc),
            "end": datetime(2026, 4, 16, 0, 0, tzinfo=timezone.utc),
        }
        assert temporal_match_score(ts, ref) == TEMPORAL_QUERY_BOOST

    def test_item_outside_range_no_boost(self):
        ts = datetime(2026, 4, 10, tzinfo=timezone.utc)
        ref = {
            "start": datetime(2026, 4, 15, tzinfo=timezone.utc),
            "end": datetime(2026, 4, 16, tzinfo=timezone.utc),
        }
        assert temporal_match_score(ts, ref) == 0.0

    def test_none_timestamp(self):
        assert temporal_match_score(None, {"start": datetime.now(timezone.utc), "end": datetime.now(timezone.utc)}) == 0.0

    def test_none_ref(self):
        assert temporal_match_score(datetime.now(timezone.utc), None) == 0.0


# ---------------------------------------------------------------------------
# Combined temporal boost
# ---------------------------------------------------------------------------

class TestComputeTemporalBoost:
    def test_temporal_query_with_match(self):
        """Temporal query + item in time window → temporal match boost."""
        now = datetime.now(timezone.utc)
        ts = now - timedelta(days=5)
        boost = compute_temporal_boost(ts, "what happened 5 days ago", now=now)
        # Should get temporal_match_score (item roughly in the 5-days-ago window)
        assert boost >= 0.0

    def test_non_temporal_query_recent_item(self):
        """Non-temporal query + recent item → recency boost."""
        now = datetime.now(timezone.utc)
        ts = now - timedelta(hours=6)
        boost = compute_temporal_boost(ts, "fix the parsing function", now=now)
        assert boost == RECENCY_WEIGHT

    def test_non_temporal_query_old_item(self):
        """Non-temporal query + old item → staleness penalty."""
        now = datetime.now(timezone.utc)
        ts = now - timedelta(days=200)
        boost = compute_temporal_boost(ts, "fix the parsing function", now=now)
        assert boost < 0

    def test_temporal_query_no_extractable_ref(self):
        """Temporal query where we can't extract a specific time → neutral."""
        boost = compute_temporal_boost(
            datetime.now(timezone.utc),
            "when did this happen and what changed in the history"
        )
        assert boost == 0.0


# ---------------------------------------------------------------------------
# Batch scoring
# ---------------------------------------------------------------------------

class TestBatchScoring:
    def test_batch_basic(self):
        now = datetime.now(timezone.utc)
        items = [
            {"id": "recent", "first_seen": now - timedelta(hours=1)},
            {"id": "old", "first_seen": now - timedelta(days=100)},
            {"id": "mid", "first_seen": now - timedelta(days=10)},
        ]
        boosts = compute_temporal_boost_batch(items, "fix the code", now=now)
        assert boosts["recent"] > boosts["mid"] > boosts["old"]

    def test_batch_temporal_query(self):
        now = datetime.now(timezone.utc)
        items = [
            {"id": "match", "first_seen": now - timedelta(days=5)},
            {"id": "no-match", "first_seen": now - timedelta(days=100)},
        ]
        boosts = compute_temporal_boost_batch(items, "what happened 5 days ago", now=now)
        # "match" should have some boost (near the reference), "no-match" should be 0
        assert isinstance(boosts["match"], float)
        assert isinstance(boosts["no-match"], float)

    def test_batch_empty(self):
        assert compute_temporal_boost_batch([], "query") == {}

    def test_batch_missing_id_skipped(self):
        items = [{"first_seen": datetime.now(timezone.utc)}]  # No id
        assert compute_temporal_boost_batch(items, "query") == {}
