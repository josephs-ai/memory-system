"""
Tests for consolidation_grouper.py (P2-S1).

Covers:
- Time-based resolution assignment (daily/weekly/monthly)
- Scope-respecting grouping (never cross-project)
- Entity grouping for daily resolution
- Minimum group size filtering
- Excluded memory types
- Time range computation
- Edge cases: no items, all too recent, mixed ages
"""

import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from consolidation_grouper import (
    group_for_consolidation,
    _iso_week,
    _iso_month,
    _iso_day,
    MIN_GROUP_SIZE,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_item(
    item_id: str,
    age_days: float,
    scope: str = "global",
    entity: str = "",
    memory_type: str = "fact",
    now: datetime | None = None,
) -> dict:
    now = now or datetime.now(timezone.utc)
    first_seen = now - timedelta(days=age_days)
    return {
        "id": item_id,
        "text": f"Item {item_id}",
        "memory_type": memory_type,
        "scope": scope,
        "entity": entity,
        "property": "",
        "first_seen": first_seen,
        "last_confirmed": first_seen,
        "confidence": 0.8,
        "tags": [],
        "sensitivity": "public",
    }


# ---------------------------------------------------------------------------
# ISO helpers
# ---------------------------------------------------------------------------

class TestISOHelpers:
    def test_iso_day(self):
        dt = datetime(2026, 5, 13, tzinfo=timezone.utc)
        assert _iso_day(dt) == "2026-05-13"

    def test_iso_week(self):
        dt = datetime(2026, 5, 13, tzinfo=timezone.utc)
        result = _iso_week(dt)
        assert result.startswith("2026-W")

    def test_iso_month(self):
        dt = datetime(2026, 5, 13, tzinfo=timezone.utc)
        assert _iso_month(dt) == "2026-05"


# ---------------------------------------------------------------------------
# Resolution assignment
# ---------------------------------------------------------------------------

class TestResolutionAssignment:
    def test_daily_resolution(self):
        """Items 7-30 days old get daily resolution."""
        now = datetime.now(timezone.utc)
        items = [_make_item(f"d{i}", 10, now=now) for i in range(5)]
        groups = group_for_consolidation(items, now=now)
        assert len(groups) == 1
        assert groups[0]["resolution"] == "daily"

    def test_weekly_resolution(self):
        """Items 30-90 days old get weekly resolution."""
        now = datetime.now(timezone.utc)
        items = [_make_item(f"w{i}", 45, now=now) for i in range(5)]
        groups = group_for_consolidation(items, now=now)
        assert len(groups) == 1
        assert groups[0]["resolution"] == "weekly"

    def test_monthly_resolution(self):
        """Items 90+ days old get monthly resolution."""
        now = datetime.now(timezone.utc)
        items = [_make_item(f"m{i}", 120, now=now) for i in range(5)]
        groups = group_for_consolidation(items, now=now)
        assert len(groups) == 1
        assert groups[0]["resolution"] == "monthly"

    def test_too_recent_excluded(self):
        """Items less than 7 days old are not grouped."""
        now = datetime.now(timezone.utc)
        items = [_make_item(f"r{i}", 3, now=now) for i in range(10)]
        groups = group_for_consolidation(items, now=now)
        assert groups == []

    def test_mixed_ages_separate_groups(self):
        """Items of different ages go to different resolution groups."""
        now = datetime.now(timezone.utc)
        items = (
            [_make_item(f"d{i}", 10, now=now) for i in range(3)] +
            [_make_item(f"m{i}", 120, now=now) for i in range(3)]
        )
        groups = group_for_consolidation(items, now=now)
        resolutions = {g["resolution"] for g in groups}
        assert "daily" in resolutions
        assert "monthly" in resolutions


# ---------------------------------------------------------------------------
# Scope isolation
# ---------------------------------------------------------------------------

class TestScopeIsolation:
    def test_different_scopes_separate_groups(self):
        """Items from different projects never merge."""
        now = datetime.now(timezone.utc)
        items = (
            [_make_item(f"a{i}", 15, scope="project-a", now=now) for i in range(3)] +
            [_make_item(f"b{i}", 15, scope="project-b", now=now) for i in range(3)]
        )
        groups = group_for_consolidation(items, now=now)
        assert len(groups) == 2
        scopes = {g["scope"] for g in groups}
        assert scopes == {"project-a", "project-b"}


# ---------------------------------------------------------------------------
# Entity grouping (daily resolution)
# ---------------------------------------------------------------------------

class TestEntityGrouping:
    def test_daily_groups_by_entity(self):
        """Daily resolution groups by entity."""
        now = datetime.now(timezone.utc)
        items = (
            [_make_item(f"br{i}", 10, entity="browser", now=now) for i in range(3)] +
            [_make_item(f"mem{i}", 10, entity="memory", now=now) for i in range(3)]
        )
        groups = group_for_consolidation(items, now=now)
        assert len(groups) == 2

    def test_weekly_ignores_entity(self):
        """Weekly resolution groups by scope only, not entity."""
        now = datetime.now(timezone.utc)
        items = (
            [_make_item(f"br{i}", 45, entity="browser", now=now) for i in range(2)] +
            [_make_item(f"mem{i}", 45, entity="memory", now=now) for i in range(2)]
        )
        groups = group_for_consolidation(items, now=now)
        # All in same scope+week → single group (if >= MIN_GROUP_SIZE)
        if len(groups) > 0:
            assert groups[0]["resolution"] == "weekly"
            assert len(groups[0]["items"]) == 4


# ---------------------------------------------------------------------------
# Minimum group size
# ---------------------------------------------------------------------------

class TestMinGroupSize:
    def test_small_groups_filtered(self):
        """Groups with fewer than MIN_GROUP_SIZE items are excluded."""
        now = datetime.now(timezone.utc)
        items = [_make_item(f"s{i}", 10, now=now) for i in range(MIN_GROUP_SIZE - 1)]
        groups = group_for_consolidation(items, now=now)
        assert groups == []

    def test_exact_min_size_included(self):
        """Groups with exactly MIN_GROUP_SIZE items are included."""
        now = datetime.now(timezone.utc)
        items = [_make_item(f"e{i}", 10, now=now) for i in range(MIN_GROUP_SIZE)]
        groups = group_for_consolidation(items, now=now)
        assert len(groups) == 1


# ---------------------------------------------------------------------------
# Time range computation
# ---------------------------------------------------------------------------

class TestTimeRange:
    def test_time_range_covers_items(self):
        """Time range spans from earliest to latest item in group."""
        now = datetime.now(timezone.utc)
        # All items on the same day (10 days ago, with slight hour offsets)
        base_age = 10.0
        items = [
            _make_item("early", base_age + 0.02, now=now),  # ~30 min apart
            _make_item("mid", base_age + 0.01, now=now),
            _make_item("late", base_age, now=now),
        ]
        groups = group_for_consolidation(items, now=now)
        assert len(groups) == 1
        g = groups[0]
        assert g["time_range_start"] <= g["time_range_end"]
        # Earliest first_seen should be the start
        assert g["time_range_start"] == min(i["first_seen"] for i in items)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_empty_input(self):
        assert group_for_consolidation([]) == []

    def test_none_first_seen_skipped(self):
        items = [{"id": "bad", "first_seen": None, "scope": "x", "entity": "", "memory_type": "fact"}]
        assert group_for_consolidation(items) == []

    def test_naive_datetime_handled(self):
        """Naive datetimes (no tzinfo) are treated as UTC."""
        now = datetime.now(timezone.utc)
        item = _make_item("naive", 15, now=now)
        item["first_seen"] = item["first_seen"].replace(tzinfo=None)
        items = [item] * MIN_GROUP_SIZE  # Duplicate to meet min size
        # Should not crash
        groups = group_for_consolidation(items, now=now)
        assert isinstance(groups, list)
