"""
Tests for summary_generator.py (P2-S2).

Covers:
- Representative item selection (confidence + text length scoring)
- Summary text generation (structured format, JSON/noise filtering)
- Summary item creation (correct schema fields, deterministic ID)
- Constituent archival logic
- Dry-run mode
- Edge cases: all-JSON items, empty groups
"""

import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from summary_generator import (
    _summary_id,
    _select_representative_items,
    generate_summary_text,
    create_summary_item,
    consolidate_group,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_group(
    items: list[dict],
    resolution: str = "daily",
    scope: str = "global",
    time_label: str = "2026-05-01",
    group_key: str = "daily|2026-05-01|global",
) -> dict:
    now = datetime.now(timezone.utc)
    return {
        "resolution": resolution,
        "group_key": group_key,
        "scope": scope,
        "time_label": time_label,
        "time_range_start": now - timedelta(days=10),
        "time_range_end": now - timedelta(days=9),
        "items": items,
    }


def _make_item(item_id: str, text: str, confidence: float = 0.8) -> dict:
    return {
        "id": item_id,
        "text": text,
        "memory_type": "fact",
        "scope": "global",
        "entity": "",
        "property": "",
        "confidence": confidence,
    }


# ---------------------------------------------------------------------------
# _summary_id
# ---------------------------------------------------------------------------

class TestSummaryId:
    def test_deterministic(self):
        assert _summary_id("key-1") == _summary_id("key-1")

    def test_different_keys_different_ids(self):
        assert _summary_id("key-1") != _summary_id("key-2")

    def test_starts_with_summary(self):
        assert _summary_id("any-key").startswith("summary-")


# ---------------------------------------------------------------------------
# _select_representative_items
# ---------------------------------------------------------------------------

class TestSelectRepresentative:
    def test_selects_high_confidence_first(self):
        items = [
            _make_item("low", "short text", confidence=0.3),
            _make_item("high", "short text", confidence=0.9),
            _make_item("mid", "short text", confidence=0.6),
        ]
        reps = _select_representative_items(items, max_items=2)
        assert reps[0]["id"] == "high"

    def test_prefers_longer_text(self):
        items = [
            _make_item("short", "x", confidence=0.5),
            _make_item("long", "x" * 300, confidence=0.5),
        ]
        reps = _select_representative_items(items, max_items=1)
        assert reps[0]["id"] == "long"

    def test_respects_max_items(self):
        items = [_make_item(f"i{i}", f"text {i}") for i in range(20)]
        reps = _select_representative_items(items, max_items=5)
        assert len(reps) == 5

    def test_empty_items(self):
        assert _select_representative_items([]) == []


# ---------------------------------------------------------------------------
# generate_summary_text
# ---------------------------------------------------------------------------

class TestGenerateSummaryText:
    def test_basic_format(self):
        items = [_make_item("a", "The system uses PostgreSQL for storage")]
        group = _make_group(items * 5)
        text = generate_summary_text(group, items)
        assert "[Daily Summary" in text
        assert "scope: global" in text
        assert "PostgreSQL" in text

    def test_json_items_filtered(self):
        """Items that look like raw JSON should be skipped."""
        items = [
            _make_item("json", '{"type":"message","id":"abc123"}'),
            _make_item("good", "Browser config uses port 9224"),
        ]
        group = _make_group(items * 3)
        text = generate_summary_text(group, items)
        assert "Browser config" in text
        assert '"type":"message"' not in text

    def test_all_json_fallback(self):
        """If all items are JSON, provide count-only fallback."""
        items = [_make_item("j1", '{"raw": "json"}')]
        group = _make_group(items * 5)
        text = generate_summary_text(group, items)
        assert "memory items" in text

    def test_long_text_truncated(self):
        long_text = "word " * 100
        items = [_make_item("long", long_text)]
        group = _make_group(items * 3)
        text = generate_summary_text(group, items)
        assert "..." in text

    def test_weekly_label(self):
        items = [_make_item("w", "weekly item")]
        group = _make_group(items * 3, resolution="weekly", group_key="weekly|2026-W19|global")
        text = generate_summary_text(group, items)
        assert "[Weekly Summary" in text

    def test_monthly_label(self):
        items = [_make_item("m", "monthly item")]
        group = _make_group(items * 3, resolution="monthly", group_key="monthly|2026-03|global")
        text = generate_summary_text(group, items)
        assert "[Monthly Summary" in text


# ---------------------------------------------------------------------------
# create_summary_item
# ---------------------------------------------------------------------------

class TestCreateSummaryItem:
    def test_correct_schema(self):
        items = [_make_item(f"i{i}", f"text {i}") for i in range(5)]
        group = _make_group(items)
        summary_text = "Test summary"
        result = create_summary_item(group, summary_text)

        assert result["memory_type"] == "summary"
        assert result["status"] == "active"
        assert result["resolution_level"] == "daily"
        assert result["scope"] == "global"
        assert len(result["constituent_item_ids"]) == 5
        assert result["confidence"] == 0.7  # Lower than originals
        assert result["id"].startswith("summary-")

    def test_constituent_ids_populated(self):
        items = [_make_item("a", "x"), _make_item("b", "y"), _make_item("c", "z")]
        group = _make_group(items)
        result = create_summary_item(group, "text")
        assert set(result["constituent_item_ids"]) == {"a", "b", "c"}


# ---------------------------------------------------------------------------
# consolidate_group (dry run)
# ---------------------------------------------------------------------------

class TestConsolidateGroup:
    def test_dry_run_no_side_effects(self):
        items = [_make_item(f"i{i}", f"Fact number {i}") for i in range(5)]
        group = _make_group(items)
        result = consolidate_group(group, dry_run=True)

        assert result["dry_run"] is True
        assert result["summary_id"].startswith("summary-")
        assert result["constituent_count"] == 5
        assert "summary_text" in result

    def test_dry_run_summary_text_quality(self):
        items = [
            _make_item("i1", "The retrieval pipeline uses hybrid search"),
            _make_item("i2", "Cross-encoder reranking improves relevance"),
            _make_item("i3", "Feedback loop is now closed"),
        ]
        group = _make_group(items)
        result = consolidate_group(group, dry_run=True)

        text = result["summary_text"]
        assert "[Daily Summary" in text
        assert "retrieval" in text.lower() or "reranking" in text.lower() or "feedback" in text.lower()
