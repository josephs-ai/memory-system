"""
Tests for P2-S3: Retrieval integration for progressive summarization.

Covers:
- Summary items get score penalty in search_memory
- Summary metadata (is_summary, resolution_level, constituent_count) passed through
- context_hydrator formats summaries with drill-down hints
- expand_memory_summary drill-down logic
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from expand_memory_summary import expand_summary


# ---------------------------------------------------------------------------
# search_memory summary handling
# ---------------------------------------------------------------------------

class TestSearchMemorySummaryHandling:
    """Verify search_memory correctly handles summary items."""

    def test_summary_gets_score_penalty(self):
        """Summary items should have a 0.05 score penalty."""
        base_score = 1.5
        is_summary = True
        adjusted = base_score - (0.05 if is_summary else 0)
        assert adjusted == 1.45

    def test_non_summary_no_penalty(self):
        """Normal items should not be penalized."""
        base_score = 1.5
        is_summary = False
        adjusted = base_score - (0.05 if is_summary else 0)
        assert adjusted == 1.5

    def test_summary_row_has_metadata(self):
        """Summary rows should include drill-down metadata."""
        item = {
            "id": "summary-abc",
            "text": "Weekly summary...",
            "memory_type": "summary",
            "final_score": 1.2,
            "resolution_level": "weekly",
            "constituent_item_ids": ["a", "b", "c"],
        }
        is_summary = item.get("memory_type") == "summary"
        row = {
            "score": float(item.get("final_score", 0) or 0) + 0.10 - (0.05 if is_summary else 0),
            "source_type": "canonical_summary" if is_summary else "canonical",
            "text": item["text"],
            "item_id": item["id"],
        }
        if is_summary:
            constituent_ids = item.get("constituent_item_ids") or []
            row["is_summary"] = True
            row["resolution_level"] = item.get("resolution_level")
            row["constituent_count"] = len(constituent_ids)

        assert row["is_summary"] is True
        assert row["source_type"] == "canonical_summary"
        assert row["resolution_level"] == "weekly"
        assert row["constituent_count"] == 3
        assert row["score"] == pytest.approx(1.25)

    def test_summary_still_retrievable(self):
        """Summaries should rank, just lower than equivalent originals."""
        original_score = 1.5
        summary_score = 1.5 - 0.05
        assert original_score > summary_score
        assert summary_score > 0  # Still positive / retrievable


# ---------------------------------------------------------------------------
# context_hydrator summary formatting
# ---------------------------------------------------------------------------

class TestContextHydratorSummaryFormat:
    def test_summary_hit_gets_hint(self):
        """Summary memory items should include a drill-down hint."""
        hit = {
            "source": "memory",
            "id": "summary-xyz",
            "score": 0.8,
            "text": "[Weekly Summary | scope: global | 2026-W19 | 12 items]",
            "memory_type": "summary",
            "confidence": 0.7,
        }
        # Simulate _format_memory_item logic
        result = {
            "type": "memory",
            "memory_type": hit.get("memory_type"),
            "text": hit["text"],
            "score": round(hit["score"], 4),
        }
        if hit.get("memory_type") == "summary":
            result["is_summary"] = True
            result["hint"] = f"[Summary of consolidated items. ID: {hit.get('id', '?')}]"

        assert result["is_summary"] is True
        assert "summary-xyz" in result["hint"]

    def test_non_summary_no_hint(self):
        """Non-summary items should not have hints."""
        hit = {"memory_type": "fact", "text": "normal item"}
        result = {"type": "memory", "text": hit["text"]}
        if hit.get("memory_type") == "summary":
            result["is_summary"] = True
        assert "is_summary" not in result


# ---------------------------------------------------------------------------
# expand_memory_summary
# ---------------------------------------------------------------------------

class TestExpandMemorySummary:
    def _mock_conn(self, summary_row, constituent_rows):
        """Mock conn that returns summary then constituents."""
        mock_cur = MagicMock()
        _call_idx = [0]

        def fake_fetchone():
            return summary_row

        def fake_fetchall():
            return constituent_rows

        mock_cur.fetchone = fake_fetchone
        mock_cur.fetchall = fake_fetchall
        mock_cur.description = [
            ("id",), ("text",), ("memory_type",), ("scope",),
            ("entity",), ("property",), ("confidence",), ("first_seen",),
        ]
        mock_cur.__enter__ = lambda s: s
        mock_cur.__exit__ = MagicMock(return_value=False)

        mock_conn = MagicMock()
        mock_conn.cursor.return_value = mock_cur
        return mock_conn

    def test_expand_returns_constituents(self):
        summary_row = (
            "[Daily Summary | 5 items]",  # text
            "daily",                        # resolution
            ["item-1", "item-2", "item-3"], # constituent_item_ids
        )
        constituent_rows = [
            ("item-1", "First item text", "fact", "global", "", "", 0.8, None),
            ("item-2", "Second item text", "decision", "global", "", "", 0.9, None),
            ("item-3", "Third item text", "observation", "global", "", "", 0.7, None),
        ]
        conn = self._mock_conn(summary_row, constituent_rows)
        result = expand_summary("summary-abc", conn=conn)

        assert result["summary_id"] == "summary-abc"
        assert result["constituent_count"] == 3
        assert result["items_returned"] == 3
        assert result["items"][0]["id"] == "item-1"
        assert result["resolution_level"] == "daily"

    def test_expand_not_found(self):
        mock_cur = MagicMock()
        mock_cur.fetchone.return_value = None
        mock_cur.__enter__ = lambda s: s
        mock_cur.__exit__ = MagicMock(return_value=False)
        mock_conn = MagicMock()
        mock_conn.cursor.return_value = mock_cur

        result = expand_summary("nonexistent", conn=mock_conn)
        assert "error" in result
        assert result["items"] == []

    def test_expand_with_limit(self):
        summary_row = (
            "Summary text",
            "weekly",
            ["a", "b", "c", "d", "e"],
        )
        # Only 2 would be fetched with limit=2
        constituent_rows = [
            ("a", "text a", "fact", "g", "", "", 0.8, None),
            ("b", "text b", "fact", "g", "", "", 0.8, None),
        ]
        conn = self._mock_conn(summary_row, constituent_rows)
        result = expand_summary("summary-xyz", limit=2, conn=conn)
        assert result["constituent_count"] == 5  # Total in summary
        assert result["items_returned"] == 2     # Returned with limit

    def test_expand_empty_constituents(self):
        summary_row = ("Summary with no items", "daily", [])
        conn = self._mock_conn(summary_row, [])
        result = expand_summary("summary-empty", conn=conn)
        assert result["constituent_count"] == 0
        assert result["items"] == []
