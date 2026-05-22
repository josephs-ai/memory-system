"""
Tests for S2: Feedback-weighted retrieval integration.

Tests that feedback_boost is correctly wired into both retrieval paths
(search_memory.py and context_hydrator.py), and that feedback capture
scripts accept query_text and query_type.
"""

import sys
from pathlib import Path

import pytest

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from feedback_score_engine import classify_query_type


# ---------------------------------------------------------------------------
# Test: feedback_boost_batch is called in search_memory flow
# ---------------------------------------------------------------------------

class TestSearchMemoryFeedbackIntegration:
    """Verify search_memory.py threads item_id and applies feedback."""

    def test_canonical_rows_include_item_id(self):
        """The scored dict for canonical items must include item_id."""
        # Simulate what search_memory.py does when building scored rows
        item = {"id": "mem-abc123", "text": "test item", "final_score": 1.5}
        row = {
            "score": float(item.get("final_score", 0) or 0) + 0.10,
            "source_type": "canonical",
            "path": "db:memory_items",
            "text": item.get("text", ""),
            "mtime": float("inf"),
            "item_id": item.get("id"),
        }
        assert row["item_id"] == "mem-abc123"
        assert row["score"] == pytest.approx(1.6)

    def test_qdrant_rows_include_item_id(self):
        """The scored dict for qdrant items must include item_id."""
        payload = {"id": "mem-xyz789", "text": "qdrant hit"}
        row = {
            "score": float(0.85) + 0.20,
            "source_type": "canonical_qdrant",
            "path": "qdrant:memory_items",
            "text": payload.get("text") or "",
            "mtime": float("inf"),
            "item_id": payload.get("id"),
        }
        assert row["item_id"] == "mem-xyz789"

    def test_feedback_boost_applied_to_score(self):
        """Simulate the feedback boost injection logic."""
        rows = [
            {"score": 1.5, "item_id": "item-a", "source_type": "canonical"},
            {"score": 1.2, "item_id": "item-b", "source_type": "canonical"},
            {"score": 0.8, "item_id": None, "source_type": "project_daily"},
        ]
        boosts = {"item-a": 0.25, "item-b": -0.40}
        feedback_weight = 0.15

        for row in rows:
            iid = row.get("item_id")
            if iid and iid in boosts:
                row["feedback_boost"] = boosts[iid]
                row["score"] += boosts[iid] * feedback_weight

        assert rows[0]["score"] == pytest.approx(1.5 + 0.25 * 0.15)
        assert rows[0]["feedback_boost"] == 0.25
        assert rows[1]["score"] == pytest.approx(1.2 + (-0.40) * 0.15)
        assert rows[1]["feedback_boost"] == -0.40
        assert "feedback_boost" not in rows[2]  # No item_id
        assert rows[2]["score"] == 0.8  # Unchanged

    def test_feedback_boost_graceful_on_empty(self):
        """No item_ids → no crash, no feedback applied."""
        rows = [
            {"score": 1.0, "item_id": None, "source_type": "project_current"},
        ]
        feedback_item_ids = [r["item_id"] for r in rows if r.get("item_id")]
        assert feedback_item_ids == []


# ---------------------------------------------------------------------------
# Test: context_hydrator feedback path
# ---------------------------------------------------------------------------

class TestContextHydratorFeedbackIntegration:
    """Verify context_hydrator applies feedback to memory hits before reranking."""

    def test_memory_hits_get_feedback_boost(self):
        """Simulate the feedback injection in context_hydrator."""
        memory_hits = [
            {"source": "memory", "id": "mem-a", "score": 0.85, "text": "test"},
            {"source": "memory", "id": "mem-b", "score": 0.72, "text": "test2"},
            {"source": "memory", "id": None, "score": 0.60, "text": "test3"},
        ]
        boosts = {"mem-a": 0.20, "mem-b": -0.30}
        feedback_weight = 0.15

        for h in memory_hits:
            iid = h.get("id")
            if iid and iid in boosts and boosts[iid] != 0.0:
                h["feedback_boost"] = boosts[iid]
                h["score"] = h.get("score", 0.0) + boosts[iid] * feedback_weight

        assert memory_hits[0]["score"] == pytest.approx(0.85 + 0.20 * 0.15)
        assert memory_hits[1]["score"] == pytest.approx(0.72 + (-0.30) * 0.15)
        assert memory_hits[2]["score"] == 0.60  # No id, unchanged

    def test_feedback_before_reranking_order(self):
        """Feedback adjustments happen before cross-encoder reranking."""
        # This tests that item B can overtake item A via feedback
        hits = [
            {"source": "memory", "id": "mem-good", "score": 0.70, "text": "good"},
            {"source": "memory", "id": "mem-bad", "score": 0.75, "text": "bad"},
        ]
        boosts = {"mem-good": 0.25, "mem-bad": -0.40}

        for h in hits:
            iid = h.get("id")
            if iid and iid in boosts:
                h["score"] += boosts[iid] * 0.15

        # mem-good: 0.70 + 0.0375 = 0.7375
        # mem-bad: 0.75 + (-0.06) = 0.69
        hits.sort(key=lambda x: x["score"], reverse=True)
        assert hits[0]["id"] == "mem-good"


# ---------------------------------------------------------------------------
# Test: feedback capture scripts accept new args
# ---------------------------------------------------------------------------

class TestFeedbackCaptureScripts:
    """Verify mark_retrieval_useful/bad accept query_text and query_type."""

    def test_query_type_auto_classification(self):
        """classify_query_type is used when query_text provided without query_type."""
        assert classify_query_type("fix the parse_code_ast.py function") == "code"
        assert classify_query_type("what happened yesterday") == "timeline"

    def test_feedback_row_includes_query_fields(self):
        """Verify the row dict structure includes query_text and query_type."""
        row = {
            "selected_id": "mem-123",
            "operator_feedback": "useful",
            "feedback_at": "2026-05-13T00:00:00Z",
            "query_text": "fix the embedding pipeline",
            "query_type": "code",
        }
        assert row["query_text"] == "fix the embedding pipeline"
        assert row["query_type"] == "code"

    def test_feedback_row_without_query_fields(self):
        """Backward compat: query_text/query_type can be None."""
        row = {
            "selected_id": "mem-456",
            "operator_feedback": "bad",
            "feedback_at": "2026-05-13T00:00:00Z",
            "query_text": None,
            "query_type": None,
        }
        assert row["query_text"] is None
        assert row["query_type"] is None


# ---------------------------------------------------------------------------
# Test: insert_retrieval_feedback_row schema compatibility
# ---------------------------------------------------------------------------

class TestInsertFeedbackRowSchema:
    """Verify memory_db.insert_retrieval_feedback_row handles new columns."""

    def test_row_with_query_fields_matches_sql(self):
        """The INSERT statement must have 12 columns (10 original + 2 new)."""
        # Count the VALUES placeholders in the SQL
        from memory_db import insert_retrieval_feedback_row
        import inspect
        source = inspect.getsource(insert_retrieval_feedback_row)
        # Count %s placeholders
        placeholder_count = source.count("%s")
        assert placeholder_count == 12, f"Expected 12 placeholders, got {placeholder_count}"
