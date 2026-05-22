"""
Tests for memory_knowledge_graph.py (P4: Memory Knowledge Graph).

Covers:
- Explicit supersedes link detection
- Entity+property co-occurrence links
- Type-based + temporal proximity links
- Deduplication (highest confidence wins)
- Cross-scope isolation
- Edge cases: empty items, missing fields
"""

import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from memory_knowledge_graph import (
    detect_supersedes_links,
    detect_entity_links,
    detect_type_based_links,
    detect_all_links,
    REL_SUPERSEDES,
    REL_CAUSED_BY,
    REL_RELATES_TO,
    REL_SUPPORTS,
    TEMPORAL_PROXIMITY_HOURS,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_item(
    item_id: str,
    memory_type: str = "fact",
    entity: str = "",
    prop: str = "",
    value: str = "",
    scope: str = "global",
    supersedes: str | None = None,
    age_hours: float = 0,
) -> dict:
    now = datetime.now(timezone.utc)
    return {
        "id": item_id,
        "text": f"Item {item_id}",
        "memory_type": memory_type,
        "scope": scope,
        "entity": entity,
        "property": prop,
        "value": value,
        "first_seen": now - timedelta(hours=age_hours),
        "last_confirmed": now - timedelta(hours=age_hours),
        "supersedes": supersedes,
        "confidence": 0.8,
    }


# ---------------------------------------------------------------------------
# Explicit supersedes
# ---------------------------------------------------------------------------

class TestDetectSupersedesLinks:
    def test_explicit_supersedes(self):
        items = [
            _make_item("new", supersedes="old"),
            _make_item("old"),
        ]
        links = detect_supersedes_links(items)
        assert len(links) == 1
        assert links[0]["source_id"] == "new"
        assert links[0]["target_id"] == "old"
        assert links[0]["relationship"] == REL_SUPERSEDES
        assert links[0]["confidence"] == 1.0

    def test_no_supersedes(self):
        items = [_make_item("a"), _make_item("b")]
        links = detect_supersedes_links(items)
        assert links == []

    def test_dangling_supersedes_ignored(self):
        """Supersedes pointing to non-existent item → ignored."""
        items = [_make_item("a", supersedes="nonexistent")]
        links = detect_supersedes_links(items)
        assert links == []


# ---------------------------------------------------------------------------
# Entity+property links
# ---------------------------------------------------------------------------

class TestDetectEntityLinks:
    def test_same_entity_prop_different_values(self):
        items = [
            _make_item("new", entity="browser", prop="port", value="9224", age_hours=0),
            _make_item("old", entity="browser", prop="port", value="9222", age_hours=48),
        ]
        links = detect_entity_links(items)
        assert len(links) == 1
        assert links[0]["relationship"] == REL_SUPERSEDES
        assert links[0]["confidence"] == 0.85

    def test_same_entity_prop_same_value(self):
        items = [
            _make_item("a", entity="memory", prop="backend", value="postgres", age_hours=0),
            _make_item("b", entity="memory", prop="backend", value="postgres", age_hours=24),
        ]
        links = detect_entity_links(items)
        assert len(links) == 1
        assert links[0]["relationship"] == REL_RELATES_TO

    def test_different_entities_no_link(self):
        items = [
            _make_item("a", entity="browser", prop="port", value="9224"),
            _make_item("b", entity="memory", prop="port", value="5432"),
        ]
        links = detect_entity_links(items)
        assert links == []

    def test_no_entity_no_link(self):
        items = [_make_item("a"), _make_item("b")]
        links = detect_entity_links(items)
        assert links == []


# ---------------------------------------------------------------------------
# Type-based + temporal links
# ---------------------------------------------------------------------------

class TestDetectTypeBasedLinks:
    def test_decision_lesson_caused_by(self):
        items = [
            _make_item("decision", memory_type="decision", scope="proj", age_hours=0),
            _make_item("lesson", memory_type="lesson_learned", scope="proj", age_hours=12),
        ]
        links = detect_type_based_links(items)
        assert len(links) >= 1
        rels = {lnk["relationship"] for lnk in links}
        assert REL_CAUSED_BY in rels or REL_RELATES_TO in rels

    def test_fact_supports_rule(self):
        items = [
            _make_item("fact", memory_type="fact", scope="proj", age_hours=0),
            _make_item("rule", memory_type="rule", scope="proj", age_hours=6),
        ]
        links = detect_type_based_links(items)
        assert any(lnk["relationship"] == REL_SUPPORTS for lnk in links)

    def test_too_far_apart_no_link(self):
        """Items more than TEMPORAL_PROXIMITY_HOURS apart → no type-based link."""
        items = [
            _make_item("a", memory_type="decision", scope="proj", age_hours=0),
            _make_item("b", memory_type="lesson_learned", scope="proj", age_hours=TEMPORAL_PROXIMITY_HOURS + 24),
        ]
        links = detect_type_based_links(items)
        assert links == []

    def test_cross_scope_no_link(self):
        """Items in different scopes don't get type-based links."""
        items = [
            _make_item("a", memory_type="decision", scope="proj-a", age_hours=0),
            _make_item("b", memory_type="lesson_learned", scope="proj-b", age_hours=0),
        ]
        links = detect_type_based_links(items)
        assert links == []


# ---------------------------------------------------------------------------
# detect_all_links (combined + dedup)
# ---------------------------------------------------------------------------

class TestDetectAllLinks:
    def test_combines_strategies(self):
        items = [
            _make_item("new", entity="config", prop="port", value="9224",
                       supersedes="old", memory_type="decision", scope="proj", age_hours=0),
            _make_item("old", entity="config", prop="port", value="9222",
                       memory_type="decision", scope="proj", age_hours=24),
        ]
        links = detect_all_links(items)
        # Should have links from multiple strategies
        assert len(links) >= 1
        # Explicit supersedes should win with confidence=1.0
        supersedes_links = [lnk for lnk in links if lnk["relationship"] == REL_SUPERSEDES]
        assert any(lnk["confidence"] == 1.0 for lnk in supersedes_links)

    def test_dedup_keeps_highest_confidence(self):
        items = [
            _make_item("a", entity="x", prop="y", value="1",
                       supersedes="b", age_hours=0),
            _make_item("b", entity="x", prop="y", value="2", age_hours=24),
        ]
        links = detect_all_links(items)
        # Explicit supersedes (conf=1.0) should beat entity-based (conf=0.85)
        supersedes = [lnk for lnk in links if lnk["source_id"] == "a" and lnk["target_id"] == "b" and lnk["relationship"] == REL_SUPERSEDES]
        assert len(supersedes) == 1
        assert supersedes[0]["confidence"] == 1.0

    def test_empty_items(self):
        assert detect_all_links([]) == []

    def test_single_item(self):
        links = detect_all_links([_make_item("solo")])
        assert links == []
