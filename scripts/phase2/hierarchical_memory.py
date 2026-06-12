"""
hierarchical_memory.py — Multi-resolution memory views and reversible summaries.

Phase 6, Items 16-18:
16. Reversible summaries (linked to source memories, expandable)
17. Hierarchical memory views (item → event → day → week → project → profile)
18. Summary quality evaluation

Design:
- Summaries are NOT lossy blobs — they link back to source items
- Contradictions are preserved, not flattened
- Summaries have confidence that reflects source confidence
- Each level can be expanded back to its constituents

No LLM calls for structure — LLM used only for text generation
when explicitly requested.
"""
from __future__ import annotations

import logging
import os
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any

LOGGER = logging.getLogger("openclaw.hierarchical_memory")

DB_DSN = os.environ.get("OPENCLAW_MEMORY_DSN", "dbname=openclaw_memory")


# ---------------------------------------------------------------------------
# Resolution levels
# ---------------------------------------------------------------------------

class ResolutionLevel(str, Enum):
    ITEM = "item"            # Individual memory items
    EVENT = "event"          # Grouped by session/event
    DAY = "day"              # Daily summaries
    WEEK = "week"            # Weekly summaries
    PROJECT = "project"      # Per-project summaries
    PROFILE = "profile"      # Long-term profile summaries


@dataclass
class MemoryNode:
    """A node in the hierarchical memory tree."""
    level: ResolutionLevel
    key: str                    # Unique key (date, project_id, etc.)
    label: str = ""
    item_count: int = 0
    avg_confidence: float = 0.0
    has_contradictions: bool = False
    contradiction_count: int = 0
    source_item_ids: list[str] = field(default_factory=list)
    children: list[MemoryNode] = field(default_factory=list)
    summary_text: str = ""
    memory_types: dict[str, int] = field(default_factory=dict)  # type → count
    date_range: tuple[str, str] = ("", "")

    def to_dict(self, include_children: bool = True, max_depth: int = 3) -> dict:
        d = {
            "level": self.level.value,
            "key": self.key,
            "label": self.label,
            "item_count": self.item_count,
            "avg_confidence": round(self.avg_confidence, 4),
            "has_contradictions": self.has_contradictions,
            "contradiction_count": self.contradiction_count,
            "memory_types": self.memory_types,
            "date_range": self.date_range,
            "source_item_count": len(self.source_item_ids),
        }
        if self.summary_text:
            d["summary"] = self.summary_text[:500]
        if include_children and max_depth > 0:
            d["children"] = [c.to_dict(True, max_depth - 1) for c in self.children]
        else:
            d["child_count"] = len(self.children)
        return d


# ---------------------------------------------------------------------------
# Summary structure (reversible)
# ---------------------------------------------------------------------------

@dataclass
class ReversibleSummary:
    """A summary that can be expanded back to its source items."""
    summary_id: str
    level: ResolutionLevel
    text: str
    source_item_ids: list[str]
    source_count: int
    confidence: float           # Min confidence of sources
    avg_confidence: float
    has_unresolved_contradictions: bool
    contradiction_details: list[str] = field(default_factory=list)
    preserved_ambiguities: list[str] = field(default_factory=list)
    date_range: tuple[str, str] = ("", "")

    def to_dict(self) -> dict:
        return {
            "summary_id": self.summary_id,
            "level": self.level.value,
            "text": self.text,
            "source_count": self.source_count,
            "confidence": round(self.confidence, 4),
            "avg_confidence": round(self.avg_confidence, 4),
            "has_unresolved_contradictions": self.has_unresolved_contradictions,
            "contradiction_details": self.contradiction_details,
            "preserved_ambiguities": self.preserved_ambiguities,
            "expandable": True,
            "source_item_ids": self.source_item_ids[:20],  # Cap for display
        }


# ---------------------------------------------------------------------------
# Hierarchy builder
# ---------------------------------------------------------------------------

def build_hierarchy(
    project_id: str | None = None,
    days_back: int = 30,
    conn: Any = None,
) -> MemoryNode:
    """
    Build a hierarchical memory tree from the database.

    Structure:
        Root (profile/project)
        └── Week nodes
            └── Day nodes
                └── Event nodes (by session)
                    └── Item nodes
    """
    should_close = False
    if conn is None:
        import psycopg
        conn = psycopg.connect(DB_DSN)
        should_close = True

    try:
        with conn.cursor() as cur:
            now = datetime.now(timezone.utc)
            start = now - timedelta(days=days_back)

            where = "status IN ('active', 'candidate') AND created_at >= %s"
            params: list[Any] = [start]

            if project_id:
                where += " AND (scope = %s OR scope = 'stable' OR scope IS NULL)"
                params.append(project_id)

            cur.execute(f"""
                SELECT id, text, memory_type, entity, confidence, source_session,
                       created_at, COALESCE(contradiction_count, 0)
                FROM memory_items
                WHERE {where}
                ORDER BY created_at
            """, params)

            items = []
            for row in cur.fetchall():
                items.append({
                    "id": row[0],
                    "text": row[1] or "",
                    "memory_type": row[2] or "",
                    "entity": row[3] or "",
                    "confidence": float(row[4] or 0),
                    "source_session": row[5] or "",
                    "created_at": row[6],
                    "contradiction_count": int(row[7] or 0),
                })

        # Build tree bottom-up
        # Group by day
        by_day: dict[str, list[dict]] = defaultdict(list)
        for item in items:
            if item["created_at"]:
                day_key = item["created_at"].strftime("%Y-%m-%d")
            else:
                day_key = "unknown"
            by_day[day_key].append(item)

        # Group days into weeks
        by_week: dict[str, list[str]] = defaultdict(list)
        for day_key in sorted(by_day.keys()):
            if day_key == "unknown":
                week_key = "unknown"
            else:
                try:
                    dt = datetime.strptime(day_key, "%Y-%m-%d")
                    week_start = dt - timedelta(days=dt.weekday())
                    week_key = f"week-{week_start.strftime('%Y-%m-%d')}"
                except ValueError:
                    week_key = "unknown"
            by_week[week_key].append(day_key)

        # Build nodes
        root = MemoryNode(
            level=ResolutionLevel.PROJECT if project_id else ResolutionLevel.PROFILE,
            key=project_id or "all",
            label=f"Project: {project_id}" if project_id else "All Memory",
        )

        all_item_ids = []
        all_confidences = []
        type_counts: dict[str, int] = defaultdict(int)
        total_contradictions = 0

        for week_key in sorted(by_week.keys()):
            week_node = MemoryNode(level=ResolutionLevel.WEEK, key=week_key, label=week_key)

            for day_key in sorted(by_week[week_key]):
                day_items = by_day[day_key]
                day_node = MemoryNode(level=ResolutionLevel.DAY, key=day_key, label=day_key)

                # Group by session within day
                by_session: dict[str, list[dict]] = defaultdict(list)
                for item in day_items:
                    session = item.get("source_session", "unknown") or "unknown"
                    by_session[session].append(item)

                for session_key, session_items in by_session.items():
                    event_node = MemoryNode(
                        level=ResolutionLevel.EVENT,
                        key=f"{day_key}:{session_key[:16]}",
                        label=f"Session {session_key[:16]}",
                        item_count=len(session_items),
                        source_item_ids=[i["id"] for i in session_items],
                    )

                    confidences = [i["confidence"] for i in session_items if i["confidence"]]
                    if confidences:
                        event_node.avg_confidence = sum(confidences) / len(confidences)

                    event_contras = sum(i["contradiction_count"] for i in session_items)
                    event_node.has_contradictions = event_contras > 0
                    event_node.contradiction_count = event_contras

                    for i in session_items:
                        if i["memory_type"]:
                            event_node.memory_types[i["memory_type"]] = event_node.memory_types.get(i["memory_type"], 0) + 1

                    day_node.children.append(event_node)
                    day_node.source_item_ids.extend(event_node.source_item_ids)

                day_node.item_count = sum(c.item_count for c in day_node.children)
                day_confidences = [i["confidence"] for i in day_items if i["confidence"]]
                if day_confidences:
                    day_node.avg_confidence = sum(day_confidences) / len(day_confidences)
                day_contras = sum(i["contradiction_count"] for i in day_items)
                day_node.has_contradictions = day_contras > 0
                day_node.contradiction_count = day_contras

                if day_items:
                    dates = [str(i["created_at"]) for i in day_items if i["created_at"]]
                    if dates:
                        day_node.date_range = (min(dates), max(dates))

                week_node.children.append(day_node)
                week_node.source_item_ids.extend(day_node.source_item_ids)

                # Accumulate for root
                for i in day_items:
                    all_item_ids.append(i["id"])
                    all_confidences.append(i["confidence"])
                    if i["memory_type"]:
                        type_counts[i["memory_type"]] += 1
                    total_contradictions += i["contradiction_count"]

            week_node.item_count = sum(c.item_count for c in week_node.children)
            week_all_confs = [c for c in all_confidences if c]  # Use all so far for running avg
            if week_all_confs:
                week_node.avg_confidence = sum(week_all_confs) / len(week_all_confs)

            root.children.append(week_node)

        root.item_count = len(all_item_ids)
        root.source_item_ids = all_item_ids
        root.memory_types = dict(type_counts)
        root.has_contradictions = total_contradictions > 0
        root.contradiction_count = total_contradictions
        if all_confidences:
            root.avg_confidence = sum(all_confidences) / len(all_confidences)

        return root

    finally:
        if should_close:
            conn.close()


# ---------------------------------------------------------------------------
# Reversible summary generation
# ---------------------------------------------------------------------------

def generate_extractive_summary(
    items: list[dict],
    level: ResolutionLevel,
    max_lines: int = 10,
) -> ReversibleSummary:
    """
    Generate an extractive (non-LLM) reversible summary from a set of items.

    Picks the highest-confidence items as representative lines.
    Preserves contradictions as explicit notes.
    """
    import hashlib

    if not items:
        return ReversibleSummary(
            summary_id="empty",
            level=level,
            text="No items.",
            source_item_ids=[],
            source_count=0,
            confidence=0.0,
            avg_confidence=0.0,
            has_unresolved_contradictions=False,
        )

    # Sort by confidence
    sorted_items = sorted(items, key=lambda x: float(x.get("confidence", 0) or 0), reverse=True)
    top_items = sorted_items[:max_lines]

    # Build summary text
    lines = []
    for item in top_items:
        text = (item.get("text") or "")[:200]
        mtype = item.get("memory_type", "")
        prefix = f"[{mtype}] " if mtype else ""
        lines.append(f"{prefix}{text}")

    # Check for contradictions
    contradicted = [i for i in items if int(i.get("contradiction_count", 0) or 0) > 0]
    contradiction_details = []
    if contradicted:
        for c in contradicted[:5]:
            contradiction_details.append(
                f"Item {c['id'][:16]} has {c.get('contradiction_count', 0)} active contradiction(s)"
            )

    # Preserved ambiguities (items with low confidence)
    ambiguous = [i for i in items if 0 < float(i.get("confidence", 0) or 0) < 0.5]
    preserved_ambiguities = [
        f"{i.get('text', '')[:100]} (confidence: {i.get('confidence', 0):.0%})"
        for i in ambiguous[:3]
    ]

    confidences = [float(i.get("confidence", 0) or 0) for i in items]
    min_conf = min(confidences) if confidences else 0
    avg_conf = sum(confidences) / len(confidences) if confidences else 0

    source_ids = [i.get("id", "") for i in items if i.get("id")]
    dates = [str(i.get("created_at", "")) for i in items if i.get("created_at")]

    summary_id = hashlib.sha256(
        "|".join(source_ids).encode()
    ).hexdigest()[:12]

    return ReversibleSummary(
        summary_id=summary_id,
        level=level,
        text="\n".join(lines),
        source_item_ids=source_ids,
        source_count=len(items),
        confidence=min_conf,
        avg_confidence=avg_conf,
        has_unresolved_contradictions=len(contradicted) > 0,
        contradiction_details=contradiction_details,
        preserved_ambiguities=preserved_ambiguities,
        date_range=(min(dates), max(dates)) if dates else ("", ""),
    )


def summarize_day(
    date: str,
    conn: Any = None,
    max_lines: int = 15,
) -> ReversibleSummary:
    """Generate a reversible summary for a single day."""
    should_close = False
    if conn is None:
        import psycopg
        conn = psycopg.connect(DB_DSN)
        should_close = True

    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT id, text, memory_type, entity, confidence, source_session,
                       created_at, COALESCE(contradiction_count, 0)
                FROM memory_items
                WHERE status = 'active'
                  AND DATE(created_at) = %s
                ORDER BY confidence DESC
            """, (date,))

            items = [
                {
                    "id": r[0], "text": r[1], "memory_type": r[2], "entity": r[3],
                    "confidence": float(r[4] or 0), "source_session": r[5],
                    "created_at": r[6], "contradiction_count": int(r[7] or 0),
                }
                for r in cur.fetchall()
            ]

        return generate_extractive_summary(items, ResolutionLevel.DAY, max_lines)

    finally:
        if should_close:
            conn.close()
