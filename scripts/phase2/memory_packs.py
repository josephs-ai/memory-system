"""
memory_packs.py — Query-specific context assembly.

Phase 5, Items 13-15:
13. Build memory packs (concise fact pack, decision history, preference profile, etc.)
14. Task-aware retrieval policies
15. Recency + durability balancing by task type

Instead of dumping raw memory items into context, assemble structured
"memory packs" optimized for the task at hand.

Pack types:
- fact_pack: concise factual statements
- decision_history: chronological decisions with rationale
- preference_profile: user preferences organized by category
- project_snapshot: current project state
- code_context: implementation patterns, bug history, architecture rules
- timeline_summary: chronological event summary

Task-aware retrieval modes:
- coding: prioritize patterns, bugs, architecture
- assistant: prioritize preferences, decisions, schedule
- review: prioritize decisions, constraints, defects
- planning: prioritize goals, open issues, prior attempts

Recency modes:
- recent_first: favor fresh items
- canonical_first: favor high-confidence durable items
- balanced: default weighted blend
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

LOGGER = logging.getLogger("openclaw.memory_packs")

DB_DSN = os.environ.get("OPENCLAW_MEMORY_DSN", "dbname=openclaw_memory")


# ---------------------------------------------------------------------------
# Pack types
# ---------------------------------------------------------------------------

class PackType(str, Enum):
    FACT_PACK = "fact_pack"
    DECISION_HISTORY = "decision_history"
    PREFERENCE_PROFILE = "preference_profile"
    PROJECT_SNAPSHOT = "project_snapshot"
    CODE_CONTEXT = "code_context"
    TIMELINE_SUMMARY = "timeline_summary"


class TaskType(str, Enum):
    CODING = "coding"
    ASSISTANT = "assistant"
    REVIEW = "review"
    PLANNING = "planning"
    GENERAL = "general"


class RecencyMode(str, Enum):
    RECENT_FIRST = "recent_first"
    CANONICAL_FIRST = "canonical_first"
    BALANCED = "balanced"
    TIMELINE = "timeline"
    STRICT_HIGH_CONFIDENCE = "strict_high_confidence"


# Task → preferred memory types and pack types
TASK_POLICIES: dict[TaskType, dict] = {
    TaskType.CODING: {
        "pack_types": [PackType.CODE_CONTEXT, PackType.FACT_PACK],
        "memory_types": ["implementation_pattern", "bug_history", "learned_fix", "rule", "decision"],
        "recency_mode": RecencyMode.BALANCED,
        "min_confidence": 0.5,
        "prefer_entities": ["code", "architecture", "api", "database", "config"],
    },
    TaskType.ASSISTANT: {
        "pack_types": [PackType.PREFERENCE_PROFILE, PackType.FACT_PACK],
        "memory_types": ["preference", "fact", "decision"],
        "recency_mode": RecencyMode.RECENT_FIRST,
        "min_confidence": 0.4,
        "prefer_entities": ["user_preference", "schedule", "contact"],
    },
    TaskType.REVIEW: {
        "pack_types": [PackType.DECISION_HISTORY, PackType.CODE_CONTEXT],
        "memory_types": ["decision", "rule", "bug_history", "implementation_pattern"],
        "recency_mode": RecencyMode.CANONICAL_FIRST,
        "min_confidence": 0.6,
        "prefer_entities": [],
    },
    TaskType.PLANNING: {
        "pack_types": [PackType.PROJECT_SNAPSHOT, PackType.DECISION_HISTORY, PackType.TIMELINE_SUMMARY],
        "memory_types": ["decision", "fact", "rule"],
        "recency_mode": RecencyMode.BALANCED,
        "min_confidence": 0.5,
        "prefer_entities": ["project", "milestone", "goal"],
    },
    TaskType.GENERAL: {
        "pack_types": [PackType.FACT_PACK],
        "memory_types": [],  # No filter
        "recency_mode": RecencyMode.BALANCED,
        "min_confidence": 0.4,
        "prefer_entities": [],
    },
}


# ---------------------------------------------------------------------------
# Memory pack structure
# ---------------------------------------------------------------------------

@dataclass
class MemoryPackItem:
    """A single item in a memory pack, formatted for context injection."""
    item_id: str
    text: str
    memory_type: str = ""
    entity: str = ""
    confidence: float = 0.0
    source: str = ""
    date: str = ""
    is_current_truth: bool = True
    contradiction_note: str = ""

    def to_context_line(self) -> str:
        """Format as a single context line for injection."""
        parts = []
        if self.memory_type:
            parts.append(f"[{self.memory_type}]")
        parts.append(self.text)
        if self.confidence < 0.6:
            parts.append(f"(confidence: {self.confidence:.0%})")
        if not self.is_current_truth:
            parts.append("(SUPERSEDED)")
        if self.contradiction_note:
            parts.append(f"⚠ {self.contradiction_note}")
        return " ".join(parts)


@dataclass
class MemoryPack:
    """An assembled memory pack ready for context injection."""
    pack_type: PackType
    task_type: TaskType
    recency_mode: RecencyMode
    items: list[MemoryPackItem] = field(default_factory=list)
    total_chars: int = 0
    total_tokens_approx: int = 0

    def to_context_block(self, max_items: int = 20) -> str:
        """Render the pack as a context block for injection into prompts."""
        lines = [f"## Memory: {self.pack_type.value.replace('_', ' ').title()}"]

        for item in self.items[:max_items]:
            lines.append(f"- {item.to_context_line()}")

        text = "\n".join(lines)
        self.total_chars = len(text)
        self.total_tokens_approx = len(text) // 4
        return text

    def to_dict(self) -> dict:
        return {
            "pack_type": self.pack_type.value,
            "task_type": self.task_type.value,
            "recency_mode": self.recency_mode.value,
            "item_count": len(self.items),
            "total_chars": self.total_chars,
            "total_tokens_approx": self.total_tokens_approx,
        }


# ---------------------------------------------------------------------------
# Task type detection
# ---------------------------------------------------------------------------

import re

_ASSISTANT_PATTERNS = re.compile(
    r'\b(prefer|preference|favorite|favourite|remind|remember|schedule|appointment|'
    r'weather|email|message|call|contact|habit|routine|like|dislike|'
    r'what do i|what is my|what are my|how do i usually|my default)\b',
    re.IGNORECASE,
)

_REVIEW_PATTERNS = re.compile(
    r'\b(review|audit|check|verify|validate|inspect|assess|evaluate|critique)\b',
    re.IGNORECASE,
)

_PLANNING_PATTERNS = re.compile(
    r'\b(plan|roadmap|milestone|goal|priority|schedule|timeline|phase|sprint|backlog)\b',
    re.IGNORECASE,
)

_CODING_PATTERNS = re.compile(
    r'\b(implement|write code|function|class|method|bug|fix|refactor|debug|compile|'
    r'import|module|endpoint|schema|index|syntax error|stack trace|traceback)\b',
    re.IGNORECASE,
)


def detect_task_type(query: str) -> TaskType:
    """Heuristic task type detection from query text.

    Precedence (most specific wins):
    1. Assistant — personal preferences, reminders, habits
    2. Review — audit/verify/critique (checked before coding to avoid
       'debug review issue' skewing to coding)
    3. Planning — roadmaps, goals, milestones
    4. Coding — implementation, bugs, refactoring
    5. General — fallback
    """
    # Score each category by match count; highest wins, with tie-break by precedence
    scores: list[tuple[int, int, TaskType]] = []  # (match_count, precedence, type)

    scores.append((len(_ASSISTANT_PATTERNS.findall(query)), 0, TaskType.ASSISTANT))
    scores.append((len(_REVIEW_PATTERNS.findall(query)), 1, TaskType.REVIEW))
    scores.append((len(_PLANNING_PATTERNS.findall(query)), 2, TaskType.PLANNING))
    scores.append((len(_CODING_PATTERNS.findall(query)), 3, TaskType.CODING))

    # Sort by match count descending, then precedence ascending (lower = higher priority)
    scores.sort(key=lambda x: (-x[0], x[1]))

    if scores[0][0] > 0:
        return scores[0][2]
    return TaskType.GENERAL


# ---------------------------------------------------------------------------
# Pack assembly
# ---------------------------------------------------------------------------

def _apply_recency_ordering(items: list[dict], mode: RecencyMode) -> list[dict]:
    """Sort items based on recency mode."""
    if mode == RecencyMode.RECENT_FIRST:
        return sorted(items, key=lambda x: x.get("created_at") or "", reverse=True)
    elif mode == RecencyMode.CANONICAL_FIRST:
        return sorted(items, key=lambda x: float(x.get("confidence", 0) or 0), reverse=True)
    elif mode == RecencyMode.TIMELINE:
        return sorted(items, key=lambda x: x.get("created_at") or "")
    elif mode == RecencyMode.STRICT_HIGH_CONFIDENCE:
        return sorted(
            [i for i in items if float(i.get("confidence", 0) or 0) >= 0.8],
            key=lambda x: float(x.get("confidence", 0) or 0),
            reverse=True,
        )
    else:  # balanced
        # Score = 0.6 * confidence + 0.4 * recency_rank
        by_recency = sorted(items, key=lambda x: x.get("created_at") or "", reverse=True)
        for rank, item in enumerate(by_recency):
            item["_recency_rank"] = 1.0 - (rank / max(len(by_recency), 1))
        return sorted(
            items,
            key=lambda x: 0.6 * float(x.get("confidence", 0) or 0) + 0.4 * x.get("_recency_rank", 0),
            reverse=True,
        )


def assemble_pack(
    pack_type: PackType,
    task_type: TaskType = TaskType.GENERAL,
    query: str = "",
    project_id: str | None = None,
    max_items: int = 20,
    recency_mode: RecencyMode | None = None,
    conn: Any = None,
) -> MemoryPack:
    """
    Assemble a memory pack from the database.

    Fetches relevant items based on pack_type and task_type,
    applies recency/durability ordering, and formats for injection.
    """
    policy = TASK_POLICIES.get(task_type, TASK_POLICIES[TaskType.GENERAL])
    if recency_mode is None:
        recency_mode = RecencyMode(policy["recency_mode"])

    min_confidence = policy.get("min_confidence", 0.4)
    memory_types = policy.get("memory_types", [])
    prefer_entities = policy.get("prefer_entities", [])

    should_close = False
    if conn is None:
        import psycopg
        conn = psycopg.connect(DB_DSN)
        should_close = True

    pack = MemoryPack(pack_type=pack_type, task_type=task_type, recency_mode=recency_mode)

    try:
        with conn.cursor() as cur:
            # Build query based on pack type
            where_clauses = ["status = 'active'"]
            params: list[Any] = []

            if min_confidence > 0:
                where_clauses.append("COALESCE(confidence, 0) >= %s")
                params.append(min_confidence)

            if memory_types:
                placeholders = ",".join(["%s"] * len(memory_types))
                where_clauses.append(f"memory_type IN ({placeholders})")
                params.extend(memory_types)

            if project_id:
                where_clauses.append("(scope = %s OR scope = 'stable' OR scope IS NULL)")
                params.append(project_id)

            # Pack-type specific filters
            if pack_type == PackType.PREFERENCE_PROFILE:
                where_clauses.append("memory_type = 'preference'")
            elif pack_type == PackType.DECISION_HISTORY:
                where_clauses.append("memory_type IN ('decision', 'rule')")
            elif pack_type == PackType.CODE_CONTEXT:
                where_clauses.append("memory_type IN ('implementation_pattern', 'bug_history', 'learned_fix', 'rule')")

            # Current truth only (unless timeline)
            if pack_type != PackType.TIMELINE_SUMMARY:
                try:
                    where_clauses.append("COALESCE(is_current_truth, TRUE) = TRUE")
                except Exception:
                    pass

            where_sql = " AND ".join(where_clauses)
            sql = f"""
                SELECT id, text, memory_type, entity, confidence, source_agent,
                       created_at, COALESCE(is_current_truth, TRUE), COALESCE(contradiction_count, 0)
                FROM memory_items
                WHERE {where_sql}
                ORDER BY created_at DESC
                LIMIT %s
            """
            params.append(max_items * 3)  # Fetch extra for filtering

            cur.execute(sql, params)
            rows = cur.fetchall()

            items = []
            for row in rows:
                items.append({
                    "id": row[0],
                    "text": row[1] or "",
                    "memory_type": row[2] or "",
                    "entity": row[3] or "",
                    "confidence": float(row[4] or 0),
                    "source_agent": row[5] or "",
                    "created_at": str(row[6] or ""),
                    "is_current_truth": bool(row[7]),
                    "contradiction_count": int(row[8] or 0),
                })

            # Apply entity preference boost
            if prefer_entities:
                for item in items:
                    if any(pe in (item.get("entity") or "").lower() for pe in prefer_entities):
                        item["confidence"] = min(1.0, float(item.get("confidence", 0)) + 0.1)

            # Apply recency ordering
            ordered = _apply_recency_ordering(items, recency_mode)

            # Convert to pack items
            for item in ordered[:max_items]:
                contradiction_note = ""
                if item.get("contradiction_count", 0) > 0:
                    contradiction_note = f"{item['contradiction_count']} active contradiction(s)"

                pack.items.append(MemoryPackItem(
                    item_id=item["id"],
                    text=item["text"],
                    memory_type=item["memory_type"],
                    entity=item["entity"],
                    confidence=item["confidence"],
                    source=item["source_agent"],
                    date=item["created_at"],
                    is_current_truth=item["is_current_truth"],
                    contradiction_note=contradiction_note,
                ))

    finally:
        if should_close:
            conn.close()

    return pack


def assemble_task_pack(
    query: str,
    task_type: TaskType | None = None,
    project_id: str | None = None,
    max_items: int = 20,
    conn: Any = None,
) -> list[MemoryPack]:
    """
    Auto-detect task type and assemble appropriate memory packs.

    Returns multiple packs based on the task policy.
    """
    if task_type is None:
        task_type = detect_task_type(query)

    policy = TASK_POLICIES.get(task_type, TASK_POLICIES[TaskType.GENERAL])
    pack_types = policy.get("pack_types", [PackType.FACT_PACK])

    packs = []
    for pt in pack_types:
        pack = assemble_pack(
            pack_type=pt,
            task_type=task_type,
            query=query,
            project_id=project_id,
            max_items=max_items,
            conn=conn,
        )
        if pack.items:
            packs.append(pack)

    return packs
