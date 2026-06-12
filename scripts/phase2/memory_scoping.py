"""
memory_scoping.py — Multi-agent memory ownership, visibility, and routing.

Phase 7, Items 19-21:
19. Scoped memory ownership / visibility layers
20. Memory routing between agents
21. Anti-pollution controls for shared memory

Memory layers:
- PRIVATE: agent's own working memory (only visible to that agent)
- PROJECT: shared within a project (visible to all agents on that project)
- CANONICAL: vetted truths (globally visible, high promotion bar)
- USER: user-level facts/preferences (visible to all agents for that user)
- EPHEMERAL: task-scoped, auto-expires (visible only during task lifetime)

Routing:
- Determines what memory each agent needs based on role and task
- Controls promotion from private → project → canonical
- Generates handoff summaries when passing work between agents

Anti-pollution:
- Source agent tagging on all items
- Stricter promotion for shared/canonical memory
- Confidence weighting by agent role trust
- Cross-agent agreement tracking
- Dispute flags for conflicting agent claims
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

LOGGER = logging.getLogger("openclaw.memory_scoping")

DB_DSN = os.environ.get("OPENCLAW_MEMORY_DSN", "dbname=openclaw_memory")


# ---------------------------------------------------------------------------
# Scope layers
# ---------------------------------------------------------------------------

class MemoryLayer(str, Enum):
    PRIVATE = "private"
    PROJECT = "project"
    CANONICAL = "canonical"
    USER = "user"
    EPHEMERAL = "ephemeral"


class AgentRole(str, Enum):
    ARCHITECT = "architect"
    BUILDER = "builder"
    REVIEWER = "reviewer"
    TESTER = "tester"
    PERF = "perf"
    ORCHESTRATOR = "orchestrator"
    ASSISTANT = "assistant"
    UNKNOWN = "unknown"


# Role trust levels (higher = more trusted for promoting to shared memory)
ROLE_TRUST: dict[AgentRole, float] = {
    AgentRole.ARCHITECT: 0.95,
    AgentRole.REVIEWER: 0.90,
    AgentRole.TESTER: 0.85,
    AgentRole.BUILDER: 0.80,
    AgentRole.PERF: 0.80,
    AgentRole.ORCHESTRATOR: 0.75,
    AgentRole.ASSISTANT: 0.70,
    AgentRole.UNKNOWN: 0.50,
}

# Layer promotion requirements
LAYER_PROMOTION: dict[tuple[MemoryLayer, MemoryLayer], dict] = {
    # private → project: moderate bar
    (MemoryLayer.PRIVATE, MemoryLayer.PROJECT): {
        "min_confidence": 0.65,
        "min_mentions": 1,
        "requires_explicit_authority": False,
    },
    # project → canonical: high bar
    (MemoryLayer.PROJECT, MemoryLayer.CANONICAL): {
        "min_confidence": 0.85,
        "min_mentions": 2,
        "min_agents": 2,  # Must be corroborated by multiple agents
        "requires_explicit_authority": True,
        "max_contradictions": 0,
    },
    # private → canonical: very high bar (skip project)
    (MemoryLayer.PRIVATE, MemoryLayer.CANONICAL): {
        "min_confidence": 0.90,
        "min_mentions": 3,
        "min_agents": 2,
        "requires_explicit_authority": True,
        "max_contradictions": 0,
    },
    # any → user: requires user-sourced authority
    (MemoryLayer.PRIVATE, MemoryLayer.USER): {
        "min_confidence": 0.70,
        "requires_user_authority": True,
    },
    (MemoryLayer.PROJECT, MemoryLayer.USER): {
        "min_confidence": 0.70,
        "requires_user_authority": True,
    },
}


# ---------------------------------------------------------------------------
# Scoped item
# ---------------------------------------------------------------------------

@dataclass
class ScopedMemoryItem:
    """A memory item with explicit scope metadata."""
    item_id: str
    text: str
    layer: MemoryLayer
    owner_agent: str
    project_id: str = ""
    visibility: list[str] = field(default_factory=list)  # Agent IDs that can see this
    source_agent: str = ""
    confidence: float = 0.0
    role_trust: float = 0.5
    dispute_flag: bool = False
    dispute_agents: list[str] = field(default_factory=list)
    cross_agent_agreement: int = 0  # Number of agents that corroborate

    def to_dict(self) -> dict:
        return {
            "item_id": self.item_id,
            "layer": self.layer.value,
            "owner_agent": self.owner_agent,
            "project_id": self.project_id,
            "visibility": self.visibility,
            "confidence": round(self.confidence, 4),
            "role_trust": round(self.role_trust, 4),
            "dispute_flag": self.dispute_flag,
            "cross_agent_agreement": self.cross_agent_agreement,
        }


# ---------------------------------------------------------------------------
# Visibility rules
# ---------------------------------------------------------------------------

def get_visible_layers(
    agent_id: str,
    agent_role: AgentRole,
    project_id: str | None = None,
) -> list[MemoryLayer]:
    """Determine which memory layers an agent can access."""
    layers = [MemoryLayer.CANONICAL, MemoryLayer.USER]

    # All agents see their own private memory
    layers.append(MemoryLayer.PRIVATE)

    # Project memory if assigned to a project
    if project_id:
        layers.append(MemoryLayer.PROJECT)

    # Ephemeral only during active tasks
    layers.append(MemoryLayer.EPHEMERAL)

    return layers


def build_visibility_filter(
    agent_id: str,
    agent_role: AgentRole,
    project_id: str | None = None,
) -> dict:
    """
    Build a SQL-compatible filter for memory visibility.

    Returns a dict with 'where_clause' and 'params' for use in queries.
    """
    layers = get_visible_layers(agent_id, agent_role, project_id)
    layer_values = [l.value for l in layers]

    conditions = []
    params: list[Any] = []

    # Canonical and user layers: always visible
    conditions.append("memory_layer IN ('canonical', 'user')")

    # Project layer: only if matching project
    if project_id:
        conditions.append("(memory_layer = 'project' AND (project_id = %s OR project_id IS NULL))")
        params.append(project_id)

    # Private layer: only own items
    conditions.append("(memory_layer = 'private' AND source_agent = %s)")
    params.append(agent_id)

    # Ephemeral: only own items (auto-expires separately)
    conditions.append("(memory_layer = 'ephemeral' AND source_agent = %s)")
    params.append(agent_id)

    where = "(" + " OR ".join(conditions) + ")"
    return {"where_clause": where, "params": params}


# ---------------------------------------------------------------------------
# Promotion between layers
# ---------------------------------------------------------------------------

@dataclass
class PromotionResult:
    """Result of attempting to promote an item to a higher layer."""
    item_id: str
    from_layer: MemoryLayer
    to_layer: MemoryLayer
    promoted: bool
    reason: str
    checks: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "item_id": self.item_id,
            "from": self.from_layer.value,
            "to": self.to_layer.value,
            "promoted": self.promoted,
            "reason": self.reason,
            "checks": self.checks,
        }


def check_layer_promotion(
    item: dict,
    from_layer: MemoryLayer,
    to_layer: MemoryLayer,
    corroboration: dict | None = None,
) -> PromotionResult:
    """
    Check if an item can be promoted from one layer to another.

    Args:
        item: memory item dict
        from_layer: current layer
        to_layer: target layer
        corroboration: dict with 'mention_count', 'distinct_agents', 'distinct_sessions'
    """
    key = (from_layer, to_layer)
    reqs = LAYER_PROMOTION.get(key)

    if reqs is None:
        return PromotionResult(
            item_id=str(item.get("id", "")),
            from_layer=from_layer,
            to_layer=to_layer,
            promoted=False,
            reason=f"No promotion path from {from_layer.value} to {to_layer.value}",
        )

    checks = []
    failed = []

    confidence = float(item.get("confidence", 0) or 0)
    authority = str(item.get("authority_basis", "unknown")).lower()
    corr = corroboration or {}

    # Confidence check
    min_conf = reqs.get("min_confidence", 0)
    if confidence >= min_conf:
        checks.append(f"confidence: {confidence:.2f} >= {min_conf}")
    else:
        failed.append(f"confidence: {confidence:.2f} < {min_conf}")

    # Mention count
    min_mentions = reqs.get("min_mentions", 0)
    if min_mentions > 0:
        mentions = corr.get("mention_count", 1)
        if mentions >= min_mentions:
            checks.append(f"mentions: {mentions} >= {min_mentions}")
        else:
            failed.append(f"mentions: {mentions} < {min_mentions}")

    # Multi-agent corroboration
    min_agents = reqs.get("min_agents", 0)
    if min_agents > 0:
        agents = corr.get("distinct_agents", 1)
        if agents >= min_agents:
            checks.append(f"agents: {agents} >= {min_agents}")
        else:
            failed.append(f"agents: {agents} < {min_agents}")

    # Explicit authority
    if reqs.get("requires_explicit_authority"):
        explicit = {"user_explicit", "developer_explicit", "tester_explicit", "system_tool_verified"}
        if authority in explicit:
            checks.append(f"authority: {authority} (explicit)")
        else:
            failed.append(f"authority: {authority} (not explicit)")

    # User authority
    if reqs.get("requires_user_authority"):
        if authority == "user_explicit":
            checks.append("user_authority: confirmed")
        else:
            failed.append(f"user_authority: {authority} (need user_explicit)")

    # Contradictions
    max_contradictions = reqs.get("max_contradictions")
    if max_contradictions is not None:
        contras = int(item.get("contradiction_count", 0) or 0)
        if contras <= max_contradictions:
            checks.append(f"contradictions: {contras} <= {max_contradictions}")
        else:
            failed.append(f"contradictions: {contras} > {max_contradictions}")

    promoted = len(failed) == 0
    reason = "all checks passed" if promoted else "; ".join(failed)

    return PromotionResult(
        item_id=str(item.get("id", "")),
        from_layer=from_layer,
        to_layer=to_layer,
        promoted=promoted,
        reason=reason,
        checks=checks + [f"FAILED: {f}" for f in failed],
    )


# ---------------------------------------------------------------------------
# Handoff summaries
# ---------------------------------------------------------------------------

@dataclass
class HandoffSummary:
    """Summary generated when passing work between agents."""
    from_agent: str
    to_agent: str
    project_id: str
    context_items: list[dict] = field(default_factory=list)
    decisions_made: list[str] = field(default_factory=list)
    open_questions: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    item_count: int = 0

    def to_context_block(self) -> str:
        """Render as a context block for the receiving agent."""
        lines = [
            f"## Handoff from {self.from_agent} → {self.to_agent}",
            f"Project: {self.project_id}",
            f"Items transferred: {self.item_count}",
        ]

        if self.decisions_made:
            lines.append("\n### Decisions Made:")
            for d in self.decisions_made:
                lines.append(f"- {d}")

        if self.open_questions:
            lines.append("\n### Open Questions:")
            for q in self.open_questions:
                lines.append(f"- {q}")

        if self.warnings:
            lines.append("\n### Warnings:")
            for w in self.warnings:
                lines.append(f"- ⚠ {w}")

        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "from_agent": self.from_agent,
            "to_agent": self.to_agent,
            "project_id": self.project_id,
            "item_count": self.item_count,
            "decisions": len(self.decisions_made),
            "open_questions": len(self.open_questions),
            "warnings": len(self.warnings),
        }


def generate_handoff(
    from_agent: str,
    to_agent: str,
    project_id: str,
    max_items: int = 30,
    conn: Any = None,
) -> HandoffSummary:
    """
    Generate a handoff summary for passing work between agents.

    Collects:
    - Decisions and rules from the originating agent
    - Open questions
    - Relevant project memory
    - Contradiction warnings
    """
    should_close = False
    if conn is None:
        import psycopg
        conn = psycopg.connect(DB_DSN)
        should_close = True

    handoff = HandoffSummary(
        from_agent=from_agent,
        to_agent=to_agent,
        project_id=project_id,
    )

    try:
        with conn.cursor() as cur:
            # Get decisions and rules from the source agent
            cur.execute("""
                SELECT id, text, memory_type, confidence, COALESCE(contradiction_count, 0)
                FROM memory_items
                WHERE source_agent = %s
                  AND status = 'active'
                  AND memory_type IN ('decision', 'rule', 'preference')
                  AND (scope = %s OR scope = 'stable' OR scope IS NULL)
                ORDER BY confidence DESC
                LIMIT %s
            """, (from_agent, project_id, max_items))

            for row in cur.fetchall():
                item = {"id": row[0], "text": row[1], "type": row[2], "confidence": row[3], "contradictions": row[4]}
                handoff.context_items.append(item)

                if row[2] == "decision":
                    handoff.decisions_made.append(row[1][:200])
                elif row[4] > 0:
                    handoff.warnings.append(f"Item {row[0][:12]} has {row[4]} contradiction(s): {row[1][:100]}")

            # Get open questions
            cur.execute("""
                SELECT text FROM memory_items
                WHERE source_agent = %s
                  AND status = 'active'
                  AND memory_type = 'open_question'
                  AND (scope = %s OR scope = 'stable' OR scope IS NULL)
                ORDER BY created_at DESC
                LIMIT 10
            """, (from_agent, project_id))

            for row in cur.fetchall():
                handoff.open_questions.append(row[0][:200])

            handoff.item_count = len(handoff.context_items)

    finally:
        if should_close:
            conn.close()

    return handoff


# ---------------------------------------------------------------------------
# Anti-pollution: dispute detection
# ---------------------------------------------------------------------------

def detect_cross_agent_disputes(
    project_id: str | None = None,
    conn: Any = None,
) -> list[dict]:
    """
    Find items where different agents have conflicting claims
    on the same entity+property slot.

    Returns list of disputes with both sides and their agent sources.
    """
    should_close = False
    if conn is None:
        import psycopg
        conn = psycopg.connect(DB_DSN)
        should_close = True

    disputes = []
    try:
        with conn.cursor() as cur:
            scope_filter = ""
            params: list[Any] = []
            if project_id:
                scope_filter = "AND (a.scope = %s OR a.scope = 'stable')"
                params.append(project_id)

            cur.execute(f"""
                SELECT a.id, a.text, a.source_agent, a.value, a.confidence,
                       b.id, b.text, b.source_agent, b.value, b.confidence,
                       a.entity, a.property
                FROM memory_items a
                JOIN memory_items b
                  ON a.entity = b.entity
                 AND a.property = b.property
                 AND a.id < b.id
                WHERE a.status = 'active' AND b.status = 'active'
                  AND a.source_agent IS DISTINCT FROM b.source_agent
                  AND a.value IS DISTINCT FROM b.value
                  AND a.entity IS NOT NULL AND a.property IS NOT NULL
                  {scope_filter}
                ORDER BY a.entity, a.property
                LIMIT 100
            """, params)

            for row in cur.fetchall():
                disputes.append({
                    "entity": row[10],
                    "property": row[11],
                    "agent_a": {"id": row[0], "text": row[1][:200], "agent": row[2], "value": row[3], "confidence": float(row[4] or 0)},
                    "agent_b": {"id": row[5], "text": row[6][:200], "agent": row[7], "value": row[8], "confidence": float(row[9] or 0)},
                })

    finally:
        if should_close:
            conn.close()

    return disputes


# ---------------------------------------------------------------------------
# Schema migration for scoping columns
# ---------------------------------------------------------------------------

SCOPING_MIGRATION_SQL = """
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'memory_items' AND column_name = 'memory_layer'
    ) THEN
        ALTER TABLE memory_items ADD COLUMN memory_layer TEXT DEFAULT 'canonical';
    END IF;
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'memory_items' AND column_name = 'owner_agent'
    ) THEN
        ALTER TABLE memory_items ADD COLUMN owner_agent TEXT;
    END IF;
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'memory_items' AND column_name = 'project_id'
    ) THEN
        ALTER TABLE memory_items ADD COLUMN project_id TEXT;
    END IF;
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'memory_items' AND column_name = 'cross_agent_agreement'
    ) THEN
        ALTER TABLE memory_items ADD COLUMN cross_agent_agreement INT DEFAULT 0;
    END IF;
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'memory_items' AND column_name = 'dispute_flag'
    ) THEN
        ALTER TABLE memory_items ADD COLUMN dispute_flag BOOLEAN DEFAULT FALSE;
    END IF;
END $$;

CREATE INDEX IF NOT EXISTS idx_memory_items_layer ON memory_items(memory_layer);
CREATE INDEX IF NOT EXISTS idx_memory_items_owner ON memory_items(owner_agent);
CREATE INDEX IF NOT EXISTS idx_memory_items_project ON memory_items(project_id);
"""


def apply_scoping_migration(conn: Any = None):
    """Apply scoping columns to memory_items table."""
    should_close = False
    if conn is None:
        import psycopg
        conn = psycopg.connect(DB_DSN)
        should_close = True

    try:
        with conn.cursor() as cur:
            cur.execute(SCOPING_MIGRATION_SQL)
        conn.commit()
        LOGGER.info("scoping migration applied")
    finally:
        if should_close:
            conn.close()
