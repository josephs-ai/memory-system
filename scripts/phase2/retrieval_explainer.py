"""
retrieval_explainer.py — Retrieval explanation and provenance system.

Phase 3, Items 7-9:
7. Retrieval explanations (why each item ranked)
8. Provenance bundles (source, dates, lifecycle, edges)
9. "Why not retrieved?" diagnostics

Every retrieved item can explain:
- Lexical match contribution (FTS score)
- Vector similarity contribution
- Reranker effect (before/after delta)
- Temporal boost/penalty
- Feedback boost/penalty
- Graph expansion origin
- Lifecycle eligibility
- Contradiction state

Provenance bundles attach:
- Source transcript/artifact
- Creation date, last reconfirmed
- Current lifecycle state
- Superseded-by / supported-by edges
- Why included for this query

"Why not retrieved?" answers:
- Was it never extracted?
- Filtered by lifecycle?
- Missing embedding?
- Reranked out?
- Below score threshold?
- Graph disconnected?
- Temporal penalty too strong?

Design: Purely additive — wraps existing retrieval output.
No LLM calls. All deterministic trace assembly.
"""
from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

LOGGER = logging.getLogger("openclaw.retrieval_explainer")

DB_DSN = os.environ.get("OPENCLAW_MEMORY_DSN", "dbname=openclaw_memory")


# ---------------------------------------------------------------------------
# Explanation structures
# ---------------------------------------------------------------------------

@dataclass
class ScoreContribution:
    """A single scoring component's contribution to the final rank."""
    component: str       # "vector", "fts", "rerank", "temporal", "feedback", "graph", "lifecycle"
    raw_score: float     # The component's own score
    weight: float        # Weight applied
    weighted_score: float  # raw * weight
    detail: str          # Human-readable explanation

    def to_dict(self) -> dict:
        return {
            "component": self.component,
            "raw_score": round(self.raw_score, 4),
            "weight": round(self.weight, 4),
            "weighted_score": round(self.weighted_score, 4),
            "detail": self.detail,
        }


@dataclass
class ProvenanceBundle:
    """Full provenance for a retrieved memory item."""
    item_id: str
    source_transcript: str = ""
    source_agent: str = ""
    source_session: str = ""
    source_chunk: str = ""
    created_at: str = ""
    last_confirmed: str = ""
    first_seen: str = ""
    lifecycle_state: str = ""
    is_current_truth: bool = True
    supersedes: str = ""
    superseded_by: list[str] = field(default_factory=list)
    supported_by: list[str] = field(default_factory=list)
    contradiction_count: int = 0
    contradictions: list[dict] = field(default_factory=list)
    confidence: float = 0.0
    mention_count: int = 0
    memory_type: str = ""
    entity: str = ""
    property: str = ""
    scope: str = ""

    def to_dict(self) -> dict:
        return {
            "item_id": self.item_id,
            "source_agent": self.source_agent,
            "source_session": self.source_session,
            "source_chunk": self.source_chunk,
            "created_at": self.created_at,
            "last_confirmed": self.last_confirmed,
            "lifecycle_state": self.lifecycle_state,
            "is_current_truth": self.is_current_truth,
            "supersedes": self.supersedes,
            "superseded_by": self.superseded_by,
            "contradiction_count": self.contradiction_count,
            "contradictions": self.contradictions,
            "confidence": round(self.confidence, 4),
            "memory_type": self.memory_type,
            "entity": self.entity,
            "scope": self.scope,
        }


@dataclass
class RetrievalExplanation:
    """Full explanation for why an item was retrieved and how it ranked."""
    item_id: str
    text: str
    final_rank: int
    final_score: float
    contributions: list[ScoreContribution] = field(default_factory=list)
    provenance: ProvenanceBundle | None = None
    included_reason: str = ""  # "top_k_cutoff", "graph_expansion", etc.

    def to_dict(self) -> dict:
        return {
            "item_id": self.item_id,
            "text": self.text[:300],
            "final_rank": self.final_rank,
            "final_score": round(self.final_score, 4),
            "contributions": [c.to_dict() for c in self.contributions],
            "provenance": self.provenance.to_dict() if self.provenance else None,
            "included_reason": self.included_reason,
        }


# ---------------------------------------------------------------------------
# Score decomposition
# ---------------------------------------------------------------------------

def decompose_scores(item: dict, query: str = "") -> list[ScoreContribution]:
    """
    Break down a retrieval result into its component score contributions.
    Works with results from both the search service and benchmark pipeline.
    """
    contributions = []

    # Vector similarity
    vec_score = float(item.get("vec_score", 0) or 0)
    vec_weight = float(item.get("vec_weight", 0.6) or 0.6)
    if vec_score > 0:
        contributions.append(ScoreContribution(
            component="vector",
            raw_score=vec_score,
            weight=vec_weight,
            weighted_score=vec_score * vec_weight,
            detail=f"Embedding cosine similarity: {vec_score:.4f}",
        ))

    # FTS/BM25
    fts_score = float(item.get("fts_score", 0) or 0)
    fts_weight = float(item.get("fts_weight", 0.4) or 0.4)
    if fts_score > 0:
        contributions.append(ScoreContribution(
            component="fts",
            raw_score=fts_score,
            weight=fts_weight,
            weighted_score=fts_score * fts_weight,
            detail=f"Full-text search rank: {fts_score:.4f}",
        ))

    # Hybrid base score
    hybrid_score = float(item.get("hybrid_score", 0) or 0)
    if hybrid_score > 0 and not (vec_score or fts_score):
        contributions.append(ScoreContribution(
            component="hybrid",
            raw_score=hybrid_score,
            weight=1.0,
            weighted_score=hybrid_score,
            detail=f"Pre-computed hybrid score: {hybrid_score:.4f}",
        ))

    # Reranker
    rerank_score = item.get("rerank_score")
    if rerank_score is not None:
        pre_rerank = hybrid_score or (vec_score * vec_weight + fts_score * fts_weight)
        rerank_delta = float(rerank_score) - pre_rerank if pre_rerank else 0
        contributions.append(ScoreContribution(
            component="rerank",
            raw_score=float(rerank_score),
            weight=1.0,
            weighted_score=float(rerank_score),
            detail=f"Cross-encoder rerank score: {rerank_score:.4f} (delta from hybrid: {rerank_delta:+.4f})",
        ))

    # Temporal boost
    temporal = float(item.get("temporal_boost", 0) or 0)
    if temporal != 0:
        contributions.append(ScoreContribution(
            component="temporal",
            raw_score=temporal,
            weight=1.0,
            weighted_score=temporal,
            detail=f"Temporal {'boost' if temporal > 0 else 'penalty'}: {temporal:+.4f}",
        ))

    # Feedback boost
    feedback = float(item.get("feedback_boost", 0) or item.get("ranking_bonus", 0) or 0)
    feedback_penalty = float(item.get("ranking_penalty", 0) or 0)
    net_feedback = feedback - feedback_penalty
    if net_feedback != 0:
        contributions.append(ScoreContribution(
            component="feedback",
            raw_score=net_feedback,
            weight=1.0,
            weighted_score=net_feedback,
            detail=f"Feedback {'boost' if net_feedback > 0 else 'penalty'}: {net_feedback:+.4f} (bonus={feedback:.3f} penalty={feedback_penalty:.3f})",
        ))

    # Source type (graph expansion, qdrant, db)
    source_type = item.get("source_type", "")
    if source_type == "graph":
        contributions.append(ScoreContribution(
            component="graph",
            raw_score=1.0,
            weight=0.0,
            weighted_score=0.0,
            detail="Retrieved via Neo4j graph expansion",
        ))
    elif source_type:
        contributions.append(ScoreContribution(
            component="source",
            raw_score=0.0,
            weight=0.0,
            weighted_score=0.0,
            detail=f"Retrieved from {source_type} source",
        ))

    # Lifecycle eligibility
    status = item.get("status", "")
    if status:
        is_eligible = status in ("active", "candidate")
        contributions.append(ScoreContribution(
            component="lifecycle",
            raw_score=1.0 if is_eligible else 0.0,
            weight=0.0,
            weighted_score=0.0,
            detail=f"Lifecycle state: {status} ({'eligible' if is_eligible else 'filtered'})",
        ))

    return contributions


# ---------------------------------------------------------------------------
# Provenance fetching
# ---------------------------------------------------------------------------

def fetch_provenance(item_id: str, conn: Any = None) -> ProvenanceBundle:
    """Fetch full provenance for a memory item from the database."""
    should_close = False
    if conn is None:
        import psycopg
        conn = psycopg.connect(DB_DSN)
        should_close = True

    bundle = ProvenanceBundle(item_id=item_id)

    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT text, source_agent, source_session, source_chunk,
                       created_at, last_confirmed, first_seen, status,
                       supersedes, confidence, memory_type, entity, property, scope,
                       COALESCE(contradiction_count, 0),
                       COALESCE(is_current_truth, TRUE)
                FROM memory_items WHERE id = %s
            """, (item_id,))

            row = cur.fetchone()
            if row:
                bundle.source_agent = row[1] or ""
                bundle.source_session = row[2] or ""
                bundle.source_chunk = row[3] or ""
                bundle.created_at = str(row[4] or "")
                bundle.last_confirmed = str(row[5] or "")
                bundle.first_seen = str(row[6] or "")
                bundle.lifecycle_state = row[7] or ""
                bundle.supersedes = row[8] or ""
                bundle.confidence = float(row[9] or 0)
                bundle.memory_type = row[10] or ""
                bundle.entity = row[11] or ""
                bundle.property = row[12] or ""
                bundle.scope = row[13] or ""
                bundle.contradiction_count = int(row[14] or 0)
                bundle.is_current_truth = bool(row[15])

            # Find items that supersede this one
            cur.execute("""
                SELECT id FROM memory_items WHERE supersedes = %s AND status = 'active'
            """, (item_id,))
            bundle.superseded_by = [r[0] for r in cur.fetchall()]

            # Find active contradictions
            try:
                cur.execute("""
                    SELECT ce.edge_id, ce.item_a_id, ce.item_b_id,
                           ce.detection_method, ce.confidence, ce.resolution_state
                    FROM contradiction_edges ce
                    WHERE (ce.item_a_id = %s OR ce.item_b_id = %s)
                      AND ce.resolution_state = 'unresolved'
                """, (item_id, item_id))

                for edge_row in cur.fetchall():
                    other_id = edge_row[2] if edge_row[1] == item_id else edge_row[1]
                    bundle.contradictions.append({
                        "edge_id": edge_row[0],
                        "contradicts": other_id,
                        "method": edge_row[3],
                        "confidence": float(edge_row[4]),
                    })
            except Exception:
                pass  # contradiction_edges table may not exist yet

    finally:
        if should_close:
            conn.close()

    return bundle


def fetch_provenance_batch(item_ids: list[str], conn: Any = None) -> dict[str, ProvenanceBundle]:
    """Fetch provenance for multiple items efficiently."""
    if not item_ids:
        return {}

    should_close = False
    if conn is None:
        import psycopg
        conn = psycopg.connect(DB_DSN)
        should_close = True

    bundles = {}
    try:
        with conn.cursor() as cur:
            placeholders = ",".join(["%s"] * len(item_ids))
            cur.execute(f"""
                SELECT id, text, source_agent, source_session, source_chunk,
                       created_at, last_confirmed, first_seen, status,
                       supersedes, confidence, memory_type, entity, property, scope,
                       COALESCE(contradiction_count, 0),
                       COALESCE(is_current_truth, TRUE)
                FROM memory_items WHERE id IN ({placeholders})
            """, item_ids)

            for row in cur.fetchall():
                iid = row[0]
                b = ProvenanceBundle(item_id=iid)
                b.source_agent = row[2] or ""
                b.source_session = row[3] or ""
                b.source_chunk = row[4] or ""
                b.created_at = str(row[5] or "")
                b.last_confirmed = str(row[6] or "")
                b.first_seen = str(row[7] or "")
                b.lifecycle_state = row[8] or ""
                b.supersedes = row[9] or ""
                b.confidence = float(row[10] or 0)
                b.memory_type = row[11] or ""
                b.entity = row[12] or ""
                b.property = row[13] or ""
                b.scope = row[14] or ""
                b.contradiction_count = int(row[15] or 0)
                b.is_current_truth = bool(row[16])
                bundles[iid] = b

    finally:
        if should_close:
            conn.close()

    return bundles


# ---------------------------------------------------------------------------
# "Why not retrieved?" diagnostics
# ---------------------------------------------------------------------------

@dataclass
class WhyNotResult:
    """Diagnosis for why an expected item didn't appear in results."""
    item_id: str
    found_in_db: bool = False
    status: str = ""
    has_embedding: bool = False
    lifecycle_eligible: bool = False
    was_in_candidates: bool = False
    candidate_rank: int | None = None
    candidate_score: float | None = None
    reranked_out: bool = False
    rerank_score: float | None = None
    temporal_penalty: float = 0.0
    feedback_penalty: float = 0.0
    below_threshold: bool = False
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "item_id": self.item_id,
            "found_in_db": self.found_in_db,
            "status": self.status,
            "has_embedding": self.has_embedding,
            "lifecycle_eligible": self.lifecycle_eligible,
            "was_in_candidates": self.was_in_candidates,
            "candidate_rank": self.candidate_rank,
            "candidate_score": self.candidate_score,
            "reranked_out": self.reranked_out,
            "rerank_score": self.rerank_score,
            "temporal_penalty": round(self.temporal_penalty, 4),
            "feedback_penalty": round(self.feedback_penalty, 4),
            "below_threshold": self.below_threshold,
            "reasons": self.reasons,
            "diagnosis": self._diagnose(),
        }

    def _diagnose(self) -> str:
        """Generate a human-readable diagnosis."""
        if not self.found_in_db:
            return "Item does not exist in memory_items table — was it never extracted?"
        if not self.lifecycle_eligible:
            return f"Item filtered by lifecycle: status='{self.status}' (only 'active'/'candidate' are eligible)"
        if not self.has_embedding:
            return "Item has no embedding vector — it cannot be found by vector search"
        if self.reranked_out:
            return f"Item was in initial candidates (rank {self.candidate_rank}, score {self.candidate_score:.4f}) but reranked out (rerank score: {self.rerank_score:.4f})"
        if self.below_threshold:
            return f"Item scored below the top-K cutoff (rank {self.candidate_rank}, score {self.candidate_score:.4f})"
        if self.temporal_penalty < -0.05:
            return f"Item was penalized by temporal scoring: {self.temporal_penalty:+.4f}"
        if self.feedback_penalty < -0.05:
            return f"Item was penalized by negative feedback: {self.feedback_penalty:+.4f}"
        if self.was_in_candidates:
            return f"Item was in candidates (rank {self.candidate_rank}) but didn't make final top-K"
        return "Item exists and is eligible but wasn't matched by any search source (vector similarity too low, no keyword overlap)"


def diagnose_why_not(
    item_id: str,
    query: str,
    retrieved_ids: list[str],
    candidate_pool: list[dict] | None = None,
    conn: Any = None,
) -> WhyNotResult:
    """
    Diagnose why a specific item wasn't in the retrieval results.

    Args:
        item_id: the expected item
        query: the search query
        retrieved_ids: IDs that were returned
        candidate_pool: if available, the full pre-top-K candidate list
        conn: DB connection
    """
    result = WhyNotResult(item_id=item_id)

    if item_id in retrieved_ids:
        result.reasons.append("item_was_actually_retrieved")
        result.found_in_db = True
        result.was_in_candidates = True
        return result

    should_close = False
    if conn is None:
        import psycopg
        conn = psycopg.connect(DB_DSN)
        should_close = True

    try:
        with conn.cursor() as cur:
            # Check if item exists
            cur.execute("""
                SELECT id, status, text, confidence,
                       COALESCE(ranking_bonus, 0), COALESCE(ranking_penalty, 0)
                FROM memory_items WHERE id = %s
            """, (item_id,))
            row = cur.fetchone()

            if not row:
                result.found_in_db = False
                result.reasons.append("not_in_database")
                return result

            result.found_in_db = True
            result.status = row[1]
            result.lifecycle_eligible = row[1] in ("active", "candidate")
            result.feedback_penalty = -float(row[5] or 0)

            if not result.lifecycle_eligible:
                result.reasons.append(f"lifecycle_filtered (status={row[1]})")
                return result

            # Check for embedding (Qdrant or inline)
            try:
                cur.execute("""
                    SELECT EXISTS (
                        SELECT 1 FROM information_schema.columns
                        WHERE table_name = 'memory_items' AND column_name = 'embedding'
                    )
                """)
                has_emb_col = cur.fetchone()[0]

                if has_emb_col:
                    cur.execute("SELECT embedding IS NOT NULL FROM memory_items WHERE id = %s", (item_id,))
                    emb_row = cur.fetchone()
                    result.has_embedding = bool(emb_row and emb_row[0])
                else:
                    result.has_embedding = True  # External embedding store (Qdrant)
            except Exception:
                result.has_embedding = True  # Assume yes if we can't check

            if not result.has_embedding:
                result.reasons.append("missing_embedding")

            # Check if it was in the candidate pool
            if candidate_pool:
                for i, c in enumerate(candidate_pool):
                    cid = c.get("id", c.get("item_id", ""))
                    if cid == item_id:
                        result.was_in_candidates = True
                        result.candidate_rank = i + 1
                        result.candidate_score = float(c.get("hybrid_score", 0) or c.get("score", 0) or 0)
                        result.rerank_score = c.get("rerank_score")

                        if result.rerank_score is not None and result.candidate_score:
                            if float(result.rerank_score) < result.candidate_score * 0.5:
                                result.reranked_out = True
                                result.reasons.append("reranked_out")

                        result.below_threshold = True
                        result.reasons.append("below_top_k_cutoff")
                        break

                if not result.was_in_candidates:
                    result.reasons.append("not_in_candidate_pool")
            else:
                result.reasons.append("candidate_pool_not_available")

    finally:
        if should_close:
            conn.close()

    return result


# ---------------------------------------------------------------------------
# Full explanation builder
# ---------------------------------------------------------------------------

def explain_retrieval(
    query: str,
    results: list[dict],
    conn: Any = None,
    include_provenance: bool = True,
    candidate_pool: list[dict] | None = None,
) -> list[RetrievalExplanation]:
    """
    Build full explanations for a set of retrieval results.

    Args:
        query: the search query
        results: list of result dicts from the search pipeline
        conn: DB connection
        include_provenance: whether to fetch provenance from DB
        candidate_pool: full pre-top-K candidates (for "why not" diagnostics)

    Returns:
        List of RetrievalExplanation objects, one per result.
    """
    explanations = []

    # Batch fetch provenance if requested
    provenance_map: dict[str, ProvenanceBundle] = {}
    if include_provenance:
        item_ids = [str(r.get("id", r.get("item_id", ""))) for r in results if r.get("id") or r.get("item_id")]
        if item_ids:
            provenance_map = fetch_provenance_batch(item_ids, conn)

    for rank, item in enumerate(results, 1):
        item_id = str(item.get("id", item.get("item_id", f"rank-{rank}")))
        text = str(item.get("text", ""))[:300]

        contributions = decompose_scores(item, query)
        provenance = provenance_map.get(item_id)

        final_score = float(
            item.get("rerank_score")
            or item.get("hybrid_score")
            or item.get("score")
            or 0
        )

        explanation = RetrievalExplanation(
            item_id=item_id,
            text=text,
            final_rank=rank,
            final_score=final_score,
            contributions=contributions,
            provenance=provenance,
            included_reason="top_k" if rank <= 10 else "extended",
        )
        explanations.append(explanation)

    return explanations


def explain_why_not(
    query: str,
    expected_ids: list[str],
    retrieved_ids: list[str],
    candidate_pool: list[dict] | None = None,
    conn: Any = None,
) -> list[WhyNotResult]:
    """
    Diagnose why expected items didn't appear in results.

    Args:
        query: the search query
        expected_ids: item IDs that should have been retrieved
        retrieved_ids: item IDs that were actually returned
        candidate_pool: full pre-top-K candidates
        conn: DB connection

    Returns:
        List of WhyNotResult diagnostics.
    """
    missing = [eid for eid in expected_ids if eid not in retrieved_ids]
    return [
        diagnose_why_not(eid, query, retrieved_ids, candidate_pool, conn)
        for eid in missing
    ]
