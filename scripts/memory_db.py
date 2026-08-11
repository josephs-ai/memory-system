"""
PostgreSQL database operations for memory items — CRUD, feedback tracking,
retrieval logging, and schema management. Core data layer for the memory system.
"""
import atexit
import hashlib
import os
from pathlib import Path

from psycopg.types.json import Jsonb
from psycopg_pool import ConnectionPool

WORKSPACE = Path.home() / ".openclaw" / "workspace"
DEFAULT_DSN = os.environ.get("OPENCLAW_MEMORY_DB_DSN", "dbname=openclaw_memory")

POOL = ConnectionPool(
    conninfo=DEFAULT_DSN,
    min_size=1,
    max_size=8,
    kwargs={"autocommit": False},
)

MEMORY_ITEM_COLUMNS = [
    "id",
    "text",
    "memory_type",
    "scope",
    "project_id",
    "subproject_id",
    "workflow_id",
    "pipeline_id",
    "context_scope_id",
    "context_scope_type",
    "context_scope_payload",
    "inheritance_policy",
    "scope_confidence",
    "entity",
    "property",
    "value",
    "status",
    "confidence",
    "importance",
    "freshness_class",
    "source_agent",
    "source_session",
    "source_chunk",
    "first_seen",
    "last_confirmed",
    "supersedes",
    "tags",
    "notes",
    "candidate_id",
    "candidate_score",
    "candidate_reasons",
    "suggested_route",
    "target_file",
    "target_section",
    "sensitivity",
    "approved_at",
    "approved_by",
    "approval_source",
    "rejected_at",
    "rejected_by",
    "rejection_reason",
    "rejection_source",
    "ranking_bonus",
    "ranking_penalty",
    "feedback_last_applied_at",
]

JSONB_COLUMNS = {"tags", "candidate_reasons", "context_scope_payload"}

UPSERT_SQL = f"""
INSERT INTO memory_items (
    {", ".join(MEMORY_ITEM_COLUMNS)}
) VALUES (
    {", ".join(f"%({c})s" for c in MEMORY_ITEM_COLUMNS)}
)
ON CONFLICT (id) DO UPDATE SET
    text = EXCLUDED.text,
    memory_type = EXCLUDED.memory_type,
    scope = EXCLUDED.scope,
    project_id = EXCLUDED.project_id,
    subproject_id = EXCLUDED.subproject_id,
    workflow_id = EXCLUDED.workflow_id,
    pipeline_id = EXCLUDED.pipeline_id,
    context_scope_id = EXCLUDED.context_scope_id,
    context_scope_type = EXCLUDED.context_scope_type,
    context_scope_payload = EXCLUDED.context_scope_payload,
    inheritance_policy = EXCLUDED.inheritance_policy,
    scope_confidence = EXCLUDED.scope_confidence,
    entity = EXCLUDED.entity,
    property = EXCLUDED.property,
    value = EXCLUDED.value,
    status = EXCLUDED.status,
    confidence = EXCLUDED.confidence,
    importance = EXCLUDED.importance,
    freshness_class = EXCLUDED.freshness_class,
    source_agent = EXCLUDED.source_agent,
    source_session = EXCLUDED.source_session,
    source_chunk = EXCLUDED.source_chunk,
    first_seen = COALESCE(memory_items.first_seen, EXCLUDED.first_seen),
    last_confirmed = EXCLUDED.last_confirmed,
    supersedes = EXCLUDED.supersedes,
    tags = EXCLUDED.tags,
    notes = EXCLUDED.notes,
    candidate_id = EXCLUDED.candidate_id,
    candidate_score = EXCLUDED.candidate_score,
    candidate_reasons = EXCLUDED.candidate_reasons,
    suggested_route = EXCLUDED.suggested_route,
    target_file = EXCLUDED.target_file,
    target_section = EXCLUDED.target_section,
    sensitivity = EXCLUDED.sensitivity,
    approved_at = EXCLUDED.approved_at,
    approved_by = EXCLUDED.approved_by,
    approval_source = EXCLUDED.approval_source,
    rejected_at = EXCLUDED.rejected_at,
    rejected_by = EXCLUDED.rejected_by,
    rejection_reason = EXCLUDED.rejection_reason,
    rejection_source = EXCLUDED.rejection_source,
    ranking_bonus = EXCLUDED.ranking_bonus,
    ranking_penalty = EXCLUDED.ranking_penalty,
    feedback_last_applied_at = EXCLUDED.feedback_last_applied_at,
    updated_at = now()
"""


def normalize_memory_item(item: dict) -> dict:
    row = {}
    for col in MEMORY_ITEM_COLUMNS:
        value = item.get(col)

        if col in JSONB_COLUMNS:
            if value is None:
                value = [] if col in {"tags", "candidate_reasons"} else {}
            value = Jsonb(value)

        elif col == "importance":
            raw = item.get("importance", item.get("retention_priority", value))
            try:
                value = float(raw) if raw is not None and raw != "" else 0.0
            except (TypeError, ValueError):
                value = 0.0

        elif col == "confidence":
            try:
                value = float(value) if value is not None and value != "" else 0.0
            except (TypeError, ValueError):
                value = 0.0

        elif col == "status":
            value = value or "active"

        elif col == "sensitivity":
            value = value or "public"

        elif col == "ranking_bonus":
            value = 0 if value is None else value

        elif col == "ranking_penalty":
            value = 0 if value is None else value

        row[col] = value

    return row


def get_conn():
    return POOL.connection()


def upsert_memory_item(item: dict):
    row = normalize_memory_item(item)
    with POOL.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(UPSERT_SQL, row)
        conn.commit()


def upsert_memory_items(items: list[dict]):
    rows = [normalize_memory_item(x) for x in items]
    if not rows:
        return

    with POOL.connection() as conn:
        with conn.cursor() as cur:
            cur.executemany(UPSERT_SQL, rows)
        conn.commit()

def upsert_pending_stable(items: list[dict]):
    if not items:
        return
    with POOL.connection() as conn:
        with conn.cursor() as cur:
            cur.executemany(
                """
                INSERT INTO memory_pending_stable (id, payload)
                VALUES (%s, %s)
                ON CONFLICT (id) DO UPDATE SET
                    payload = EXCLUDED.payload
                """,
                [(item.get("id"), Jsonb(item)) for item in items],
            )
        conn.commit()


def upsert_inbox(items: list[dict]):
    if not items:
        return
    with POOL.connection() as conn:
        with conn.cursor() as cur:
            cur.executemany(
                """
                INSERT INTO memory_inbox (id, payload)
                VALUES (%s, %s)
                ON CONFLICT (id) DO UPDATE SET
                    payload = EXCLUDED.payload
                """,
                [(item.get("id"), Jsonb(item)) for item in items],
            )
        conn.commit()


def upsert_discarded(items: list[dict]):
    if not items:
        return
    with POOL.connection() as conn:
        with conn.cursor() as cur:
            cur.executemany(
                """
                INSERT INTO memory_discarded (id, payload)
                VALUES (%s, %s)
                ON CONFLICT (id) DO UPDATE SET
                    payload = EXCLUDED.payload
                """,
                [(item.get("id"), Jsonb(item)) for item in items],
            )
        conn.commit()

def insert_retrieval_feedback(rows: list[dict]):
    if not rows:
        return
    with POOL.connection() as conn:
        with conn.cursor() as cur:
            cur.executemany(
                """
                INSERT INTO retrieval_feedback (
                    selected_id,
                    selected_score,
                    selected_status,
                    operator_feedback,
                    feedback_note,
                    feedback_at,
                    candidate_file,
                    target_file,
                    include_superseded,
                    payload
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                [
                    (
                        row.get("selected_id"),
                        row.get("selected_score"),
                        row.get("selected_status"),
                        row.get("operator_feedback"),
                        row.get("feedback_note"),
                        row.get("feedback_at"),
                        row.get("candidate_file"),
                        row.get("target_file"),
                        row.get("include_superseded"),
                        Jsonb(row),
                    )
                    for row in rows
                ],
            )
        conn.commit()


def upsert_global_insights(rows: list[dict]):
    if not rows:
        return
    with POOL.connection() as conn:
        with conn.cursor() as cur:
            cur.executemany(
                """
                INSERT INTO global_insights (
                    topic,
                    text,
                    source_project,
                    promoted_by
                ) VALUES (%s, %s, %s, %s)
                ON CONFLICT (topic, text) DO UPDATE SET
                    source_project = EXCLUDED.source_project,
                    promoted_by = EXCLUDED.promoted_by
                """,
                [
                    (
                        row.get("topic"),
                        row.get("text"),
                        row.get("source_project"),
                        row.get("promoted_by"),
                    )
                    for row in rows
                ],
            )
        conn.commit()

def fetch_canonical_items(status: str = "active"):
    with POOL.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    id, text, memory_type, scope, entity, property, value, status,
                    confidence, importance, freshness_class,
                    source_agent, source_session, source_chunk,
                    first_seen, last_confirmed, supersedes,
                    tags, notes, candidate_id, candidate_score, candidate_reasons,
                    suggested_route, target_file, target_section,
                    sensitivity, approved_at, approved_by, approval_source,
                    rejected_at, rejected_by, rejection_reason, rejection_source,
                    ranking_bonus, ranking_penalty, feedback_last_applied_at
                FROM memory_items
                WHERE status = %s
                ORDER BY id
                """,
                (status,),
            )
            cols = [d[0] for d in cur.description]
            rows = []
            for rec in cur.fetchall():
                row = dict(zip(cols, rec))
                row["tags"] = row.get("tags") or []
                row["candidate_reasons"] = row.get("candidate_reasons") or []
                rows.append(row)
            return rows


def search_canonical_items_simple(query: str, status: str = "active", limit: int = 20):
    q = f"%{query.lower()}%"
    with POOL.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    id, text, memory_type, scope, entity, property, value, status,
                    confidence, importance, freshness_class,
                    source_agent, source_session, source_chunk,
                    first_seen, last_confirmed, supersedes,
                    tags, notes, candidate_id, candidate_score, candidate_reasons,
                    suggested_route, target_file, target_section,
                    sensitivity, approved_at, approved_by, approval_source,
                    rejected_at, rejected_by, rejection_reason, rejection_source,
                    ranking_bonus, ranking_penalty, feedback_last_applied_at
                FROM memory_items
                WHERE status = %s
                  AND LOWER(text) LIKE %s
                ORDER BY confidence DESC NULLS LAST, importance DESC NULLS LAST, id
                LIMIT %s
                """,
                (status, q, limit),
            )
            cols = [d[0] for d in cur.description]
            rows = []
            for rec in cur.fetchall():
                row = dict(zip(cols, rec))
                row["tags"] = row.get("tags") or []
                row["candidate_reasons"] = row.get("candidate_reasons") or []
                rows.append(row)
            return rows

def fetch_pending_stable_item(item_id: str):
    with POOL.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT payload FROM memory_pending_stable WHERE id = %s",
                (item_id,),
            )
            row = cur.fetchone()
            return row[0] if row else None


def delete_pending_stable_item(item_id: str):
    with POOL.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM memory_pending_stable WHERE id = %s",
                (item_id,),
            )
        conn.commit()


def insert_discarded_item(item: dict):
    with POOL.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO memory_discarded (id, payload)
                VALUES (%s, %s)
                ON CONFLICT (id) DO UPDATE SET
                    payload = EXCLUDED.payload
                """,
                (item.get("id"), Jsonb(item)),
            )
        conn.commit()

def fetch_queue_ids(table_name: str):
    allowed = {
        "memory_pending_stable",
        "memory_inbox",
        "memory_discarded",
    }
    if table_name not in allowed:
        raise ValueError(f"unsupported table: {table_name}")

    with POOL.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(f"SELECT id FROM {table_name}")
            return {row[0] for row in cur.fetchall()}


def queue_item_exists_anywhere(item_id: str):
    with POOL.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT EXISTS(SELECT 1 FROM memory_pending_stable WHERE id = %s)
                    OR EXISTS(SELECT 1 FROM memory_inbox WHERE id = %s)
                    OR EXISTS(SELECT 1 FROM memory_discarded WHERE id = %s)
                    OR EXISTS(SELECT 1 FROM memory_items WHERE id = %s)
                """,
                (item_id, item_id, item_id, item_id),
            )
            return bool(cur.fetchone()[0])

def fetch_memory_items_by_status(status: str):
    with POOL.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    id, text, memory_type, scope, entity, property, value, status,
                    confidence, importance, freshness_class,
                    source_agent, source_session, source_chunk,
                    first_seen, last_confirmed, supersedes,
                    tags, notes, candidate_id, candidate_score, candidate_reasons,
                    suggested_route, target_file, target_section,
                    sensitivity, approved_at, approved_by, approval_source,
                    rejected_at, rejected_by, rejection_reason, rejection_source,
                    ranking_bonus, ranking_penalty, feedback_last_applied_at
                FROM memory_items
                WHERE status = %s
                ORDER BY id
                """,
                (status,),
            )
            cols = [d[0] for d in cur.description]
            rows = []
            for rec in cur.fetchall():
                row = dict(zip(cols, rec))
                row["tags"] = row.get("tags") or []
                row["candidate_reasons"] = row.get("candidate_reasons") or []
                rows.append(row)
            return rows

def fetch_memory_items(statuses: list[str] | None = None):
    with POOL.connection() as conn:
        with conn.cursor() as cur:
            if statuses:
                cur.execute(
                    """
                    SELECT
                        id, text, memory_type, scope, entity, property, value, status,
                        confidence, importance, freshness_class,
                        source_agent, source_session, source_chunk,
                        first_seen, last_confirmed, supersedes,
                        tags, notes, candidate_id, candidate_score, candidate_reasons,
                        suggested_route, target_file, target_section,
                        sensitivity, approved_at, approved_by, approval_source,
                        rejected_at, rejected_by, rejection_reason, rejection_source,
                        ranking_bonus, ranking_penalty, feedback_last_applied_at
                    FROM memory_items
                    WHERE status = ANY(%s)
                    ORDER BY id
                    """,
                    (statuses,),
                )
            else:
                cur.execute(
                    """
                    SELECT
                        id, text, memory_type, scope, entity, property, value, status,
                        confidence, importance, freshness_class,
                        source_agent, source_session, source_chunk,
                        first_seen, last_confirmed, supersedes,
                        tags, notes, candidate_id, candidate_score, candidate_reasons,
                        suggested_route, target_file, target_section,
                        sensitivity, approved_at, approved_by, approval_source,
                        rejected_at, rejected_by, rejection_reason, rejection_source,
                        ranking_bonus, ranking_penalty, feedback_last_applied_at
                    FROM memory_items
                    ORDER BY id
                    """
                )

            cols = [d[0] for d in cur.description]
            rows = []
            for rec in cur.fetchall():
                row = dict(zip(cols, rec))
                row["tags"] = row.get("tags") or []
                row["candidate_reasons"] = row.get("candidate_reasons") or []
                rows.append(row)
            return rows

def feedback_stats_db(item_id: str):
    with POOL.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    COALESCE(SUM(CASE WHEN operator_feedback = 'useful' THEN 1 ELSE 0 END), 0),
                    COALESCE(SUM(CASE WHEN operator_feedback = 'bad' THEN 1 ELSE 0 END), 0)
                FROM retrieval_feedback
                WHERE selected_id = %s
                """,
                (item_id,),
            )
            useful, bad = cur.fetchone()
            return int(useful), int(bad)

def insert_retrieval_feedback_row(row: dict):
    with POOL.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO retrieval_feedback (
                    selected_id,
                    selected_score,
                    selected_status,
                    operator_feedback,
                    feedback_note,
                    feedback_at,
                    candidate_file,
                    target_file,
                    include_superseded,
                    query_text,
                    query_type,
                    payload
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    row.get("selected_id"),
                    row.get("selected_score"),
                    row.get("selected_status"),
                    row.get("operator_feedback"),
                    row.get("feedback_note"),
                    row.get("feedback_at"),
                    row.get("candidate_file"),
                    row.get("target_file"),
                    row.get("include_superseded"),
                    row.get("query_text"),
                    row.get("query_type"),
                    Jsonb(row),
                ),
            )
        conn.commit()

def update_memory_item_ranking(item_id: str, *, ranking_bonus=None, ranking_penalty=None, feedback_last_applied_at=None):
    sets = []
    params = []

    if ranking_bonus is not None:
        sets.append("ranking_bonus = %s")
        params.append(ranking_bonus)

    if ranking_penalty is not None:
        sets.append("ranking_penalty = %s")
        params.append(ranking_penalty)

    if feedback_last_applied_at is not None:
        sets.append("feedback_last_applied_at = %s")
        params.append(feedback_last_applied_at)

    if not sets:
        return

    params.append(item_id)

    with POOL.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                UPDATE memory_items
                SET {", ".join(sets)}, updated_at = now()
                WHERE id = %s
                """,
                params,
            )
        conn.commit()


def fetch_memory_status_map():
    rows = fetch_memory_items()
    out = {}
    for row in rows:
        out[row["id"]] = (row.get("status", "unknown"), row)
    return out


def all_feedback_item_ids():
    with POOL.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT DISTINCT selected_id
                FROM retrieval_feedback
                WHERE selected_id IS NOT NULL
                """
            )
            return [row[0] for row in cur.fetchall()]

def fetch_discarded_payloads():
    """Load every discarded payload. Only safe on small tables -- see fetch_rejected_matches."""
    with POOL.connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT payload FROM memory_discarded")
            return [row[0] for row in cur.fetchall()]


def fetch_rejected_matches(items: list[dict]) -> tuple[set[str], set[tuple]]:
    """
    Return the discarded text/slot keys that match `items`, and nothing else.

    memory_discarded grows without bound (millions of rows / GBs), so pulling the
    whole table into Python to dedupe a handful of candidates is what made the
    checkpoint pipeline allocate ~8GB and get killed on timeout. Probe only the
    keys we actually care about.

    NOTE: this filters in Postgres rather than in Python, which is the fix that
    matters, but there is no index backing either predicate yet -- neither
    memory_db_schema.sql nor any migration creates one on md5(lower(trim(text)))
    or on (entity, property, value). These are sequential scans with a cheap
    filter. Adding those two functional indexes is the remaining win; the
    trim() above is written to match an index built the same way.
    """
    texts = sorted({t for t in ((i.get("text") or "").strip().lower() for i in items) if t})
    slots = sorted(
        {
            (i["entity"], i["property"], i["value"])
            for i in items
            if i.get("entity") and i.get("property") and i.get("value")
        }
    )

    text_set: set[str] = set()
    slot_set: set[tuple] = set()
    if not texts and not slots:
        return text_set, slot_set

    with POOL.connection() as conn:
        with conn.cursor() as cur:
            if texts:
                digests = [hashlib.md5(t.encode("utf-8")).hexdigest() for t in texts]
                cur.execute(
                    """
                    SELECT lower(trim(payload->>'text'))
                    FROM memory_discarded
                    WHERE md5(lower(trim(payload->>'text'))) = ANY(%s)
                    """,
                    (digests,),
                )
                # Re-check in Python: md5 collisions are absurdly unlikely, but the
                # index is on the digest, so the digest match alone is not proof.
                wanted = set(texts)
                text_set = {row[0] for row in cur.fetchall() if row[0] in wanted}

            if slots:
                cur.execute(
                    """
                    SELECT payload->>'entity', payload->>'property',
                           payload->>'value', payload->>'scope'
                    FROM memory_discarded
                    WHERE (payload->>'entity', payload->>'property', payload->>'value')
                          IN (SELECT * FROM unnest(%s::text[], %s::text[], %s::text[]))
                    """,
                    ([s[0] for s in slots], [s[1] for s in slots], [s[2] for s in slots]),
                )
                for entity, prop, value, scope in cur.fetchall():
                    slot_set.add((entity, prop, value))
                    if scope is not None:
                        slot_set.add((entity, prop, value, scope))

    return text_set, slot_set


def candidate_matches_rejected(item: dict) -> tuple[bool, dict | None]:
    """
    Single-item variant of fetch_rejected_matches. Probes by index rather than
    scanning memory_discarded (millions of rows) once per candidate.

    Match is on trim()+lower() of the stored text, mirroring the .strip().lower()
    applied to the candidate. Comparing a stripped candidate against an unstripped
    stored value silently dropped any row saved with a trailing newline, so a
    previously-rejected item could re-route to AUTO/INBOX. An exact lowercase text match, or an exact
    (entity, property, value) slot match, counts as previously rejected. The old
    `same_scope_slot` term was strictly narrower than `exact_slot` -- it could
    only be true when exact_slot already was -- so it never changed the result.
    """
    text = (item.get("text") or "").strip().lower()
    entity = item.get("entity")
    prop = item.get("property")
    value = item.get("value")

    with POOL.connection() as conn:
        with conn.cursor() as cur:
            if text:
                cur.execute(
                    """
                    SELECT payload FROM memory_discarded
                    WHERE md5(lower(trim(payload->>'text'))) = %s
                      AND lower(trim(payload->>'text')) = %s
                    LIMIT 1
                    """,
                    (hashlib.md5(text.encode("utf-8")).hexdigest(), text),
                )
                row = cur.fetchone()
                if row:
                    return True, row[0]

            if entity and prop and value:
                cur.execute(
                    """
                    SELECT payload FROM memory_discarded
                    WHERE payload->>'entity' = %s
                      AND payload->>'property' = %s
                      AND payload->>'value' = %s
                    LIMIT 1
                    """,
                    (entity, prop, value),
                )
                row = cur.fetchone()
                if row:
                    return True, row[0]

    return False, None
def low_utility_feedback_rows(min_bad: int = 1):
    with POOL.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    selected_id,
                    COALESCE(SUM(CASE WHEN operator_feedback = 'useful' THEN 1 ELSE 0 END), 0) AS useful_count,
                    COALESCE(SUM(CASE WHEN operator_feedback = 'bad' THEN 1 ELSE 0 END), 0) AS bad_count
                FROM retrieval_feedback
                WHERE selected_id IS NOT NULL
                GROUP BY selected_id
                HAVING COALESCE(SUM(CASE WHEN operator_feedback = 'bad' THEN 1 ELSE 0 END), 0) >= %s
                ORDER BY bad_count DESC, useful_count ASC, selected_id
                """,
                (min_bad,),
            )
            return [
                {
                    "selected_id": row[0],
                    "useful": int(row[1]),
                    "bad": int(row[2]),
                }
                for row in cur.fetchall()
            ]


def search_memory_items_by_terms(query: str, *, status: str = "active", limit: int = 50):
    query = (query or "").strip()
    if not query:
        return []

    sql = """
        SELECT
            id, text, memory_type, lifecycle_state, scope, entity, property, value, status,
            confidence, importance, freshness_class,
            source_agent, source_session, source_chunk,
            first_seen, last_confirmed, supersedes,
            tags, notes, candidate_id, candidate_score, candidate_reasons,
            suggested_route, target_file, target_section,
            sensitivity, approved_at, approved_by, approval_source,
            rejected_at, rejected_by, rejection_reason, rejection_source,
            ranking_bonus, ranking_penalty, feedback_last_applied_at,
            ts_rank_cd(
                to_tsvector(
                    'english',
                    COALESCE(text, '') || ' ' ||
                    COALESCE(entity, '') || ' ' ||
                    COALESCE(property, '') || ' ' ||
                    COALESCE(value, '') || ' ' ||
                    COALESCE(memory_type, '') || ' ' ||
                    COALESCE(scope, '')
                ),
                websearch_to_tsquery('english', %s)
            ) AS fts_rank
        FROM memory_items
        WHERE status = %s
          AND to_tsvector(
                'english',
                COALESCE(text, '') || ' ' ||
                COALESCE(entity, '') || ' ' ||
                COALESCE(property, '') || ' ' ||
                COALESCE(value, '') || ' ' ||
                COALESCE(memory_type, '') || ' ' ||
                COALESCE(scope, '')
              ) @@ websearch_to_tsquery('english', %s)
        ORDER BY
            fts_rank DESC,
            confidence DESC NULLS LAST,
            importance DESC NULLS LAST,
            last_confirmed DESC NULLS LAST,
            id
        LIMIT %s
    """

    with POOL.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (query, status, query, limit))
            cols = [d[0] for d in cur.description]
            rows = []
            for rec in cur.fetchall():
                row = dict(zip(cols, rec))
                row["tags"] = row.get("tags") or []
                row["candidate_reasons"] = row.get("candidate_reasons") or []
                rows.append(row)
            return rows

def search_memory_items_by_terms_rbac(
    query: str,
    *,
    status: str = "active",
    allowed_sensitivities: list[str] | None = None,
    limit: int = 50,
):
    query = (query or "").strip()
    if not query:
        return []

    allowed_sensitivities = list(allowed_sensitivities or ["public"])

    sql = """
        SELECT
            id, text, memory_type, lifecycle_state, scope, entity, property, value, status,
            confidence, importance, freshness_class,
            source_agent, source_session, source_chunk,
            first_seen, last_confirmed, supersedes,
            tags, notes, candidate_id, candidate_score, candidate_reasons,
            suggested_route, target_file, target_section,
            sensitivity, approved_at, approved_by, approval_source,
            rejected_at, rejected_by, rejection_reason, rejection_source,
            ranking_bonus, ranking_penalty, feedback_last_applied_at,
            ts_rank_cd(
                to_tsvector(
                    'english',
                    COALESCE(text, '') || ' ' ||
                    COALESCE(entity, '') || ' ' ||
                    COALESCE(property, '') || ' ' ||
                    COALESCE(value, '') || ' ' ||
                    COALESCE(memory_type, '') || ' ' ||
                    COALESCE(scope, '')
                ),
                websearch_to_tsquery('english', %s)
            ) AS fts_rank
        FROM memory_items
        WHERE status = %s
          AND sensitivity = ANY(%s)
          AND to_tsvector(
                'english',
                COALESCE(text, '') || ' ' ||
                COALESCE(entity, '') || ' ' ||
                COALESCE(property, '') || ' ' ||
                COALESCE(value, '') || ' ' ||
                COALESCE(memory_type, '') || ' ' ||
                COALESCE(scope, '')
              ) @@ websearch_to_tsquery('english', %s)
        ORDER BY
            fts_rank DESC,
            confidence DESC NULLS LAST,
            importance DESC NULLS LAST,
            last_confirmed DESC NULLS LAST,
            id
        LIMIT %s
    """

    with POOL.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (query, status, allowed_sensitivities, query, limit))
            cols = [d[0] for d in cur.description]
            rows = []
            for rec in cur.fetchall():
                row = dict(zip(cols, rec))
                row["tags"] = row.get("tags") or []
                row["candidate_reasons"] = row.get("candidate_reasons") or []
                rows.append(row)
            return rows

def upsert_memory_embedding(memory_id: str, model_name: str, embedding: list[float]):
    with POOL.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO memory_item_embeddings (memory_id, model_name, embedding, updated_at)
                VALUES (%s, %s, %s, now())
                ON CONFLICT (memory_id) DO UPDATE SET
                    model_name = EXCLUDED.model_name,
                    embedding = EXCLUDED.embedding,
                    updated_at = now()
                """,
                (memory_id, model_name, embedding),
            )
        conn.commit()


def fetch_memory_items_for_embedding(status: str = "active"):
    with POOL.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    id,
                    text,
                    memory_type,
                    scope,
                    project_id,
                    subproject_id,
                    workflow_id,
                    pipeline_id,
                    context_scope_id,
                    context_scope_type,
                    context_scope_payload,
                    inheritance_policy,
                    scope_confidence,
                    entity,
                    property,
                    value,
                    status,
                    confidence,
                    0.8 as importance,
                    freshness_class,
                    source_agent,
                    source_session,
                    source_chunk
                FROM memory_items
                WHERE status = %s
                ORDER BY id
                """,
                (status,),
            )

            rows = cur.fetchall()
            return [
                {
                    "id": row[0],
                    "text": row[1],
                    "memory_type": row[2],
                    "scope": row[3],
                    "project_id": row[4],
                    "subproject_id": row[5],
                    "workflow_id": row[6],
                    "pipeline_id": row[7],
                    "context_scope_id": row[8],
                    "context_scope_type": row[9],
                    "context_scope_payload": row[10],
                    "inheritance_policy": row[11],
                    "scope_confidence": row[12],
                    "entity": row[13],
                    "property": row[14],
                    "value": row[15],
                    "status": row[16],
                    "confidence": row[17],
                    "importance": row[18],
                    "freshness_class": row[19],
                    "source_agent": row[20],
                    "source_session": row[21],
                    "source_chunk": row[22],
                }
                for row in rows
            ]

def hybrid_search_memory_items(
    query_embedding,
    *,
    query_text: str,
    status: str = "active",
    allowed_sensitivities: list[str] | None = None,
    limit: int = 20,
    source_agent_prefix: str | None = None,
    after_ts=None,
    before_ts=None,
    half_life_days: float = 30.0,
    recency_weight: float = 0.6,
):
    """
    after_ts / before_ts hard-filter by event time; recency_* tune the decay.

    The filter and the decay do different jobs and both are needed. "What did we
    do yesterday" carries a window, and a window has to be a FILTER: any additive
    recency term still loses to fts_rank * 3.0 on an old document that repeats
    the query words, which is why "phase 4 yesterday" kept returning months-old
    phase-4 docs. The decay handles the untimed general case, where nothing in
    the question says when.

    Rows with no event time are never excluded by a window. ~35% of the store has
    no usable timestamp, and silently dropping a third of memory whenever a
    question mentions a date would trade one wrong answer for a worse one.
    """
    allowed_sensitivities = list(allowed_sensitivities or ["public"])

    sql = """
        WITH q AS (
            SELECT
                ARRAY(
                    SELECT qt
                    FROM unnest(
                        regexp_split_to_array(
                            regexp_replace(LOWER(%s), '[^a-z0-9_ ]', ' ', 'g'),
                            '\\s+'
                        )
                    ) AS qt
                    WHERE qt <> ''
                      AND length(qt) >= 4
                      AND qt NOT IN (
                          'this', 'that', 'with', 'from', 'into', 'they', 'them',
                          'then', 'than', 'have', 'will', 'would', 'could', 'should',
                          'there', 'their', 'about', 'because', 'system'
                      )
                ) AS qterms,
                websearch_to_tsquery('english', %s) AS tsq
        ),
        scored AS (
            SELECT
                m.id, m.text, m.memory_type, m.scope, m.entity, m.property, m.value, m.status,
                m.confidence, 0.8 as importance, m.freshness_class,
                m.source_agent, m.source_session, m.source_chunk,
                m.first_seen, m.last_confirmed, m.supersedes,
                m.tags, m.notes, m.candidate_id, m.candidate_score, m.candidate_reasons,
                m.suggested_route, m.target_file, m.target_section,
                m.sensitivity, m.approved_at, m.approved_by, m.approval_source,
                m.rejected_at, m.rejected_by, m.rejection_reason, m.rejection_source,
                m.ranking_bonus, m.ranking_penalty, m.feedback_last_applied_at,

                ts_rank_cd(
                    to_tsvector(
                        'english',
                        COALESCE(m.text, '') || ' ' ||
                        COALESCE(m.entity, '') || ' ' ||
                        COALESCE(m.property, '') || ' ' ||
                        COALESCE(m.value, '') || ' ' ||
                        COALESCE(m.memory_type, '') || ' ' ||
                        COALESCE(m.scope, '')
                    ),
                    q.tsq
                ) AS fts_rank,

                1 - (e.embedding <=> %s::vector) AS vector_score,

                (
                    0.18 * (
                        SELECT COUNT(*)
                        FROM unnest(q.qterms) qt
                        WHERE qt <> ''
                          AND qt = ANY(
                              regexp_split_to_array(
                                  regexp_replace(LOWER(COALESCE(m.text, '')), '[^a-z0-9_ ]', ' ', 'g'),
                                  '\\s+'
                              )
                          )
                    )
                    +
                    0.20 * (
                        SELECT COUNT(*)
                        FROM unnest(q.qterms) qt
                        WHERE qt <> ''
                          AND qt = ANY(
                              regexp_split_to_array(
                                  regexp_replace(LOWER(COALESCE(m.entity, '')), '[^a-z0-9_ ]', ' ', 'g'),
                                  '\\s+'
                              )
                          )
                    )
                    +
                    0.45 * (
                        SELECT COUNT(*)
                        FROM unnest(q.qterms) qt
                        WHERE qt <> ''
                          AND qt = ANY(
                              regexp_split_to_array(
                                  regexp_replace(LOWER(COALESCE(m.property, '')), '[^a-z0-9_ ]', ' ', 'g'),
                                  '\\s+'
                              )
                          )
                    )
                    +
                    0.55 * (
                        SELECT COUNT(*)
                        FROM unnest(q.qterms) qt
                        WHERE qt <> ''
                          AND qt = ANY(
                              regexp_split_to_array(
                                  regexp_replace(LOWER(COALESCE(m.value, '')), '[^a-z0-9_ ]', ' ', 'g'),
                                  '\\s+'
                              )
                          )
                    )
                ) AS structured_bonus,

                COALESCE(0.8, 0.75) * 0.25 AS importance_bonus,

                -- Best available event time. created_at is insert time and is
                -- fully populated but bulk-stamped; first_seen/last_confirmed
                -- are truer but NULL on roughly a third of rows. Prefer those and
                -- fall back, so an item is dated whenever anything knows when.
                COALESCE(m.last_confirmed, m.first_seen, m.created_at) AS event_at,

                -- Half-life decay, NOT a linear bonus. The old ranking had no
                -- time term at all, so a strong keyword match from March always
                -- beat a good match from yesterday -- exactly the "phase 4"
                -- failure, where fts_rank * 3.0 on an old doc that repeats the
                -- phrase outranks the recent one that answers the question.
                --
                -- Undated rows score 0 here rather than being penalised or
                -- dropped: they keep their relevance ranking and simply gain
                -- nothing, which is the honest treatment of "we do not know
                -- when this happened".
                CASE
                    WHEN COALESCE(m.last_confirmed, m.first_seen, m.created_at) IS NULL THEN 0.0
                    ELSE EXP(
                        -LN(2) *
                        GREATEST(
                            EXTRACT(EPOCH FROM (now() - COALESCE(m.last_confirmed, m.first_seen, m.created_at))) / 86400.0,
                            0
                        ) / %s
                    )
                END AS recency_score

            FROM memory_items m
            JOIN memory_item_embeddings e ON e.memory_id = m.id
            CROSS JOIN q
            WHERE m.status = %s
              AND m.sensitivity = ANY(%s)
              AND (%s::boolean IS FALSE OR m.source_agent LIKE %s)
              -- Undated rows survive the window on purpose (see docstring).
              AND (
                  %s::timestamptz IS NULL
                  OR COALESCE(m.last_confirmed, m.first_seen, m.created_at) IS NULL
                  OR COALESCE(m.last_confirmed, m.first_seen, m.created_at) >= %s::timestamptz
              )
              AND (
                  %s::timestamptz IS NULL
                  OR COALESCE(m.last_confirmed, m.first_seen, m.created_at) IS NULL
                  OR COALESCE(m.last_confirmed, m.first_seen, m.created_at) <= %s::timestamptz
              )
        ),
        ranked AS (
            SELECT
                *,
                (
                    (fts_rank * 3.0)
                    + (vector_score * 1.1)
                    + structured_bonus
                    + importance_bonus
                    -- Weighted to break ties and lift genuinely recent matches
                    -- without letting a fresh irrelevant note outrank an old
                    -- exact answer. Recency is a tiebreaker, not a filter; a
                    -- question with a time window in it should be filtered
                    -- (see after_ts/before_ts), not merely nudged.
                    + (recency_score * %s)
                ) AS final_score
            FROM scored
        )
        SELECT *
        FROM ranked
        ORDER BY
            final_score DESC,
            confidence DESC NULLS LAST,
            importance DESC NULLS LAST,
            last_confirmed DESC NULLS LAST,
            id
        LIMIT %s
    """

    vec = "[" + ",".join(str(float(x)) for x in query_embedding) + "]"

    with POOL.connection() as conn:
        with conn.cursor() as cur:
            sa_filter = bool(source_agent_prefix)
            sa_like = (source_agent_prefix + "%") if source_agent_prefix else "%"
            cur.execute(
                sql,
                (
                    # Order follows the placeholders as they appear in the SQL
                    # TEXT, not their logical grouping: the recency CASE lives in
                    # the `scored` SELECT list, so half_life binds before the
                    # WHERE-clause parameters, and recency_weight binds later in
                    # `ranked`. Getting this wrong bound the sensitivity array to
                    # a float and failed with "ANY requires array on right side".
                    query_text,
                    query_text,
                    vec,
                    max(float(half_life_days), 0.5),
                    status,
                    allowed_sensitivities,
                    sa_filter,
                    sa_like,
                    after_ts,
                    after_ts,
                    before_ts,
                    before_ts,
                    float(recency_weight),
                    limit,
                ),
            )
            cols = [d[0] for d in cur.description]
            rows = []
            for rec in cur.fetchall():
                row = dict(zip(cols, rec))
                row["tags"] = row.get("tags") or []
                row["candidate_reasons"] = row.get("candidate_reasons") or []
                rows.append(row)
            return rows

def close_pool():
    """Shut down the connection pool and its background threads."""
    try:
        POOL.close()
    except Exception:
        pass


atexit.register(close_pool)

def count_memory_items():
    with POOL.connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM memory_items")
            return cur.fetchone()[0]
