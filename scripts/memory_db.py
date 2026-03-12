import os
from pathlib import Path

import psycopg
from psycopg.types.json import Jsonb
from psycopg_pool import ConnectionPool

WORKSPACE = Path.home() / ".openclaw" / "workspace"
DEFAULT_DSN = os.environ.get("OPENCLAW_MEMORY_DB_DSN", "dbname=openclaw_memory user=postgres")

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

JSONB_COLUMNS = {"tags", "candidate_reasons"}

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
                    payload
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
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
    with POOL.connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT payload FROM memory_discarded")
            return [row[0] for row in cur.fetchall()]


def candidate_matches_rejected(item: dict) -> tuple[bool, dict | None]:
    text = (item.get("text") or "").strip().lower()
    entity = item.get("entity")
    prop = item.get("property")
    value = item.get("value")
    scope = item.get("scope")

    for old in fetch_discarded_payloads():
        old_text = (old.get("text") or "").strip().lower()

        exact_slot = (
            entity
            and prop
            and value
            and old.get("entity") == entity
            and old.get("property") == prop
            and old.get("value") == value
        )

        exact_text = text and old_text and text == old_text

        same_scope_slot = (
            exact_slot
            and scope is not None
            and old.get("scope") == scope
        )

        if exact_text or exact_slot or same_scope_slot:
            return True, old

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
            id, text, memory_type, scope, entity, property, value, status,
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
            id, text, memory_type, scope, entity, property, value, status,
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
                SELECT id, text
                FROM memory_items
                WHERE status = %s
                ORDER BY id
                """,
                (status,),
            )
            return [{"id": row[0], "text": row[1]} for row in cur.fetchall()]


def hybrid_search_memory_items(
    query_embedding,
    *,
    query_text: str,
    status: str = "active",
    allowed_sensitivities: list[str] | None = None,
    limit: int = 20,
):
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
                m.confidence, m.importance, m.freshness_class,
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

                COALESCE(m.importance, 0.75) * 0.25 AS importance_bonus

            FROM memory_items m
            JOIN memory_item_embeddings e ON e.memory_id = m.id
            CROSS JOIN q
            WHERE m.status = %s
              AND m.sensitivity = ANY(%s)
        ),
        ranked AS (
            SELECT
                *,
                (
                    (fts_rank * 3.0)
                    + (vector_score * 1.1)
                    + structured_bonus
                    + importance_bonus
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
            cur.execute(
                sql,
                (
                    query_text,
                    query_text,
                    vec,
                    status,
                    allowed_sensitivities,
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
    POOL.close()

def count_memory_items():
    with POOL.connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM memory_items")
            return cur.fetchone()[0]
