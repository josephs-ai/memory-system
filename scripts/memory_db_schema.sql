CREATE TABLE IF NOT EXISTS memory_items (
    id TEXT PRIMARY KEY,
    text TEXT NOT NULL,
    memory_type TEXT,
    scope TEXT,
    project_id TEXT,
    subproject_id TEXT,
    workflow_id TEXT,
    pipeline_id TEXT,
    context_scope_id TEXT,
    context_scope_type TEXT,
    context_scope_payload JSONB DEFAULT '{}'::jsonb,
    inheritance_policy TEXT,
    scope_confidence DOUBLE PRECISION,
    entity TEXT,
    property TEXT,
    value TEXT,
    status TEXT NOT NULL,
    confidence DOUBLE PRECISION,
    importance DOUBLE PRECISION,
    freshness_class TEXT,
    source_agent TEXT,
    source_session TEXT,
    source_chunk TEXT,
    first_seen TIMESTAMPTZ,
    last_confirmed TIMESTAMPTZ,
    supersedes TEXT,
    tags JSONB DEFAULT '[]'::jsonb,
    notes TEXT,
    candidate_id TEXT,
    candidate_score DOUBLE PRECISION,
    candidate_reasons JSONB DEFAULT '[]'::jsonb,
    suggested_route TEXT,
    target_file TEXT,
    target_section TEXT,
    sensitivity TEXT DEFAULT 'public',
    approved_at TIMESTAMPTZ,
    approved_by TEXT,
    approval_source TEXT,
    rejected_at TIMESTAMPTZ,
    rejected_by TEXT,
    rejection_reason TEXT,
    rejection_source TEXT,
    ranking_bonus DOUBLE PRECISION DEFAULT 0,
    ranking_penalty DOUBLE PRECISION DEFAULT 0,
    feedback_last_applied_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_memory_items_status ON memory_items(status);
CREATE INDEX IF NOT EXISTS idx_memory_items_scope ON memory_items(scope);
CREATE INDEX IF NOT EXISTS idx_memory_items_context_scope_id ON memory_items(context_scope_id);
CREATE INDEX IF NOT EXISTS idx_memory_items_entity ON memory_items(entity);
CREATE INDEX IF NOT EXISTS idx_memory_items_property ON memory_items(property);
CREATE INDEX IF NOT EXISTS idx_memory_items_memory_type ON memory_items(memory_type);

CREATE TABLE IF NOT EXISTS memory_pending_stable (
    id TEXT PRIMARY KEY,
    payload JSONB NOT NULL,
    queued_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS memory_inbox (
    id TEXT PRIMARY KEY,
    payload JSONB NOT NULL,
    queued_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS memory_discarded (
    id TEXT PRIMARY KEY,
    payload JSONB NOT NULL,
    discarded_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS retrieval_feedback (
    feedback_id BIGSERIAL PRIMARY KEY,
    selected_id TEXT,
    selected_score DOUBLE PRECISION,
    selected_status TEXT,
    operator_feedback TEXT,
    feedback_note TEXT,
    feedback_at TIMESTAMPTZ,
    candidate_file TEXT,
    target_file TEXT,
    include_superseded BOOLEAN,
    payload JSONB NOT NULL
);

CREATE TABLE IF NOT EXISTS global_insights (
    insight_id BIGSERIAL PRIMARY KEY,
    topic TEXT NOT NULL,
    text TEXT NOT NULL,
    source_project TEXT,
    promoted_by TEXT,
    created_at TIMESTAMPTZ DEFAULT now(),
    UNIQUE(topic, text)
);

CREATE TABLE IF NOT EXISTS restore_points (
    restore_id BIGSERIAL PRIMARY KEY,
    label TEXT,
    path TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS project_locks (
    project_id TEXT PRIMARY KEY,
    owner TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT now(),
    created_ts DOUBLE PRECISION,
    pid INTEGER
);

-- Episodic lane.
--
-- memory_items answers "what is true" -- semantic facts, deduped by
-- (memory_type, entity, property, value). That dedup is correct for facts and
-- exactly wrong for activity: re-doing work yesterday re-asserts facts already
-- known, so the work collapses into existing rows and leaves no trace. Asking a
-- fact store "what did we do yesterday" therefore returns months-old rows that
-- merely share vocabulary.
--
-- This table answers "what happened, and when". It is append-only and is never
-- deduplicated on content: two identical days of work are two episodes. Time is
-- a hard filter here rather than a decay weight, because the question is a
-- window, not a similarity.
CREATE TABLE IF NOT EXISTS memory_episodes (
    id BIGSERIAL PRIMARY KEY,
    agent TEXT NOT NULL,
    source_session TEXT,
    started_at TIMESTAMPTZ NOT NULL,
    ended_at TIMESTAMPTZ,
    summary TEXT NOT NULL,
    artifacts JSONB NOT NULL DEFAULT '{}'::jsonb,
    item_count INTEGER NOT NULL DEFAULT 0,
    origin TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS memory_episodes_started_idx
    ON memory_episodes (started_at DESC);
CREATE INDEX IF NOT EXISTS memory_episodes_agent_started_idx
    ON memory_episodes (agent, started_at DESC);
CREATE INDEX IF NOT EXISTS memory_episodes_summary_fts_idx
    ON memory_episodes USING GIN (to_tsvector('english', summary));
