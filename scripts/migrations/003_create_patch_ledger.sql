-- Migration 003: Create patch_ledger table for tracking per-step file modifications.
-- L8-S2 of the Level 8 "Autonomous Closed-Loop Execution" implementation.
--
-- Records every file modification made by the campaign orchestrator,
-- enabling atomic rollback without corrupting the Neo4j graph or
-- leaving orphaned code chunks.

CREATE TABLE IF NOT EXISTS patch_ledger (
    id              BIGSERIAL PRIMARY KEY,
    patch_id        TEXT NOT NULL UNIQUE,          -- Deterministic ID: hash of (step_id, file_path)
    step_id         TEXT NOT NULL,                  -- Campaign step identifier
    campaign_id     TEXT,                           -- Optional campaign grouping
    file_path       TEXT NOT NULL,                  -- Absolute path of modified file
    operation       TEXT NOT NULL CHECK (operation IN ('create', 'modify', 'delete')),
    file_hash_before TEXT,                          -- SHA-256 before modification (NULL for creates)
    file_hash_after  TEXT,                          -- SHA-256 after modification (NULL for deletes)
    content_before  TEXT,                           -- Full file content before (NULL for creates)
    content_after   TEXT,                           -- Full file content after (NULL for deletes)
    chunk_ids_before TEXT[],                        -- code_chunks IDs that existed before
    chunk_ids_after  TEXT[],                        -- code_chunks IDs that exist after
    graph_nodes_affected TEXT[],                    -- Neo4j node qualified_names affected
    rolled_back     BOOLEAN NOT NULL DEFAULT FALSE, -- Whether this patch has been rolled back
    rolled_back_at  TIMESTAMPTZ,
    rollback_step_id TEXT,                          -- Step that triggered the rollback
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    metadata        JSONB                           -- Arbitrary metadata (agent name, model, etc.)
);

-- Indexes for common query patterns
CREATE INDEX IF NOT EXISTS idx_patch_ledger_step ON patch_ledger(step_id);
CREATE INDEX IF NOT EXISTS idx_patch_ledger_campaign ON patch_ledger(campaign_id);
CREATE INDEX IF NOT EXISTS idx_patch_ledger_file ON patch_ledger(file_path);
CREATE INDEX IF NOT EXISTS idx_patch_ledger_created ON patch_ledger(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_patch_ledger_not_rolled_back ON patch_ledger(file_path)
    WHERE rolled_back = FALSE;

-- Track migration
INSERT INTO schema_migrations (migration_id, description)
VALUES ('003_create_patch_ledger', 'Patch ledger for per-step file modification tracking with atomic rollback')
ON CONFLICT (migration_id) DO NOTHING;
