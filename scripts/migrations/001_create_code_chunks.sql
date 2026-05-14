-- Migration 001: Create code_chunks table for AST-based code indexing
-- Part of Zero-Reread Architecture, Slice 1 (S1)
-- Date: 2026-04-27
-- Author: Memory Coder
--
-- This table stores parsed code units (functions, classes, methods, modules)
-- extracted via Python AST parsing. Each row represents one semantically
-- meaningful code chunk that can be independently embedded and retrieved.
--
-- Safe to re-run: all statements use IF NOT EXISTS.

CREATE TABLE IF NOT EXISTS code_chunks (
    -- Identity
    id                  TEXT PRIMARY KEY,           -- deterministic hash of (file_path + chunk_type + qualified_name)
    file_path           TEXT NOT NULL,              -- absolute path to source file
    file_hash           TEXT NOT NULL,              -- SHA-256 of file content at parse time (for incremental updates)

    -- Chunk classification
    chunk_type          TEXT NOT NULL,              -- module | function | class | method
    qualified_name      TEXT NOT NULL,              -- e.g. "memory_db.fetch_memory_items_for_embedding"
    module_name         TEXT,                       -- e.g. "memory_db" (derived from filename)

    -- Code content
    signature           TEXT,                       -- e.g. "def fetch_memory_items_for_embedding(status: str = 'active'):"
    docstring           TEXT,                       -- extracted docstring if present
    body                TEXT NOT NULL,              -- full source text of the chunk
    decorators          TEXT[],                     -- list of decorator names (e.g. ARRAY['staticmethod'])

    -- Structural metadata
    imports             JSONB DEFAULT '[]'::jsonb,  -- ["os", "sys", "memory_db.fetch_memory_items_for_embedding", ...]
    line_start          INTEGER NOT NULL,           -- 1-indexed start line in source file
    line_end            INTEGER NOT NULL,           -- 1-indexed end line in source file
    parent_chunk_id     TEXT,                       -- for methods: references the class chunk id
    class_name          TEXT,                       -- for methods: the containing class name

    -- Scoping (matches memory_items pattern)
    project_id          TEXT,                       -- project association if file is in a tracked project path
    subproject_id       TEXT,                       -- subproject association if applicable

    -- Timestamps
    created_at          TIMESTAMPTZ DEFAULT now(),
    updated_at          TIMESTAMPTZ DEFAULT now(),

    -- Constraints
    CONSTRAINT valid_chunk_type CHECK (chunk_type IN ('module', 'function', 'class', 'method')),
    CONSTRAINT valid_lines CHECK (line_start >= 1 AND line_end >= line_start)
);

-- Lookup indexes
CREATE INDEX IF NOT EXISTS idx_code_chunks_file_path      ON code_chunks(file_path);
CREATE INDEX IF NOT EXISTS idx_code_chunks_qualified_name  ON code_chunks(qualified_name);
CREATE INDEX IF NOT EXISTS idx_code_chunks_module_name     ON code_chunks(module_name);
CREATE INDEX IF NOT EXISTS idx_code_chunks_chunk_type      ON code_chunks(chunk_type);
CREATE INDEX IF NOT EXISTS idx_code_chunks_project_id      ON code_chunks(project_id);
CREATE INDEX IF NOT EXISTS idx_code_chunks_file_hash       ON code_chunks(file_hash);
CREATE INDEX IF NOT EXISTS idx_code_chunks_parent          ON code_chunks(parent_chunk_id);

-- Full-text search index for code content (search by function names, variable names, etc.)
CREATE INDEX IF NOT EXISTS idx_code_chunks_fts ON code_chunks USING gin(
    to_tsvector('english',
        COALESCE(qualified_name, '') || ' ' ||
        COALESCE(signature, '') || ' ' ||
        COALESCE(docstring, '') || ' ' ||
        COALESCE(body, '')
    )
);

-- Schema version tracking table (prevents future schema drift)
CREATE TABLE IF NOT EXISTS schema_migrations (
    migration_id    TEXT PRIMARY KEY,
    description     TEXT NOT NULL,
    applied_at      TIMESTAMPTZ DEFAULT now()
);

-- Record this migration
INSERT INTO schema_migrations (migration_id, description)
VALUES ('001_create_code_chunks', 'Create code_chunks table for AST-based code indexing')
ON CONFLICT (migration_id) DO NOTHING;
