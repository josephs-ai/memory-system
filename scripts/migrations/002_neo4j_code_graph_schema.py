"""
Migration 002: Create Neo4j Code Graph schema for the Zero-Reread Architecture.

Slice 4 (S4) of the Zero-Reread implementation plan.

Creates node labels, uniqueness constraints, and indexes for the code graph.
All operations are idempotent (IF NOT EXISTS).

Node types:
    :CodeFile    — A source file tracked by the code index
    :CodeModule  — A Python module (1:1 with CodeFile for .py files)
    :Function    — A function or method extracted by the AST parser
    :Class       — A class definition

Relationship types (all strictly directional to prevent cyclic traversal blowouts):
    (:CodeFile)-[:DEFINES]->(:Function)        — file defines a top-level function
    (:CodeFile)-[:DEFINES]->(:Class)           — file defines a class
    (:Class)-[:HAS_METHOD]->(:Function)        — class contains a method
    (:Function)-[:CONTAINS]->(:Function)       — function contains a nested function
    (:Function)-[:CALLS]->(:Function)          — function calls another function
    (:CodeFile)-[:IMPORTS]->(:CodeModule)      — file imports a module
    (:Function)-[:IMPORTS]->(:CodeModule)      — function has inline imports
    (:MemoryItem)-[:ABOUT_CODE]->(:Function)   — memory item relates to a function
    (:MemoryItem)-[:ABOUT_CODE]->(:Class)      — memory item relates to a class
    (:MemoryItem)-[:ABOUT_CODE]->(:CodeFile)   — memory item relates to a file

Directionality rules:
    - All relationships flow from container → contained or from caller → callee
    - DEFINES: file → definition (downward)
    - HAS_METHOD: class → method (downward)
    - CONTAINS: outer function → inner function (downward)
    - CALLS: caller → callee (forward reference)
    - IMPORTS: importer → imported module (forward reference)
    - ABOUT_CODE: memory item → code element (annotation direction)
    - No bidirectional or undirected relationships exist
    - Graph queries should respect direction to avoid blowouts

Usage:
    python migrations/002_neo4j_code_graph_schema.py [--verify]
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

LOGGER = logging.getLogger("openclaw.migration_002")
logging.basicConfig(
    level=os.environ.get("OPENCLAW_LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)

NEO4J_URI = os.environ.get("OPENCLAW_NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.environ.get("OPENCLAW_NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.environ.get("OPENCLAW_NEO4J_PASSWORD", "neo4jpassword")


def get_driver():
    from neo4j import GraphDatabase
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))


# ---------------------------------------------------------------------------
# Constraints (uniqueness + existence)
# ---------------------------------------------------------------------------

CONSTRAINTS = [
    # CodeFile: unique by absolute file path
    (
        "codefile_path_unique",
        "CREATE CONSTRAINT codefile_path_unique IF NOT EXISTS "
        "FOR (f:CodeFile) REQUIRE f.path IS UNIQUE"
    ),
    # CodeModule: unique by module name
    (
        "codemodule_name_unique",
        "CREATE CONSTRAINT codemodule_name_unique IF NOT EXISTS "
        "FOR (m:CodeModule) REQUIRE m.name IS UNIQUE"
    ),
    # Function: unique by qualified_name (module.class.func or module.func)
    (
        "function_qname_unique",
        "CREATE CONSTRAINT function_qname_unique IF NOT EXISTS "
        "FOR (fn:Function) REQUIRE fn.qualified_name IS UNIQUE"
    ),
    # Class: unique by qualified_name (module.ClassName)
    (
        "class_qname_unique",
        "CREATE CONSTRAINT class_qname_unique IF NOT EXISTS "
        "FOR (c:Class) REQUIRE c.qualified_name IS UNIQUE"
    ),
]


# ---------------------------------------------------------------------------
# Indexes (for traversal and lookup performance)
# ---------------------------------------------------------------------------

INDEXES = [
    # CodeFile indexes
    (
        "idx_codefile_hash",
        "CREATE INDEX idx_codefile_hash IF NOT EXISTS "
        "FOR (f:CodeFile) ON (f.file_hash)"
    ),
    (
        "idx_codefile_project",
        "CREATE INDEX idx_codefile_project IF NOT EXISTS "
        "FOR (f:CodeFile) ON (f.project_id)"
    ),
    # Function indexes
    (
        "idx_function_module",
        "CREATE INDEX idx_function_module IF NOT EXISTS "
        "FOR (fn:Function) ON (fn.module_name)"
    ),
    (
        "idx_function_chunk_id",
        "CREATE INDEX idx_function_chunk_id IF NOT EXISTS "
        "FOR (fn:Function) ON (fn.chunk_id)"
    ),
    (
        "idx_function_type",
        "CREATE INDEX idx_function_type IF NOT EXISTS "
        "FOR (fn:Function) ON (fn.function_type)"
    ),
    # Class indexes
    (
        "idx_class_module",
        "CREATE INDEX idx_class_module IF NOT EXISTS "
        "FOR (c:Class) ON (c.module_name)"
    ),
    (
        "idx_class_chunk_id",
        "CREATE INDEX idx_class_chunk_id IF NOT EXISTS "
        "FOR (c:Class) ON (c.chunk_id)"
    ),
    # CodeModule indexes
    (
        "idx_codemodule_file",
        "CREATE INDEX idx_codemodule_file IF NOT EXISTS "
        "FOR (m:CodeModule) ON (m.file_path)"
    ),
    # Full-text index for natural language code search across the graph
    # (searches qualified_name + signature + docstring on Functions)
    (
        "idx_function_fulltext",
        "CREATE FULLTEXT INDEX idx_function_fulltext IF NOT EXISTS "
        "FOR (fn:Function) ON EACH [fn.qualified_name, fn.signature, fn.docstring]"
    ),
]


# ---------------------------------------------------------------------------
# Apply
# ---------------------------------------------------------------------------

def apply_schema(driver, *, verify_only: bool = False) -> dict:
    """Apply all constraints and indexes. Returns stats."""
    stats = {"constraints": 0, "indexes": 0, "errors": []}

    with driver.session() as session:
        # Apply constraints
        for name, cypher in CONSTRAINTS:
            try:
                if verify_only:
                    LOGGER.info("VERIFY constraint: %s", name)
                else:
                    session.run(cypher)
                    LOGGER.info("APPLIED constraint: %s", name)
                stats["constraints"] += 1
            except Exception as e:
                msg = f"FAILED constraint {name}: {e}"
                LOGGER.error(msg)
                stats["errors"].append(msg)

        # Apply indexes
        for name, cypher in INDEXES:
            try:
                if verify_only:
                    LOGGER.info("VERIFY index: %s", name)
                else:
                    session.run(cypher)
                    LOGGER.info("APPLIED index: %s", name)
                stats["indexes"] += 1
            except Exception as e:
                msg = f"FAILED index {name}: {e}"
                LOGGER.error(msg)
                stats["errors"].append(msg)

    return stats


def verify_schema(driver) -> dict:
    """Verify all constraints and indexes exist."""
    results = {"constraints": [], "indexes": [], "missing": []}

    with driver.session() as session:
        # Check constraints
        constraint_result = session.run("SHOW CONSTRAINTS")
        existing_constraints = {r["name"] for r in constraint_result}
        for name, _ in CONSTRAINTS:
            if name in existing_constraints:
                results["constraints"].append(name)
            else:
                results["missing"].append(f"constraint:{name}")

        # Check indexes
        index_result = session.run("SHOW INDEXES")
        existing_indexes = {r["name"] for r in index_result}
        for name, _ in INDEXES:
            if name in existing_indexes:
                results["indexes"].append(name)
            else:
                results["missing"].append(f"index:{name}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Apply Neo4j code graph schema (Migration 002)"
    )
    parser.add_argument("--verify", action="store_true",
                        help="Only verify schema exists, don't apply")
    args = parser.parse_args()

    driver = get_driver()
    try:
        if args.verify:
            results = verify_schema(driver)
            print(f"CONSTRAINTS={len(results['constraints'])}/{len(CONSTRAINTS)}")
            print(f"INDEXES={len(results['indexes'])}/{len(INDEXES)}")
            if results["missing"]:
                print(f"MISSING={','.join(results['missing'])}")
                sys.exit(1)
            else:
                print("ALL_PRESENT=true")
        else:
            stats = apply_schema(driver)
            print(f"CONSTRAINTS={stats['constraints']}")
            print(f"INDEXES={stats['indexes']}")
            if stats["errors"]:
                print(f"ERRORS={len(stats['errors'])}")
                for e in stats["errors"]:
                    print(f"  {e}")
                sys.exit(1)
            else:
                print("ERRORS=0")

            # Record migration in PostgreSQL
            try:
                import psycopg
                DB_DSN = os.environ.get("OPENCLAW_MEMORY_DSN", "dbname=openclaw_memory")
                conn = psycopg.connect(DB_DSN)
                with conn.cursor() as cur:
                    cur.execute(
                        """INSERT INTO schema_migrations (migration_id, description)
                           VALUES (%s, %s)
                           ON CONFLICT (migration_id) DO NOTHING""",
                        ("002_neo4j_code_graph_schema",
                         "Neo4j code graph schema: CodeFile, CodeModule, Function, Class nodes + DEFINES, HAS_METHOD, CONTAINS, CALLS, IMPORTS, ABOUT_CODE relationships")
                    )
                conn.commit()
                conn.close()
                LOGGER.info("Migration recorded in schema_migrations")
            except Exception as e:
                LOGGER.warning("Could not record migration in PostgreSQL: %s", e)

    finally:
        driver.close()


if __name__ == "__main__":
    main()
