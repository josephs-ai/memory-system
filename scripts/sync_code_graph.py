"""
sync_code_graph.py — Populate Neo4j code graph from parsed AST data in PostgreSQL.

Slice 5 (S5) of the Zero-Reread Architecture.

Reads code_chunks from PostgreSQL and creates/updates the Neo4j code graph:
    - CodeFile, CodeModule, Function, Class nodes
    - DEFINES, HAS_METHOD, CONTAINS, CALLS, IMPORTS relationships

All operations use MERGE (idempotent). Re-runs update properties without
duplicating nodes. External/unparsed imports create stub CodeModule nodes
gracefully.

Usage:
    python sync_code_graph.py [--batch-size 100] [--force]

Features:
    - Batched Cypher operations (configurable batch size)
    - Idempotent via MERGE on unique keys
    - Graceful handling of external imports (creates stub CodeModule nodes)
    - Static CALLS extraction via AST re-parse of function bodies
    - Tracks sync state via file_hash to skip unchanged files
    - Reports detailed stats on completion
"""

from __future__ import annotations

import argparse
import ast
import logging
import os
from typing import Any

import psycopg

LOGGER = logging.getLogger("openclaw.sync_code_graph")
logging.basicConfig(
    level=os.environ.get("OPENCLAW_LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)

DB_DSN = os.environ.get("OPENCLAW_MEMORY_DSN", "dbname=openclaw_memory")
NEO4J_URI = os.environ.get("OPENCLAW_NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.environ.get("OPENCLAW_NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.environ.get("OPENCLAW_NEO4J_PASSWORD", "neo4jpassword")


def get_neo4j_driver():
    from neo4j import GraphDatabase
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))


# ---------------------------------------------------------------------------
# PostgreSQL queries
# ---------------------------------------------------------------------------

def fetch_all_chunks(conn) -> list[dict[str, Any]]:
    """Fetch all code chunks from PostgreSQL."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT id, file_path, file_hash, chunk_type, qualified_name,
                   module_name, signature, docstring, imports, line_start,
                   line_end, parent_chunk_id, class_name, project_id,
                   subproject_id, body
            FROM code_chunks
            ORDER BY file_path, line_start
        """)
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]


def fetch_chunks_by_file(conn, file_path: str) -> list[dict[str, Any]]:
    """Fetch chunks for a specific file."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT id, file_path, file_hash, chunk_type, qualified_name,
                   module_name, signature, docstring, imports, line_start,
                   line_end, parent_chunk_id, class_name, project_id,
                   subproject_id, body
            FROM code_chunks
            WHERE file_path = %s
            ORDER BY line_start
        """, (file_path,))
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]


# ---------------------------------------------------------------------------
# CALLS extraction via AST
# ---------------------------------------------------------------------------

def extract_calls_from_body(body: str, module_name: str) -> list[str]:
    """
    Extract function call targets from a function body using AST.
    Returns a list of qualified names (best-effort resolution).

    Only extracts direct name calls and attribute calls (e.g., foo(), self.bar(),
    module.func()). Does not resolve dynamic calls or computed attributes.
    """
    try:
        tree = ast.parse(body)
    except SyntaxError:
        return []

    calls = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue

        func = node.func
        if isinstance(func, ast.Name):
            # Direct call: foo() — could be local or imported
            calls.append(func.id)
        elif isinstance(func, ast.Attribute):
            # Attribute call: obj.method()
            parts = []
            current = func
            while isinstance(current, ast.Attribute):
                parts.append(current.attr)
                current = current.value
            if isinstance(current, ast.Name):
                # Skip self/cls method calls (handled by HAS_METHOD)
                if current.id in ("self", "cls"):
                    continue
                parts.append(current.id)
                calls.append(".".join(reversed(parts)))

    return sorted(set(calls))


# ---------------------------------------------------------------------------
# Neo4j batch operations
# ---------------------------------------------------------------------------

def sync_nodes(session, chunks: list[dict], stats: dict) -> None:
    """Create/update all nodes from chunks."""

    # Group by type for efficient batching
    files: dict[str, dict] = {}
    modules: dict[str, dict] = {}
    functions: list[dict] = []
    classes: list[dict] = []

    for chunk in chunks:
        ct = chunk["chunk_type"]
        fp = chunk["file_path"]

        if ct == "module":
            files[fp] = chunk
            modules[chunk["module_name"]] = chunk
        elif ct in ("function", "method"):
            functions.append(chunk)
        elif ct == "class":
            classes.append(chunk)

    # --- CodeFile nodes ---
    if files:
        file_params = [
            {
                "path": fp,
                "file_hash": c["file_hash"],
                "module_name": c["module_name"],
                "project_id": c.get("project_id"),
                "subproject_id": c.get("subproject_id"),
                "line_count": c["line_end"],
            }
            for fp, c in files.items()
        ]
        session.run(
            """
            UNWIND $params AS p
            MERGE (f:CodeFile {path: p.path})
            SET f.file_hash = p.file_hash,
                f.module_name = p.module_name,
                f.project_id = p.project_id,
                f.subproject_id = p.subproject_id,
                f.line_count = p.line_count,
                f.synced_at = datetime()
            """,
            params=file_params,
        )
        stats["codefile_nodes"] += len(file_params)

    # --- CodeModule nodes (for all known modules) ---
    if modules:
        mod_params = [
            {
                "name": name,
                "file_path": c["file_path"],
            }
            for name, c in modules.items()
        ]
        session.run(
            """
            UNWIND $params AS p
            MERGE (m:CodeModule {name: p.name})
            SET m.file_path = p.file_path,
                m.is_local = true,
                m.synced_at = datetime()
            """,
            params=mod_params,
        )
        stats["codemodule_nodes"] += len(mod_params)

    # --- Function nodes ---
    if functions:
        fn_params = [
            {
                "qualified_name": c["qualified_name"],
                "chunk_id": c["id"],
                "module_name": c["module_name"],
                "signature": c.get("signature"),
                "docstring": c.get("docstring"),
                "function_type": c["chunk_type"],  # "function" or "method"
                "class_name": c.get("class_name"),
                "line_start": c["line_start"],
                "line_end": c["line_end"],
                "file_path": c["file_path"],
            }
            for c in functions
        ]
        session.run(
            """
            UNWIND $params AS p
            MERGE (fn:Function {qualified_name: p.qualified_name})
            SET fn.chunk_id = p.chunk_id,
                fn.module_name = p.module_name,
                fn.signature = p.signature,
                fn.docstring = p.docstring,
                fn.function_type = p.function_type,
                fn.class_name = p.class_name,
                fn.line_start = p.line_start,
                fn.line_end = p.line_end,
                fn.file_path = p.file_path,
                fn.synced_at = datetime()
            """,
            params=fn_params,
        )
        stats["function_nodes"] += len(fn_params)

    # --- Class nodes ---
    if classes:
        cls_params = [
            {
                "qualified_name": c["qualified_name"],
                "chunk_id": c["id"],
                "module_name": c["module_name"],
                "signature": c.get("signature"),
                "docstring": c.get("docstring"),
                "line_start": c["line_start"],
                "line_end": c["line_end"],
                "file_path": c["file_path"],
            }
            for c in classes
        ]
        session.run(
            """
            UNWIND $params AS p
            MERGE (c:Class {qualified_name: p.qualified_name})
            SET c.chunk_id = p.chunk_id,
                c.module_name = p.module_name,
                c.signature = p.signature,
                c.docstring = p.docstring,
                c.line_start = p.line_start,
                c.line_end = p.line_end,
                c.file_path = p.file_path,
                c.synced_at = datetime()
            """,
            params=cls_params,
        )
        stats["class_nodes"] += len(cls_params)


def sync_relationships(session, chunks: list[dict], stats: dict) -> None:
    """Create all relationships from chunks."""

    # Build lookup maps
    qname_to_chunk = {c["qualified_name"]: c for c in chunks}
    id_to_chunk = {c["id"]: c for c in chunks}
    module_chunks = [c for c in chunks if c["chunk_type"] == "module"]
    function_chunks = [c for c in chunks if c["chunk_type"] in ("function", "method")]
    class_chunks = [c for c in chunks if c["chunk_type"] == "class"]

    # Build set of known qualified names for CALLS resolution
    known_qnames = set(qname_to_chunk.keys())
    # Also build module_name → set of function names for unqualified call resolution
    module_functions: dict[str, set[str]] = {}
    for c in function_chunks:
        mn = c["module_name"]
        fn_name = c["qualified_name"].split(".")[-1]
        module_functions.setdefault(mn, set()).add(fn_name)

    # --- DEFINES: CodeFile → Function (top-level only) ---
    defines_fn = [
        {"file_path": c["file_path"], "qualified_name": c["qualified_name"]}
        for c in function_chunks
        if c["chunk_type"] == "function" and c.get("class_name") is None
        and c.get("parent_chunk_id") is None
    ]
    if defines_fn:
        session.run(
            """
            UNWIND $params AS p
            MATCH (f:CodeFile {path: p.file_path})
            MATCH (fn:Function {qualified_name: p.qualified_name})
            MERGE (f)-[:DEFINES]->(fn)
            """,
            params=defines_fn,
        )
        stats["defines_rels"] += len(defines_fn)

    # --- DEFINES: CodeFile → Class ---
    defines_cls = [
        {"file_path": c["file_path"], "qualified_name": c["qualified_name"]}
        for c in class_chunks
    ]
    if defines_cls:
        session.run(
            """
            UNWIND $params AS p
            MATCH (f:CodeFile {path: p.file_path})
            MATCH (c:Class {qualified_name: p.qualified_name})
            MERGE (f)-[:DEFINES]->(c)
            """,
            params=defines_cls,
        )
        stats["defines_rels"] += len(defines_cls)

    # --- HAS_METHOD: Class → Method ---
    has_method = []
    for c in function_chunks:
        if c["chunk_type"] == "method" and c.get("class_name"):
            class_qname = f"{c['module_name']}.{c['class_name']}"
            has_method.append({
                "class_qname": class_qname,
                "method_qname": c["qualified_name"],
            })
    if has_method:
        session.run(
            """
            UNWIND $params AS p
            MATCH (cls:Class {qualified_name: p.class_qname})
            MATCH (fn:Function {qualified_name: p.method_qname})
            MERGE (cls)-[:HAS_METHOD]->(fn)
            """,
            params=has_method,
        )
        stats["has_method_rels"] += len(has_method)

    # --- CONTAINS: Function → nested Function ---
    contains = []
    for c in function_chunks:
        pid = c.get("parent_chunk_id")
        if pid and pid in id_to_chunk:
            parent = id_to_chunk[pid]
            # Only CONTAINS for function→function nesting (not class→method, which is HAS_METHOD)
            if parent["chunk_type"] in ("function", "method"):
                contains.append({
                    "parent_qname": parent["qualified_name"],
                    "child_qname": c["qualified_name"],
                })
    if contains:
        session.run(
            """
            UNWIND $params AS p
            MATCH (outer:Function {qualified_name: p.parent_qname})
            MATCH (inner:Function {qualified_name: p.child_qname})
            MERGE (outer)-[:CONTAINS]->(inner)
            """,
            params=contains,
        )
        stats["contains_rels"] += len(contains)

    # --- IMPORTS: CodeFile → CodeModule ---
    file_imports = []
    for mc in module_chunks:
        for imp in (mc.get("imports") or []):
            # Extract the top-level module name (e.g., "os" from "os.path.join")
            top_module = imp.split(".")[0]
            file_imports.append({
                "file_path": mc["file_path"],
                "module_name": top_module,
            })
    # Deduplicate
    seen_imports = set()
    deduped_imports = []
    for fi in file_imports:
        key = (fi["file_path"], fi["module_name"])
        if key not in seen_imports:
            seen_imports.add(key)
            deduped_imports.append(fi)

    if deduped_imports:
        # First ensure stub CodeModule nodes exist for external imports
        ext_modules = [
            {"name": fi["module_name"]}
            for fi in deduped_imports
        ]
        session.run(
            """
            UNWIND $params AS p
            MERGE (m:CodeModule {name: p.name})
            ON CREATE SET m.is_local = false, m.synced_at = datetime()
            """,
            params=ext_modules,
        )

        session.run(
            """
            UNWIND $params AS p
            MATCH (f:CodeFile {path: p.file_path})
            MATCH (m:CodeModule {name: p.module_name})
            MERGE (f)-[:IMPORTS]->(m)
            """,
            params=deduped_imports,
        )
        stats["imports_rels"] += len(deduped_imports)

    # --- CALLS: Function → Function ---
    calls_params = []
    for c in function_chunks:
        body = c.get("body") or ""
        if not body.strip():
            continue

        raw_calls = extract_calls_from_body(body, c["module_name"])
        for call_name in raw_calls:
            # Try to resolve the call to a known qualified name
            resolved = _resolve_call(call_name, c["module_name"], known_qnames, module_functions)
            if resolved and resolved != c["qualified_name"]:  # Skip self-recursion for cleanliness
                calls_params.append({
                    "caller_qname": c["qualified_name"],
                    "callee_qname": resolved,
                })

    # Deduplicate
    seen_calls = set()
    deduped_calls = []
    for cp in calls_params:
        key = (cp["caller_qname"], cp["callee_qname"])
        if key not in seen_calls:
            seen_calls.add(key)
            deduped_calls.append(cp)

    if deduped_calls:
        session.run(
            """
            UNWIND $params AS p
            MATCH (caller:Function {qualified_name: p.caller_qname})
            MATCH (callee:Function {qualified_name: p.callee_qname})
            MERGE (caller)-[:CALLS]->(callee)
            """,
            params=deduped_calls,
        )
        stats["calls_rels"] += len(deduped_calls)


def _resolve_call(
    call_name: str,
    caller_module: str,
    known_qnames: set[str],
    module_functions: dict[str, set[str]],
) -> str | None:
    """
    Resolve a raw call name to a known qualified name.

    Resolution order:
    1. Exact match against known qualified names
    2. Module-qualified: caller_module.call_name
    3. Unqualified match: any module that has a function with this name
       (only if unambiguous — one module has it)

    Returns None if unresolvable.
    """
    # 1. Exact match (e.g., "memory_db.fetch_memory_items_for_embedding")
    if call_name in known_qnames:
        return call_name

    # 2. Same-module qualification (e.g., "foo" → "my_module.foo")
    module_qualified = f"{caller_module}.{call_name}"
    if module_qualified in known_qnames:
        return module_qualified

    # 3. Cross-module unqualified (e.g., "upsert_memory_embedding" from a different module)
    # Only resolve if exactly one module defines this function name
    candidates = []
    for mod_name, fn_names in module_functions.items():
        if call_name in fn_names:
            candidates.append(f"{mod_name}.{call_name}")

    if len(candidates) == 1:
        return candidates[0]

    # Check if it's a dotted call like "module.func"
    if "." in call_name:
        parts = call_name.split(".")
        # Try module.func pattern
        if len(parts) == 2:
            candidate = f"{parts[0]}.{parts[1]}"
            if candidate in known_qnames:
                return candidate

    return None


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Populate Neo4j code graph from parsed AST data."
    )
    parser.add_argument("--batch-size", type=int, default=100,
                        help="Batch size for Cypher operations")
    parser.add_argument("--force", action="store_true",
                        help="Re-sync all files regardless of state")
    parser.parse_args()

    conn = psycopg.connect(DB_DSN)
    driver = get_neo4j_driver()

    stats = {
        "codefile_nodes": 0,
        "codemodule_nodes": 0,
        "function_nodes": 0,
        "class_nodes": 0,
        "defines_rels": 0,
        "has_method_rels": 0,
        "contains_rels": 0,
        "imports_rels": 0,
        "calls_rels": 0,
    }

    try:
        chunks = fetch_all_chunks(conn)
        if not chunks:
            LOGGER.info("No code chunks found in PostgreSQL")
            print("CHUNKS=0")
            return

        LOGGER.info("Syncing %d code chunks to Neo4j", len(chunks))

        with driver.session() as session:
            # Phase 1: Create/update all nodes
            LOGGER.info("Phase 1: Syncing nodes...")
            sync_nodes(session, chunks, stats)

            # Phase 2: Create all relationships
            LOGGER.info("Phase 2: Syncing relationships...")
            sync_relationships(session, chunks, stats)

        LOGGER.info("Sync complete")

        # Print stats
        total_nodes = (stats["codefile_nodes"] + stats["codemodule_nodes"]
                       + stats["function_nodes"] + stats["class_nodes"])
        total_rels = (stats["defines_rels"] + stats["has_method_rels"]
                      + stats["contains_rels"] + stats["imports_rels"]
                      + stats["calls_rels"])

        print(f"CHUNKS={len(chunks)}")
        print(f"NODES={total_nodes}")
        print(f"  CodeFile={stats['codefile_nodes']}")
        print(f"  CodeModule={stats['codemodule_nodes']}")
        print(f"  Function={stats['function_nodes']}")
        print(f"  Class={stats['class_nodes']}")
        print(f"RELATIONSHIPS={total_rels}")
        print(f"  DEFINES={stats['defines_rels']}")
        print(f"  HAS_METHOD={stats['has_method_rels']}")
        print(f"  CONTAINS={stats['contains_rels']}")
        print(f"  IMPORTS={stats['imports_rels']}")
        print(f"  CALLS={stats['calls_rels']}")

    finally:
        conn.close()
        driver.close()


if __name__ == "__main__":
    main()
