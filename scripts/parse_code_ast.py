"""
parse_code_ast.py — AST-based code parser for the Zero-Reread Architecture.

Parses Python source files into semantically meaningful code chunks
(modules, functions, classes, methods) and upserts them into the
code_chunks PostgreSQL table.

Slice 2 (S2) of the Zero-Reread implementation plan.

Usage:
    python parse_code_ast.py /path/to/directory [--project-id my_project] [--dry-run]
    python parse_code_ast.py /path/to/single_file.py [--project-id my_project]

Features:
    - Deterministic chunk IDs (hash of file_path + chunk_type + qualified_name)
    - Incremental: skips files whose SHA-256 hash hasn't changed
    - Handles nested functions, async functions, complex decorators, dataclasses
    - Extracts imports at module and function/class level
    - Upserts with ON CONFLICT for safe re-runs
    - Removes stale chunks when a file is re-parsed (chunks from old parse deleted)
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import logging
import os
import sys
import textwrap
from pathlib import Path
from typing import Any

import psycopg
from psycopg.types.json import Jsonb

LOGGER = logging.getLogger("openclaw.parse_code_ast")
logging.basicConfig(
    level=os.environ.get("OPENCLAW_AST_LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)

DB_DSN = os.environ.get("OPENCLAW_MEMORY_DSN", "dbname=openclaw_memory")


# ---------------------------------------------------------------------------
# Chunk ID generation
# ---------------------------------------------------------------------------

def make_chunk_id(file_path: str, chunk_type: str, qualified_name: str) -> str:
    """Deterministic chunk ID from (file_path, chunk_type, qualified_name)."""
    raw = f"{file_path}::{chunk_type}::{qualified_name}"
    return "cc-" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


# ---------------------------------------------------------------------------
# Source extraction helpers
# ---------------------------------------------------------------------------

def _file_hash(content: str) -> str:
    """SHA-256 of file content."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _get_source_segment(lines: list[str], node: ast.AST) -> str:
    """Extract source lines for an AST node using its line range."""
    start = node.lineno - 1  # 0-indexed
    end = node.end_lineno     # end_lineno is 1-indexed, inclusive
    if end is None:
        end = start + 1
    return "\n".join(lines[start:end])


def _get_decorator_names(node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef) -> list[str]:
    """Extract decorator names as strings."""
    names = []
    for dec in node.decorator_list:
        if isinstance(dec, ast.Name):
            names.append(dec.id)
        elif isinstance(dec, ast.Attribute):
            names.append(ast.dump(dec))  # fallback for complex attrs
            # Try to reconstruct dotted name
            parts = []
            current = dec
            while isinstance(current, ast.Attribute):
                parts.append(current.attr)
                current = current.value
            if isinstance(current, ast.Name):
                parts.append(current.id)
                names[-1] = ".".join(reversed(parts))
        elif isinstance(dec, ast.Call):
            # Decorator with arguments: @decorator(args)
            func = dec.func
            if isinstance(func, ast.Name):
                names.append(func.id)
            elif isinstance(func, ast.Attribute):
                parts = []
                current = func
                while isinstance(current, ast.Attribute):
                    parts.append(current.attr)
                    current = current.value
                if isinstance(current, ast.Name):
                    parts.append(current.id)
                    names.append(".".join(reversed(parts)))
                else:
                    names.append(ast.dump(func))
            else:
                names.append(ast.dump(func))
        else:
            names.append(ast.dump(dec))
    return names


def _get_signature(node: ast.FunctionDef | ast.AsyncFunctionDef, lines: list[str]) -> str:
    """Reconstruct the function signature line(s) from source."""
    # The signature spans from the first decorator (or def) to the colon
    start = node.lineno - 1
    # Find the line with the colon that ends the signature
    for i in range(start, min(start + 20, len(lines))):
        line = lines[i]
        if line.rstrip().endswith(":"):
            sig_lines = lines[start:i + 1]
            sig = "\n".join(sig_lines).strip()
            # Remove decorators from signature — just keep the def line(s)
            result_lines = []
            in_def = False
            for sl in sig_lines:
                stripped = sl.strip()
                if stripped.startswith("def ") or stripped.startswith("async def "):
                    in_def = True
                if in_def:
                    result_lines.append(sl)
            return "\n".join(result_lines).strip() if result_lines else sig
    # Fallback: just return the first line
    return lines[start].strip()


def _extract_imports(tree_or_node: ast.AST) -> list[str]:
    """Extract import names from an AST node (module-level or function body)."""
    imports = []
    for node in ast.walk(tree_or_node):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for alias in node.names:
                if module:
                    imports.append(f"{module}.{alias.name}")
                else:
                    imports.append(alias.name)
    return sorted(set(imports))


def _module_name_from_path(file_path: str) -> str:
    """Derive module name from file path (just the stem)."""
    return Path(file_path).stem


# ---------------------------------------------------------------------------
# AST Parsing
# ---------------------------------------------------------------------------

def parse_file(file_path: str, content: str) -> list[dict[str, Any]]:
    """
    Parse a Python file into code chunks.

    Returns a list of chunk dicts ready for database insertion.
    Handles: module-level, top-level functions, top-level classes,
    methods within classes, nested functions (flattened with parent reference).
    """
    try:
        tree = ast.parse(content, filename=file_path)
    except SyntaxError as e:
        LOGGER.warning("SyntaxError parsing %s: %s", file_path, e)
        return []

    lines = content.splitlines()
    fhash = _file_hash(content)
    module = _module_name_from_path(file_path)
    chunks: list[dict[str, Any]] = []

    # --- Module chunk ---
    module_docstring = ast.get_docstring(tree)
    module_imports = _extract_imports(tree)

    # Module chunk covers the entire file
    chunks.append({
        "id": make_chunk_id(file_path, "module", module),
        "file_path": file_path,
        "file_hash": fhash,
        "chunk_type": "module",
        "qualified_name": module,
        "module_name": module,
        "signature": None,
        "docstring": module_docstring,
        "body": content,
        "decorators": None,
        "imports": module_imports,
        "line_start": 1,
        "line_end": len(lines) or 1,
        "parent_chunk_id": None,
        "class_name": None,
    })

    # --- Walk top-level definitions ---
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            _parse_function(node, file_path, fhash, module, lines, chunks,
                            parent_chunk_id=None, class_name=None)

        elif isinstance(node, ast.ClassDef):
            _parse_class(node, file_path, fhash, module, lines, chunks)

    return chunks


def _parse_function(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    file_path: str,
    fhash: str,
    module: str,
    lines: list[str],
    chunks: list[dict],
    parent_chunk_id: str | None,
    class_name: str | None,
) -> None:
    """Parse a function/method node into a chunk, including nested functions."""
    chunk_type = "method" if class_name else "function"

    if class_name:
        qname = f"{module}.{class_name}.{node.name}"
    else:
        qname = f"{module}.{node.name}"

    chunk_id = make_chunk_id(file_path, chunk_type, qname)
    body_text = _get_source_segment(lines, node)
    sig = _get_signature(node, lines)
    docstring = ast.get_docstring(node)
    decorators = _get_decorator_names(node)
    imports = _extract_imports(node)

    chunks.append({
        "id": chunk_id,
        "file_path": file_path,
        "file_hash": fhash,
        "chunk_type": chunk_type,
        "qualified_name": qname,
        "module_name": module,
        "signature": sig,
        "docstring": docstring,
        "body": body_text,
        "decorators": decorators or None,
        "imports": imports,
        "line_start": node.lineno,
        "line_end": node.end_lineno or node.lineno,
        "parent_chunk_id": parent_chunk_id,
        "class_name": class_name,
    })

    # Handle nested functions
    for child in ast.iter_child_nodes(node):
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
            nested_qname = f"{qname}.{child.name}"
            nested_id = make_chunk_id(file_path, "function", nested_qname)
            nested_body = _get_source_segment(lines, child)
            nested_sig = _get_signature(child, lines)
            nested_doc = ast.get_docstring(child)
            nested_dec = _get_decorator_names(child)
            nested_imports = _extract_imports(child)

            chunks.append({
                "id": nested_id,
                "file_path": file_path,
                "file_hash": fhash,
                "chunk_type": "function",
                "qualified_name": nested_qname,
                "module_name": module,
                "signature": nested_sig,
                "docstring": nested_doc,
                "body": nested_body,
                "decorators": nested_dec or None,
                "imports": nested_imports,
                "line_start": child.lineno,
                "line_end": child.end_lineno or child.lineno,
                "parent_chunk_id": chunk_id,
                "class_name": class_name,
            })


def _parse_class(
    node: ast.ClassDef,
    file_path: str,
    fhash: str,
    module: str,
    lines: list[str],
    chunks: list[dict],
) -> None:
    """Parse a class node: the class itself + all its methods."""
    qname = f"{module}.{node.name}"
    chunk_id = make_chunk_id(file_path, "class", qname)
    body_text = _get_source_segment(lines, node)
    docstring = ast.get_docstring(node)
    decorators = _get_decorator_names(node)

    # Class-level imports (rare but possible)
    imports = _extract_imports(node)

    chunks.append({
        "id": chunk_id,
        "file_path": file_path,
        "file_hash": fhash,
        "chunk_type": "class",
        "qualified_name": qname,
        "module_name": module,
        "signature": f"class {node.name}:",
        "docstring": docstring,
        "body": body_text,
        "decorators": decorators or None,
        "imports": imports,
        "line_start": node.lineno,
        "line_end": node.end_lineno or node.lineno,
        "parent_chunk_id": None,
        "class_name": None,
    })

    # Parse methods
    for child in ast.iter_child_nodes(node):
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
            _parse_function(child, file_path, fhash, module, lines, chunks,
                            parent_chunk_id=chunk_id, class_name=node.name)


# ---------------------------------------------------------------------------
# Database operations
# ---------------------------------------------------------------------------

def get_existing_file_hash(conn, file_path: str) -> str | None:
    """Check if we already have chunks for this file and return the stored hash."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT DISTINCT file_hash FROM code_chunks WHERE file_path = %s LIMIT 1",
            (file_path,),
        )
        row = cur.fetchone()
        return row[0] if row else None


def upsert_chunks(conn, chunks: list[dict[str, Any]]) -> int:
    """Upsert code chunks into the database. Returns number of upserted rows."""
    if not chunks:
        return 0

    upserted = 0
    with conn.cursor() as cur:
        for chunk in chunks:
            cur.execute(
                """
                INSERT INTO code_chunks (
                    id, file_path, file_hash, chunk_type, qualified_name,
                    module_name, signature, docstring, body, decorators,
                    imports, line_start, line_end, parent_chunk_id,
                    class_name, project_id, subproject_id,
                    created_at, updated_at
                ) VALUES (
                    %(id)s, %(file_path)s, %(file_hash)s, %(chunk_type)s, %(qualified_name)s,
                    %(module_name)s, %(signature)s, %(docstring)s, %(body)s, %(decorators)s,
                    %(imports)s, %(line_start)s, %(line_end)s, %(parent_chunk_id)s,
                    %(class_name)s, %(project_id)s, %(subproject_id)s,
                    now(), now()
                )
                ON CONFLICT (id) DO UPDATE SET
                    file_hash = EXCLUDED.file_hash,
                    signature = EXCLUDED.signature,
                    docstring = EXCLUDED.docstring,
                    body = EXCLUDED.body,
                    decorators = EXCLUDED.decorators,
                    imports = EXCLUDED.imports,
                    line_start = EXCLUDED.line_start,
                    line_end = EXCLUDED.line_end,
                    parent_chunk_id = EXCLUDED.parent_chunk_id,
                    updated_at = now()
                """,
                {
                    **chunk,
                    "imports": Jsonb(chunk.get("imports") or []),
                    "project_id": chunk.get("project_id"),
                    "subproject_id": chunk.get("subproject_id"),
                },
            )
            upserted += 1
    conn.commit()
    return upserted


def remove_stale_chunks(conn, file_path: str, current_chunk_ids: set[str]) -> int:
    """Remove chunks that no longer exist in the parsed file (renamed/deleted functions)."""
    with conn.cursor() as cur:
        if not current_chunk_ids:
            cur.execute("DELETE FROM code_chunks WHERE file_path = %s", (file_path,))
        else:
            placeholders = ",".join(["%s"] * len(current_chunk_ids))
            cur.execute(
                f"DELETE FROM code_chunks WHERE file_path = %s AND id NOT IN ({placeholders})",
                (file_path, *current_chunk_ids),
            )
        removed = cur.rowcount
    conn.commit()
    return removed


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

def discover_python_files(target: str) -> list[str]:
    """Find all .py files under a directory, or return the single file if given."""
    target_path = Path(target).resolve()
    if target_path.is_file() and target_path.suffix == ".py":
        return [str(target_path)]
    if target_path.is_dir():
        files = []
        # Only check path components *below* the target for hidden/cache dirs
        target_depth = len(target_path.parts)
        for py_file in sorted(target_path.rglob("*.py")):
            relative_parts = py_file.parts[target_depth:]
            if any(p.startswith(".") or p == "__pycache__" for p in relative_parts):
                continue
            files.append(str(py_file.resolve()))
        return files
    LOGGER.warning("Target is not a .py file or directory: %s", target)
    return []


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def process_file(
    conn,
    file_path: str,
    *,
    project_id: str | None = None,
    subproject_id: str | None = None,
    force: bool = False,
) -> dict[str, int]:
    """
    Parse and upsert a single file. Returns stats dict.
    Skips if file hash is unchanged (unless force=True).
    """
    stats = {"parsed": 0, "upserted": 0, "removed": 0, "skipped": 0}

    try:
        content = Path(file_path).read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as e:
        LOGGER.warning("Cannot read %s: %s", file_path, e)
        return stats

    current_hash = _file_hash(content)

    if not force:
        existing_hash = get_existing_file_hash(conn, file_path)
        if existing_hash == current_hash:
            stats["skipped"] = 1
            return stats

    chunks = parse_file(file_path, content)
    if not chunks:
        LOGGER.info("No chunks extracted from %s", file_path)
        return stats

    # Attach project metadata
    for chunk in chunks:
        chunk["project_id"] = project_id
        chunk["subproject_id"] = subproject_id

    stats["parsed"] = len(chunks)
    current_ids = {c["id"] for c in chunks}

    stats["upserted"] = upsert_chunks(conn, chunks)
    stats["removed"] = remove_stale_chunks(conn, file_path, current_ids)

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Parse Python files into code chunks for the Memory Engine."
    )
    parser.add_argument("target", help="Directory or .py file to parse")
    parser.add_argument("--project-id", default=None, help="Project ID to associate chunks with")
    parser.add_argument("--subproject-id", default=None, help="Subproject ID")
    parser.add_argument("--force", action="store_true", help="Re-parse even if file hash unchanged")
    parser.add_argument("--dry-run", action="store_true", help="Parse but don't write to DB")
    args = parser.parse_args()

    files = discover_python_files(args.target)
    if not files:
        LOGGER.error("No Python files found at: %s", args.target)
        sys.exit(1)

    LOGGER.info("Found %d Python files to process", len(files))

    total_stats = {"parsed": 0, "upserted": 0, "removed": 0, "skipped": 0, "files": 0, "errors": 0}

    if args.dry_run:
        # Dry run: parse only, no DB
        for fp in files:
            try:
                content = Path(fp).read_text(encoding="utf-8")
                chunks = parse_file(fp, content)
                total_stats["parsed"] += len(chunks)
                total_stats["files"] += 1
                for c in chunks:
                    print(f"  {c['chunk_type']:8s} {c['qualified_name']} (L{c['line_start']}-{c['line_end']})")
            except Exception as e:
                LOGGER.error("Error parsing %s: %s", fp, e)
                total_stats["errors"] += 1
    else:
        conn = psycopg.connect(DB_DSN)
        try:
            for fp in files:
                try:
                    stats = process_file(
                        conn, fp,
                        project_id=args.project_id,
                        subproject_id=args.subproject_id,
                        force=args.force,
                    )
                    total_stats["parsed"] += stats["parsed"]
                    total_stats["upserted"] += stats["upserted"]
                    total_stats["removed"] += stats["removed"]
                    total_stats["skipped"] += stats["skipped"]
                    total_stats["files"] += 1

                    if stats["skipped"]:
                        LOGGER.debug("SKIP %s (unchanged)", fp)
                    else:
                        LOGGER.info("PARSED %s: %d chunks upserted, %d stale removed",
                                    fp, stats["upserted"], stats["removed"])
                except Exception as e:
                    LOGGER.error("Error processing %s: %s", fp, e)
                    total_stats["errors"] += 1
        finally:
            conn.close()

    print(f"FILES={total_stats['files']}")
    print(f"PARSED={total_stats['parsed']}")
    print(f"UPSERTED={total_stats['upserted']}")
    print(f"REMOVED={total_stats['removed']}")
    print(f"SKIPPED={total_stats['skipped']}")
    print(f"ERRORS={total_stats['errors']}")


if __name__ == "__main__":
    main()
