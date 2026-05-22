"""
classify_compile_errors.py — UE Build.sh error classifier for the Memory Engine.

L8-S1 of the Level 8 "Autonomous Closed-Loop Execution" implementation.

Parses raw Unreal Engine Build.sh output (Clang or MSVC) and maps each error
back to specific code chunk IDs in the database. Classifies errors by category
and identifies the root cause when multiple errors cascade from a single issue.

Supports:
    - Clang error format:   file.cpp:47:5: error: use of undeclared identifier 'x'
    - MSVC error format:    file.cpp(47): error C2065: 'x': undeclared identifier
    - UE macro expansion errors (UPROPERTY, UCLASS, GENERATED_BODY)
    - Template instantiation chains (collapses to root cause)
    - Linker errors (unresolved external symbol)
    - #include errors (file not found)
    - Note/warning context lines attached to their parent error

Output: structured JSON with chunk IDs, error categories, root cause analysis,
and suggested fix strategies for each error group.

Usage:
    python classify_compile_errors.py build.log [--json] [--dsn DSN]
    cat build.log | python classify_compile_errors.py - [--json]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import psycopg

LOGGER = logging.getLogger("openclaw.classify_compile_errors")
logging.basicConfig(
    level=os.environ.get("OPENCLAW_LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)

DB_DSN = os.environ.get("OPENCLAW_MEMORY_DSN", "dbname=openclaw_memory")


# ---------------------------------------------------------------------------
# Error categories
# ---------------------------------------------------------------------------

class ErrorCategory(str, Enum):
    UNDECLARED_IDENTIFIER = "undeclared_identifier"
    TYPE_MISMATCH = "type_mismatch"
    MISSING_INCLUDE = "missing_include"
    MISSING_MEMBER = "missing_member"
    MISSING_OVERRIDE = "missing_override"
    TEMPLATE_ERROR = "template_error"
    MACRO_EXPANSION = "macro_expansion"
    LINKER_ERROR = "linker_error"
    SYNTAX_ERROR = "syntax_error"
    REDEFINITION = "redefinition"
    ACCESS_CONTROL = "access_control"
    CONVERSION_ERROR = "conversion_error"
    ARGUMENT_MISMATCH = "argument_mismatch"
    MISSING_RETURN = "missing_return"
    DEPRECATION = "deprecation"
    UNKNOWN = "unknown"


# Mapping of error message patterns to categories
CATEGORY_PATTERNS: list[tuple[re.Pattern, ErrorCategory]] = [
    (re.compile(r"use of undeclared identifier", re.I), ErrorCategory.UNDECLARED_IDENTIFIER),
    (re.compile(r"undeclared identifier", re.I), ErrorCategory.UNDECLARED_IDENTIFIER),
    (re.compile(r"was not declared in this scope", re.I), ErrorCategory.UNDECLARED_IDENTIFIER),
    (re.compile(r"unknown type name", re.I), ErrorCategory.UNDECLARED_IDENTIFIER),
    (re.compile(r"no type named", re.I), ErrorCategory.UNDECLARED_IDENTIFIER),
    (re.compile(r"no member named", re.I), ErrorCategory.MISSING_MEMBER),
    (re.compile(r"has no member", re.I), ErrorCategory.MISSING_MEMBER),
    (re.compile(r"is not a member of", re.I), ErrorCategory.MISSING_MEMBER),
    (re.compile(r"member access into incomplete type", re.I), ErrorCategory.MISSING_MEMBER),
    (re.compile(r"cannot convert", re.I), ErrorCategory.TYPE_MISMATCH),
    (re.compile(r"no viable conversion", re.I), ErrorCategory.TYPE_MISMATCH),
    (re.compile(r"incompatible type", re.I), ErrorCategory.TYPE_MISMATCH),
    (re.compile(r"cannot initialize .* with", re.I), ErrorCategory.TYPE_MISMATCH),
    (re.compile(r"implicit conversion", re.I), ErrorCategory.CONVERSION_ERROR),
    (re.compile(r"no matching function", re.I), ErrorCategory.ARGUMENT_MISMATCH),
    (re.compile(r"no matching constructor", re.I), ErrorCategory.ARGUMENT_MISMATCH),
    (re.compile(r"too (many|few) arguments", re.I), ErrorCategory.ARGUMENT_MISMATCH),
    (re.compile(r"candidate function not viable", re.I), ErrorCategory.ARGUMENT_MISMATCH),
    (re.compile(r"file not found", re.I), ErrorCategory.MISSING_INCLUDE),
    (re.compile(r"no such file or directory", re.I), ErrorCategory.MISSING_INCLUDE),
    (re.compile(r"fatal error:.*not found", re.I), ErrorCategory.MISSING_INCLUDE),
    (re.compile(r"template", re.I), ErrorCategory.TEMPLATE_ERROR),
    (re.compile(r"macro.*expand", re.I), ErrorCategory.MACRO_EXPANSION),
    (re.compile(r"expanded from macro", re.I), ErrorCategory.MACRO_EXPANSION),
    (re.compile(r"UPROPERTY|UCLASS|UFUNCTION|USTRUCT|GENERATED_BODY", re.I), ErrorCategory.MACRO_EXPANSION),
    (re.compile(r"unresolved external", re.I), ErrorCategory.LINKER_ERROR),
    (re.compile(r"undefined reference", re.I), ErrorCategory.LINKER_ERROR),
    (re.compile(r"undefined symbol", re.I), ErrorCategory.LINKER_ERROR),
    (re.compile(r"redefinition of", re.I), ErrorCategory.REDEFINITION),
    (re.compile(r"already defined", re.I), ErrorCategory.REDEFINITION),
    (re.compile(r"previous definition", re.I), ErrorCategory.REDEFINITION),
    (re.compile(r"is (a )?private member", re.I), ErrorCategory.ACCESS_CONTROL),
    (re.compile(r"is (a )?protected member", re.I), ErrorCategory.ACCESS_CONTROL),
    (re.compile(r"inaccessible", re.I), ErrorCategory.ACCESS_CONTROL),
    (re.compile(r"must override", re.I), ErrorCategory.MISSING_OVERRIDE),
    (re.compile(r"marked 'override' but does not", re.I), ErrorCategory.MISSING_OVERRIDE),
    (re.compile(r"hides overloaded virtual", re.I), ErrorCategory.MISSING_OVERRIDE),
    (re.compile(r"expected.*[;{})]", re.I), ErrorCategory.SYNTAX_ERROR),
    (re.compile(r"expected expression", re.I), ErrorCategory.SYNTAX_ERROR),
    (re.compile(r"expected.*before", re.I), ErrorCategory.SYNTAX_ERROR),
    (re.compile(r"extraneous.*before", re.I), ErrorCategory.SYNTAX_ERROR),
    (re.compile(r"non-void function.*not return", re.I), ErrorCategory.MISSING_RETURN),
    (re.compile(r"control reaches end of non-void", re.I), ErrorCategory.MISSING_RETURN),
    (re.compile(r"deprecated", re.I), ErrorCategory.DEPRECATION),
]

# Fix strategy hints per category
FIX_STRATEGIES: dict[ErrorCategory, str] = {
    ErrorCategory.UNDECLARED_IDENTIFIER: "Add missing #include or forward declaration; check spelling of type/variable name",
    ErrorCategory.TYPE_MISMATCH: "Check type compatibility; add explicit cast or correct the type",
    ErrorCategory.MISSING_INCLUDE: "Add the missing #include directive; verify the header file exists in the module",
    ErrorCategory.MISSING_MEMBER: "Verify member exists in the class declaration; check spelling; ensure header is included",
    ErrorCategory.MISSING_OVERRIDE: "Add or correct 'override' specifier; verify base class virtual method signature matches exactly",
    ErrorCategory.TEMPLATE_ERROR: "Check template parameter types; verify template specialization exists; look at the 'required from' chain for the actual usage site",
    ErrorCategory.MACRO_EXPANSION: "Check UPROPERTY/UFUNCTION specifier syntax; ensure GENERATED_BODY() is present in UCLASS; verify .generated.h is included",
    ErrorCategory.LINKER_ERROR: "Implement the declared function; check Build.cs module dependencies; verify the source file is included in the build",
    ErrorCategory.SYNTAX_ERROR: "Check for missing semicolons, braces, or parentheses near the reported line",
    ErrorCategory.REDEFINITION: "Remove duplicate definition; use #pragma once or include guards; check for conflicting headers",
    ErrorCategory.ACCESS_CONTROL: "Change member access level or use friend declaration; access through public interface instead",
    ErrorCategory.CONVERSION_ERROR: "Add explicit conversion operator or static_cast; check if implicit conversion is intended",
    ErrorCategory.ARGUMENT_MISMATCH: "Check function signature; verify argument count and types match the declaration",
    ErrorCategory.MISSING_RETURN: "Add return statement to all code paths in the function",
    ErrorCategory.DEPRECATION: "Replace deprecated API with recommended alternative (check UE migration guide)",
    ErrorCategory.UNKNOWN: "Examine the full error context; may require manual analysis",
}


# ---------------------------------------------------------------------------
# Parsed error record
# ---------------------------------------------------------------------------

@dataclass
class CompileError:
    """A single parsed compiler diagnostic."""
    file_path: str
    line: int
    column: int | None
    severity: str          # error, warning, note, fatal error
    message: str
    raw_text: str
    category: ErrorCategory = ErrorCategory.UNKNOWN
    chunk_id: str | None = None
    chunk_qualified_name: str | None = None
    chunk_type: str | None = None
    is_template_instantiation: bool = False
    is_macro_expansion: bool = False
    notes: list[str] = field(default_factory=list)


@dataclass
class ErrorGroup:
    """A group of related errors sharing a root cause."""
    root_error: CompileError
    related_errors: list[CompileError] = field(default_factory=list)
    category: ErrorCategory = ErrorCategory.UNKNOWN
    fix_strategy: str = ""
    is_cascade: bool = False

    @property
    def all_chunk_ids(self) -> list[str]:
        ids = []
        if self.root_error.chunk_id:
            ids.append(self.root_error.chunk_id)
        for e in self.related_errors:
            if e.chunk_id and e.chunk_id not in ids:
                ids.append(e.chunk_id)
        return ids


# ---------------------------------------------------------------------------
# Line parsers (Clang + MSVC)
# ---------------------------------------------------------------------------

# Clang: /path/to/file.cpp:47:5: error: message
RE_CLANG = re.compile(
    r"^(?P<file>[^\s:]+\.\w+):(?P<line>\d+):(?P<col>\d+):\s+"
    r"(?P<severity>error|warning|note|fatal error):\s+(?P<msg>.+)$"
)

# Clang without column: /path/to/file.cpp:47: error: message
RE_CLANG_NO_COL = re.compile(
    r"^(?P<file>[^\s:]+\.\w+):(?P<line>\d+):\s+"
    r"(?P<severity>error|warning|note|fatal error):\s+(?P<msg>.+)$"
)

# MSVC: file.cpp(47): error C2065: message
RE_MSVC = re.compile(
    r"^(?P<file>[^\s(]+\.\w+)\((?P<line>\d+)\):\s+"
    r"(?P<severity>error|warning|note|fatal error)\s*(?:C\d+)?:\s+(?P<msg>.+)$"
)

# Linker: symbol "void __cdecl Foo::Bar(float)" referenced from ...
RE_LINKER_SYMBOL = re.compile(
    r"(?:undefined reference to|unresolved external symbol)\s+[\"']?(?P<symbol>[^\"'\n]+)")

# Template instantiation context: "in instantiation of ..." or "required from ..."
RE_TEMPLATE_CONTEXT = re.compile(
    r"(?:in instantiation of|required from|instantiated from)", re.I
)

# Macro expansion context: "expanded from macro"
RE_MACRO_CONTEXT = re.compile(r"expanded from macro", re.I)

# UE Build.sh: Lines starting with timestamps or [N/M] progress
RE_UE_BUILD_NOISE = re.compile(
    r"^(?:\[\d+/\d+\]|Building |Compiling |Linking |---|\s*$|Total |WARNING:|LogInit)"
)


# Linker error lines: ld.lld: error: ... or ld: error: ...
RE_LINKER_LINE = re.compile(
    r"^(?:ld(?:\.lld)?|LINK)\s*:\s*(?P<severity>error|warning):\s+(?P<msg>.+)$"
)


def parse_line(line: str) -> CompileError | None:
    """Parse a single compiler output line into a CompileError, or None."""
    line = line.rstrip()

    if RE_UE_BUILD_NOISE.match(line):
        return None

    for regex in (RE_CLANG, RE_CLANG_NO_COL, RE_MSVC):
        m = regex.match(line)
        if m:
            col = int(m.group("col")) if "col" in m.groupdict() and m.group("col") else None
            err = CompileError(
                file_path=m.group("file"),
                line=int(m.group("line")),
                column=col,
                severity=m.group("severity"),
                message=m.group("msg").strip(),
                raw_text=line,
            )
            err.is_template_instantiation = bool(RE_TEMPLATE_CONTEXT.search(line))
            err.is_macro_expansion = bool(RE_MACRO_CONTEXT.search(line))
            return err

    # Linker errors (no file/line)
    m = RE_LINKER_LINE.match(line)
    if m:
        return CompileError(
            file_path="<linker>",
            line=0,
            column=None,
            severity=m.group("severity"),
            message=m.group("msg").strip(),
            raw_text=line,
        )

    return None


def classify_error(error: CompileError) -> ErrorCategory:
    """Classify an error message into a category."""
    for pattern, category in CATEGORY_PATTERNS:
        if pattern.search(error.message):
            return category
    return ErrorCategory.UNKNOWN


# ---------------------------------------------------------------------------
# Parse full build log
# ---------------------------------------------------------------------------

def parse_build_log(log_text: str) -> list[CompileError]:
    """Parse a full Build.sh log into a list of CompileErrors."""
    errors: list[CompileError] = []
    lines = log_text.splitlines()

    for line in lines:
        err = parse_line(line)
        if err:
            err.category = classify_error(err)
            errors.append(err)

    return errors


# ---------------------------------------------------------------------------
# Group related errors (template chains, macro expansions, cascades)
# ---------------------------------------------------------------------------

def group_errors(errors: list[CompileError]) -> list[ErrorGroup]:
    """
    Group related errors into ErrorGroups.

    Strategy:
    1. Notes/warnings immediately following an error attach to that error.
    2. Template instantiation chains collapse to the instantiation site.
    3. Multiple errors on the same file+line merge into one group.
    4. Remaining errors become their own groups.
    """
    if not errors:
        return []

    groups: list[ErrorGroup] = []
    current_group: ErrorGroup | None = None

    for err in errors:
        if err.severity in ("error", "fatal error"):
            # Start a new group
            if current_group is not None:
                groups.append(current_group)
            current_group = ErrorGroup(
                root_error=err,
                category=err.category,
                fix_strategy=FIX_STRATEGIES.get(err.category, ""),
            )

        elif err.severity == "note" and current_group is not None:
            # Attach note to current group
            current_group.related_errors.append(err)
            current_group.root_error.notes.append(err.message)

            # If this note reveals a template or macro context, update the group
            if err.is_template_instantiation:
                current_group.category = ErrorCategory.TEMPLATE_ERROR
                current_group.fix_strategy = FIX_STRATEGIES[ErrorCategory.TEMPLATE_ERROR]
            elif err.is_macro_expansion:
                current_group.category = ErrorCategory.MACRO_EXPANSION
                current_group.fix_strategy = FIX_STRATEGIES[ErrorCategory.MACRO_EXPANSION]

        elif err.severity == "warning":
            # Warnings become standalone groups if no current error context
            if current_group is not None and _same_location(current_group.root_error, err):
                current_group.related_errors.append(err)
            else:
                if current_group is not None:
                    groups.append(current_group)
                current_group = ErrorGroup(
                    root_error=err,
                    category=err.category,
                    fix_strategy=FIX_STRATEGIES.get(err.category, ""),
                )

    if current_group is not None:
        groups.append(current_group)

    # Detect cascade: multiple errors in the same file likely share a root cause
    _detect_cascades(groups)

    return groups


def _same_location(a: CompileError, b: CompileError) -> bool:
    """Check if two errors are at the same file and line."""
    return a.file_path == b.file_path and a.line == b.line


def _detect_cascades(groups: list[ErrorGroup]) -> None:
    """
    Mark later errors in the same file as likely cascades of the first error.

    Heuristic: if file F has errors at lines L1 < L2 < L3, and L1 is a
    MISSING_INCLUDE or UNDECLARED_IDENTIFIER, then L2 and L3 are likely
    cascades (a single missing #include causes dozens of downstream errors).
    """
    file_first_error: dict[str, int] = {}  # file_path -> index of first error group

    for i, group in enumerate(groups):
        fp = group.root_error.file_path
        if fp not in file_first_error:
            file_first_error[fp] = i
        else:
            first_idx = file_first_error[fp]
            first_cat = groups[first_idx].category
            # Missing includes and undeclared identifiers commonly cascade
            if first_cat in (
                ErrorCategory.MISSING_INCLUDE,
                ErrorCategory.UNDECLARED_IDENTIFIER,
                ErrorCategory.SYNTAX_ERROR,
            ):
                group.is_cascade = True


# ---------------------------------------------------------------------------
# Chunk ID resolution
# ---------------------------------------------------------------------------

def resolve_chunk_ids(conn, errors: list[CompileError]) -> int:
    """
    Map each error's (file_path, line) to a code_chunks row.

    Returns the number of errors that were successfully resolved.
    """
    # Collect unique file paths
    file_paths = {e.file_path for e in errors}
    if not file_paths:
        return 0

    # Build a lookup: file_path -> sorted list of (line_start, line_end, chunk_id, qualified_name, chunk_type)
    chunk_lookup: dict[str, list[tuple[int, int, str, str, str]]] = {}

    with conn.cursor() as cur:
        # Try both absolute paths and basename matching
        placeholders = ",".join(["%s"] * len(file_paths))
        cur.execute(f"""
            SELECT file_path, line_start, line_end, id, qualified_name, chunk_type
            FROM code_chunks
            WHERE file_path IN ({placeholders})
               OR file_path LIKE ANY(ARRAY[{','.join(['%s'] * len(file_paths))}])
            ORDER BY file_path, line_start
        """, (*file_paths, *[f"%/{Path(fp).name}" for fp in file_paths]))

        for row in cur.fetchall():
            fp, ls, le, cid, qn, ct = row
            chunk_lookup.setdefault(fp, []).append((ls, le, cid, qn, ct))

    resolved = 0
    for err in errors:
        # Try exact path match first, then basename match
        chunks = chunk_lookup.get(err.file_path)
        if not chunks:
            basename = Path(err.file_path).name
            for fp, cl in chunk_lookup.items():
                if fp.endswith(f"/{basename}") or Path(fp).name == basename:
                    chunks = cl
                    break

        if not chunks:
            continue

        # Find the most specific (smallest) chunk containing this line
        best_match = None
        best_span = float("inf")

        for line_start, line_end, chunk_id, qname, ctype in chunks:
            if line_start <= err.line <= line_end:
                span = line_end - line_start
                # Prefer non-module chunks (more specific)
                if ctype == "module":
                    span += 100000  # Deprioritize module-level match
                if span < best_span:
                    best_span = span
                    best_match = (chunk_id, qname, ctype)

        if best_match:
            err.chunk_id = best_match[0]
            err.chunk_qualified_name = best_match[1]
            err.chunk_type = best_match[2]
            resolved += 1

    return resolved


def resolve_linker_symbols(conn, groups: list[ErrorGroup]) -> int:
    """
    Resolve linker errors by matching symbol names to function qualified_names.

    Linker errors don't have file/line info, so we match the symbol name
    against the code_chunks qualified_name column.
    """
    resolved = 0

    for group in groups:
        if group.category != ErrorCategory.LINKER_ERROR:
            continue

        err = group.root_error
        m = RE_LINKER_SYMBOL.search(err.message)
        if not m:
            continue

        symbol = m.group("symbol").strip()
        # Extract function/method name from mangled or demangled symbol
        # e.g., "void AMyCharacter::Attack(float)" -> "Attack"
        func_match = re.search(r"(?:\w+::)?(\w+)\s*\(", symbol)
        if not func_match:
            continue

        func_name = func_match.group(1)

        # Also try to extract class name
        class_match = re.search(r"(\w+)::" + func_name, symbol)
        class_name = class_match.group(1) if class_match else None

        with conn.cursor() as cur:
            if class_name:
                cur.execute("""
                    SELECT id, qualified_name, chunk_type, file_path, line_start
                    FROM code_chunks
                    WHERE qualified_name LIKE %s AND chunk_type IN ('function', 'method')
                    LIMIT 1
                """, (f"%.{class_name}.{func_name}",))
            else:
                cur.execute("""
                    SELECT id, qualified_name, chunk_type, file_path, line_start
                    FROM code_chunks
                    WHERE qualified_name LIKE %s AND chunk_type IN ('function', 'method')
                    LIMIT 1
                """, (f"%.{func_name}",))

            row = cur.fetchone()
            if row:
                err.chunk_id = row[0]
                err.chunk_qualified_name = row[1]
                err.chunk_type = row[2]
                err.file_path = row[3]
                err.line = row[4]
                resolved += 1

    return resolved


# ---------------------------------------------------------------------------
# Full classification pipeline
# ---------------------------------------------------------------------------

def classify_build_log(
    log_text: str,
    *,
    dsn: str = DB_DSN,
    resolve_chunks: bool = True,
) -> dict[str, Any]:
    """
    Full classification pipeline: parse → classify → group → resolve chunks.

    Returns a structured dict with all error groups, chunk mappings, and
    summary statistics.
    """
    errors = parse_build_log(log_text)
    groups = group_errors(errors)

    chunk_resolved = 0
    linker_resolved = 0

    if resolve_chunks and errors:
        try:
            conn = psycopg.connect(dsn)
            try:
                chunk_resolved = resolve_chunk_ids(conn, errors)
                linker_resolved = resolve_linker_symbols(conn, groups)
            finally:
                conn.close()
        except Exception as e:
            LOGGER.warning("Could not connect to database for chunk resolution: %s", e)

    # Build summary
    error_count = sum(1 for g in groups if g.root_error.severity in ("error", "fatal error"))
    warning_count = sum(1 for g in groups if g.root_error.severity == "warning")
    cascade_count = sum(1 for g in groups if g.is_cascade)

    category_counts: dict[str, int] = {}
    for g in groups:
        cat = g.category.value
        category_counts[cat] = category_counts.get(cat, 0) + 1

    chunk_ids_affected = []
    for g in groups:
        for cid in g.all_chunk_ids:
            if cid not in chunk_ids_affected:
                chunk_ids_affected.append(cid)

    result = {
        "summary": {
            "total_diagnostics": len(errors),
            "error_groups": len(groups),
            "errors": error_count,
            "warnings": warning_count,
            "cascades": cascade_count,
            "unique_root_causes": error_count - cascade_count,
            "chunk_ids_resolved": chunk_resolved + linker_resolved,
            "chunks_affected": chunk_ids_affected,
            "category_breakdown": category_counts,
        },
        "groups": [
            {
                "category": g.category.value,
                "is_cascade": g.is_cascade,
                "fix_strategy": g.fix_strategy,
                "root_error": {
                    "file": g.root_error.file_path,
                    "line": g.root_error.line,
                    "column": g.root_error.column,
                    "severity": g.root_error.severity,
                    "message": g.root_error.message,
                    "chunk_id": g.root_error.chunk_id,
                    "chunk_name": g.root_error.chunk_qualified_name,
                    "chunk_type": g.root_error.chunk_type,
                    "notes": g.root_error.notes,
                },
                "related_count": len(g.related_errors),
                "all_chunk_ids": g.all_chunk_ids,
            }
            for g in groups
        ],
    }

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Classify UE Build.sh compile errors and map to code chunks."
    )
    parser.add_argument("logfile", help="Build log file (or '-' for stdin)")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--dsn", default=DB_DSN, help="PostgreSQL DSN")
    parser.add_argument("--no-resolve", action="store_true",
                        help="Skip chunk ID resolution (no DB required)")
    args = parser.parse_args()

    if args.logfile == "-":
        log_text = sys.stdin.read()
    else:
        log_text = Path(args.logfile).read_text(encoding="utf-8", errors="replace")

    result = classify_build_log(
        log_text,
        dsn=args.dsn,
        resolve_chunks=not args.no_resolve,
    )

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        s = result["summary"]
        print(f"DIAGNOSTICS={s['total_diagnostics']}")
        print(f"ERROR_GROUPS={s['error_groups']}")
        print(f"ERRORS={s['errors']}")
        print(f"WARNINGS={s['warnings']}")
        print(f"CASCADES={s['cascades']}")
        print(f"ROOT_CAUSES={s['unique_root_causes']}")
        print(f"CHUNKS_RESOLVED={s['chunk_ids_resolved']}")
        print()

        for g in result["groups"]:
            re_ = g["root_error"]
            cascade = " [CASCADE]" if g["is_cascade"] else ""
            chunk = f" → {re_['chunk_name']} ({re_['chunk_type']})" if re_["chunk_id"] else ""
            print(f"  [{g['category']}]{cascade} {re_['file']}:{re_['line']}{chunk}")
            print(f"    {re_['message']}")
            if g["fix_strategy"]:
                print(f"    FIX: {g['fix_strategy']}")
            if re_["notes"]:
                for note in re_["notes"][:3]:
                    print(f"    NOTE: {note}")
            print()


if __name__ == "__main__":
    main()
