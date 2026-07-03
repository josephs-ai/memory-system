"""
task_context_assembler.py — Task-scoped context assembly for Builder agents.

L8-S3 of the Level 8 "Autonomous Closed-Loop Execution" implementation.

Unlike the generic context_hydrator (which does query-based retrieval), this
module assembles targeted context packs optimized for a specific modification
task. It prioritizes:

1. The exact target chunk(s) being modified (full body)
2. Structural interfaces: header declarations, class signatures, base classes
3. Direct callers and callees of the target (signature + docstring)
4. Compile error context (if errors are provided)
5. Patch history for the target file (recent modifications)

For C++ modifications, header files (.h/.hpp) are prioritized over
implementation files (.cpp) for dependencies — the Builder needs to see
*what* a dependency exposes, not *how* it's implemented.

Token budgeting is strict and configurable. Every payload section has a
max char budget. The assembler fills sections in priority order and stops
when the total budget is reached.

Usage:
    python task_context_assembler.py --target "MyCharacter.AMyCharacter.Attack" --task "fix Health typo"
    python task_context_assembler.py --target-file "/path/to/file.cpp" --target-line 47 --task "fix error"
    python task_context_assembler.py --error-log build.log --task "fix all compile errors"
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import psycopg

LOGGER = logging.getLogger("openclaw.task_context_assembler")
logging.basicConfig(
    level=os.environ.get("OPENCLAW_LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

DB_DSN = (os.environ.get("OPENCLAW_MEMORY_DSN") or os.environ.get("OPENCLAW_MEMORY_DB_DSN") or "dbname=openclaw_memory")  # honor both canonical DSN env names

# ---------------------------------------------------------------------------
# Token budgets (chars, ~4 chars per token)
# ---------------------------------------------------------------------------

@dataclass
class TokenBudget:
    """Configurable token budget for context assembly."""
    total_max_chars: int = 24000       # ~6K tokens total budget
    target_body_max: int = 6000        # Full body of the target being modified
    target_class_max: int = 4000       # Class declaration containing the target
    header_interface_max: int = 3000   # Header/interface of dependencies
    callers_max: int = 2000            # Callers of the target (sig + docstring)
    callees_max: int = 2000            # Callees of the target (sig + docstring)
    sibling_methods_max: int = 2000    # Other methods in the same class
    error_context_max: int = 2000      # Compile error info if provided
    patch_history_max: int = 1000      # Recent patches for this file
    memory_context_max: int = 1500     # Related memory items
    imports_max: int = 500             # Import/include context


# ---------------------------------------------------------------------------
# Context sections
# ---------------------------------------------------------------------------

@dataclass
class ContextSection:
    """A single section of the assembled context."""
    label: str
    priority: int           # Lower = higher priority
    content: str
    char_count: int = 0
    source: str = ""        # Where this came from (e.g., "code_chunks", "neo4j")
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        self.char_count = len(self.content)


@dataclass
class TaskContext:
    """Complete assembled context for a Builder task."""
    task_description: str
    target_qualified_name: str | None
    target_file: str | None
    sections: list[ContextSection] = field(default_factory=list)
    total_chars: int = 0
    total_tokens_approx: int = 0
    assembly_ms: float = 0.0
    budget: TokenBudget = field(default_factory=TokenBudget)

    def to_prompt_block(self) -> str:
        """Format as a single prompt injection block for the Builder agent."""
        parts = []
        parts.append(f"## Task: {self.task_description}")
        if self.target_qualified_name:
            parts.append(f"## Target: {self.target_qualified_name}")
        parts.append("")

        for section in sorted(self.sections, key=lambda s: s.priority):
            parts.append(f"### {section.label}")
            parts.append(section.content)
            parts.append("")

        return "\n".join(parts)

    def to_dict(self) -> dict[str, Any]:
        """Serialize for JSON output."""
        return {
            "task_description": self.task_description,
            "target_qualified_name": self.target_qualified_name,
            "target_file": self.target_file,
            "total_chars": self.total_chars,
            "total_tokens_approx": self.total_tokens_approx,
            "assembly_ms": self.assembly_ms,
            "sections": [
                {
                    "label": s.label,
                    "priority": s.priority,
                    "char_count": s.char_count,
                    "source": s.source,
                }
                for s in sorted(self.sections, key=lambda s: s.priority)
            ],
            "prompt_block": self.to_prompt_block(),
        }


# ---------------------------------------------------------------------------
# Task Context Assembler
# ---------------------------------------------------------------------------

class TaskContextAssembler:
    """
    Assembles targeted context packs for Builder agents.

    Priority-ordered assembly:
    1. Target chunk (full body) — what's being modified
    2. Target class declaration — the containing structure
    3. Header interfaces of dependencies — what the target calls/uses
    4. Direct callers — who calls this function
    5. Direct callees — what this function calls
    6. Sibling methods in same class
    7. Compile errors (if provided)
    8. Patch history
    9. Related memory items
    10. Import/include context
    """

    def __init__(
        self,
        conn: psycopg.Connection,
        budget: TokenBudget | None = None,
    ):
        self._conn = conn
        self._budget = budget or TokenBudget()
        self._chars_used = 0

    def assemble(
        self,
        task_description: str,
        *,
        target_qualified_name: str | None = None,
        target_file: str | None = None,
        target_line: int | None = None,
        error_groups: list[dict] | None = None,
    ) -> TaskContext:
        """
        Assemble a complete task-scoped context pack.

        Provide either target_qualified_name or (target_file + target_line).
        If error_groups is provided (from classify_compile_errors), errors are
        included in the context.
        """
        t0 = time.monotonic()
        self._chars_used = 0

        ctx = TaskContext(
            task_description=task_description,
            target_qualified_name=target_qualified_name,
            target_file=target_file,
            budget=self._budget,
        )

        # Resolve target chunk if only file+line given
        target_chunk = None
        if target_qualified_name:
            target_chunk = self._fetch_chunk_by_name(target_qualified_name)
        elif target_file and target_line:
            target_chunk = self._fetch_chunk_by_location(target_file, target_line)

        if target_chunk:
            ctx.target_qualified_name = target_chunk["qualified_name"]
            ctx.target_file = target_chunk["file_path"]

        # --- Priority 1: Target chunk (full body) ---
        if target_chunk:
            self._add_target_body(ctx, target_chunk)

        # --- Priority 2: Containing class declaration ---
        if target_chunk and target_chunk.get("class_name"):
            self._add_class_declaration(ctx, target_chunk)

        # --- Priority 3: Header interfaces of dependencies ---
        if target_chunk:
            self._add_dependency_interfaces(ctx, target_chunk)

        # --- Priority 4 & 5: Callers and callees ---
        if target_chunk and target_chunk["chunk_type"] in ("function", "method"):
            self._add_callers(ctx, target_chunk)
            self._add_callees(ctx, target_chunk)

        # --- Priority 6: Sibling methods ---
        if target_chunk and target_chunk.get("class_name"):
            self._add_sibling_methods(ctx, target_chunk)

        # --- Priority 7: Compile errors ---
        if error_groups:
            self._add_error_context(ctx, error_groups)

        # --- Priority 8: Patch history ---
        if target_file or (target_chunk and target_chunk.get("file_path")):
            fp = target_file or target_chunk["file_path"]
            self._add_patch_history(ctx, fp)

        # --- Priority 9: Related memory items ---
        if target_chunk:
            self._add_memory_context(ctx, target_chunk)

        # --- Priority 10: Import context ---
        if target_chunk:
            self._add_import_context(ctx, target_chunk)

        # Finalize
        ctx.total_chars = sum(s.char_count for s in ctx.sections)
        ctx.total_tokens_approx = ctx.total_chars // 4
        ctx.assembly_ms = round((time.monotonic() - t0) * 1000, 1)

        return ctx

    def _budget_remaining(self) -> int:
        return max(0, self._budget.total_max_chars - self._chars_used)

    def _try_add_section(
        self, ctx: TaskContext, label: str, content: str,
        priority: int, source: str, max_chars: int,
        metadata: dict | None = None,
    ) -> bool:
        """Try to add a section within budget. Returns True if added."""
        if not content or not content.strip():
            return False

        remaining = self._budget_remaining()
        if remaining <= 0:
            return False

        effective_max = min(max_chars, remaining)
        if len(content) > effective_max:
            content = content[:effective_max] + "\n... [truncated]"

        section = ContextSection(
            label=label,
            priority=priority,
            content=content,
            source=source,
            metadata=metadata or {},
        )
        ctx.sections.append(section)
        self._chars_used += section.char_count
        return True

    # --- Section builders ---

    def _add_target_body(self, ctx: TaskContext, chunk: dict) -> None:
        """Add the full body of the target chunk."""
        body = chunk.get("body", "")
        sig = chunk.get("signature", "")
        doc = chunk.get("docstring", "")

        parts = []
        if sig:
            parts.append(f"// Signature: {sig}")
        if doc:
            parts.append(f"// Docstring: {doc}")
        parts.append(f"// File: {chunk['file_path']} (L{chunk['line_start']}-{chunk['line_end']})")
        parts.append(f"// Chunk: {chunk['qualified_name']} ({chunk['chunk_type']})")
        parts.append("")
        parts.append(body)

        self._try_add_section(
            ctx, f"Target: {chunk['qualified_name']}", "\n".join(parts),
            priority=1, source="code_chunks",
            max_chars=self._budget.target_body_max,
        )

    def _add_class_declaration(self, ctx: TaskContext, chunk: dict) -> None:
        """Add the class declaration containing the target."""
        class_name = chunk.get("class_name")
        module = chunk.get("module_name")
        if not class_name:
            return

        class_qname = f"{module}.{class_name}"
        class_chunk = self._fetch_chunk_by_name(class_qname, chunk_type="class")
        if not class_chunk:
            return

        # For the class, we want the declaration/interface, not full body
        body = class_chunk.get("body", "")
        sig = class_chunk.get("signature", "")

        parts = [f"// Class declaration: {class_qname}"]
        if sig:
            parts.append(f"// {sig}")
        parts.append(f"// File: {class_chunk['file_path']} (L{class_chunk['line_start']}-{class_chunk['line_end']})")
        parts.append("")
        parts.append(body)

        self._try_add_section(
            ctx, f"Class: {class_qname}", "\n".join(parts),
            priority=2, source="code_chunks",
            max_chars=self._budget.target_class_max,
        )

    def _add_dependency_interfaces(self, ctx: TaskContext, chunk: dict) -> None:
        """
        Add header/interface files for dependencies.

        For C++: prefer .h/.hpp files over .cpp for dependencies.
        For Python: include the module-level signatures.
        """
        chunk.get("imports") or []
        file_path = chunk.get("file_path", "")

        # Also get callees to find their source files
        callees = self._get_neo4j_callees(chunk["qualified_name"])
        dep_files = set()

        for callee in callees:
            callee_chunk = self._fetch_chunk_by_name(callee)
            if callee_chunk:
                dep_file = callee_chunk["file_path"]
                # For C++: if the callee is in a .cpp, look for the corresponding .h
                header = self._find_header_for(dep_file)
                if header:
                    dep_files.add(header)
                elif dep_file != file_path:
                    dep_files.add(dep_file)

        if not dep_files:
            return

        parts = []
        budget_per_file = max(500, self._budget.header_interface_max // max(len(dep_files), 1))

        for dep_file in sorted(dep_files)[:5]:  # Cap at 5 dependency files
            interface = self._build_file_interface(dep_file, budget_per_file)
            if interface:
                parts.append(f"// --- {Path(dep_file).name} ---")
                parts.append(interface)
                parts.append("")

        if parts:
            self._try_add_section(
                ctx, "Dependency Interfaces", "\n".join(parts),
                priority=3, source="code_chunks",
                max_chars=self._budget.header_interface_max,
            )

    def _add_callers(self, ctx: TaskContext, chunk: dict) -> None:
        """Add functions that call the target (signature + docstring)."""
        callers = self._get_neo4j_callers(chunk["qualified_name"])
        if not callers:
            return

        parts = []
        for caller_qname in callers[:8]:
            caller = self._fetch_chunk_by_name(caller_qname)
            if caller:
                sig = caller.get("signature", caller["qualified_name"])
                doc = caller.get("docstring", "")
                parts.append(f"  {sig}")
                if doc:
                    parts.append(f"    // {doc[:100]}")

        if parts:
            self._try_add_section(
                ctx, f"Called by ({len(callers)} callers)", "\n".join(parts),
                priority=4, source="neo4j",
                max_chars=self._budget.callers_max,
            )

    def _add_callees(self, ctx: TaskContext, chunk: dict) -> None:
        """Add functions that the target calls (signature + docstring)."""
        callees = self._get_neo4j_callees(chunk["qualified_name"])
        if not callees:
            return

        parts = []
        for callee_qname in callees[:8]:
            callee = self._fetch_chunk_by_name(callee_qname)
            if callee:
                sig = callee.get("signature", callee["qualified_name"])
                doc = callee.get("docstring", "")
                parts.append(f"  {sig}")
                if doc:
                    parts.append(f"    // {doc[:100]}")

        if parts:
            self._try_add_section(
                ctx, f"Calls ({len(callees)} callees)", "\n".join(parts),
                priority=5, source="neo4j",
                max_chars=self._budget.callees_max,
            )

    def _add_sibling_methods(self, ctx: TaskContext, chunk: dict) -> None:
        """Add other methods in the same class (sig only)."""
        class_name = chunk.get("class_name")
        module = chunk.get("module_name")
        if not class_name:
            return

        with self._conn.cursor() as cur:
            cur.execute("""
                SELECT qualified_name, signature, chunk_type
                FROM code_chunks
                WHERE module_name = %s AND class_name = %s
                  AND chunk_type IN ('method', 'function')
                  AND qualified_name != %s
                ORDER BY line_start
            """, (module, class_name, chunk["qualified_name"]))
            siblings = cur.fetchall()

        if not siblings:
            return

        parts = []
        for qname, sig, ctype in siblings:
            parts.append(f"  {sig or qname}")

        self._try_add_section(
            ctx, f"Sibling methods in {class_name} ({len(siblings)})", "\n".join(parts),
            priority=6, source="code_chunks",
            max_chars=self._budget.sibling_methods_max,
        )

    def _add_error_context(self, ctx: TaskContext, error_groups: list[dict]) -> None:
        """Add compile error context from the error classifier."""
        parts = []
        for i, group in enumerate(error_groups[:10]):
            root = group.get("root_error", {})
            cat = group.get("category", "unknown")
            cascade = " [CASCADE]" if group.get("is_cascade") else ""
            chunk_name = root.get("chunk_name", "unknown")

            parts.append(f"  Error {i+1}: [{cat}]{cascade}")
            parts.append(f"    {root.get('file', '?')}:{root.get('line', '?')}: {root.get('message', '?')}")
            if chunk_name:
                parts.append(f"    Chunk: {chunk_name}")
            fix = group.get("fix_strategy", "")
            if fix:
                parts.append(f"    Fix: {fix}")
            for note in root.get("notes", [])[:2]:
                parts.append(f"    Note: {note}")
            parts.append("")

        self._try_add_section(
            ctx, f"Compile Errors ({len(error_groups)})", "\n".join(parts),
            priority=7, source="error_classifier",
            max_chars=self._budget.error_context_max,
        )

    def _add_patch_history(self, ctx: TaskContext, file_path: str) -> None:
        """Add recent patch history for the file."""
        try:
            from patch_ledger import PatchLedger
            ledger = PatchLedger(self._conn)
            patches = ledger.get_file_history(file_path, limit=5)
        except Exception:
            return

        if not patches:
            return

        parts = []
        for p in patches:
            rb = " [ROLLED BACK]" if p.rolled_back else ""
            parts.append(f"  [{p.step_id}] {p.operation}{rb} ({p.created_at.strftime('%Y-%m-%d %H:%M')})")

        self._try_add_section(
            ctx, f"Recent patches for {Path(file_path).name}", "\n".join(parts),
            priority=8, source="patch_ledger",
            max_chars=self._budget.patch_history_max,
        )

    def _add_memory_context(self, ctx: TaskContext, chunk: dict) -> None:
        """Add related memory items (decisions, bug history, etc.)."""
        qname = chunk["qualified_name"]

        # Query for ABOUT_CODE edges from Neo4j
        try:
            from context_hydrator import _get_neo4j
            driver = _get_neo4j()
            with driver.session() as session:
                result = session.run("""
                    MATCH (m:MemoryItem)-[r:ABOUT_CODE]->(target)
                    WHERE target.qualified_name = $qname
                       OR target.qualified_name CONTAINS $module
                    RETURN m.id as memory_id, m.memory_type as category,
                           r.confidence as confidence
                    ORDER BY r.confidence DESC
                    LIMIT 5
                """, qname=qname, module=chunk.get("module_name", ""))
                memory_ids = [(r["memory_id"], r["category"], r["confidence"]) for r in result]
        except Exception:
            return

        if not memory_ids:
            return

        # Fetch memory item text from PostgreSQL
        parts = []
        with self._conn.cursor() as cur:
            for mem_id, category, confidence in memory_ids:
                cur.execute(
                    "SELECT text FROM memory_items WHERE id = %s",
                    (mem_id,)
                )
                row = cur.fetchone()
                if row and row[0]:
                    text = row[0][:300]
                    parts.append(f"  [{category}] (conf={confidence:.2f}): {text}")

        if parts:
            self._try_add_section(
                ctx, "Related Memory", "\n".join(parts),
                priority=9, source="memory_items+neo4j",
                max_chars=self._budget.memory_context_max,
            )

    def _add_import_context(self, ctx: TaskContext, chunk: dict) -> None:
        """Add import/include information for the target file."""
        file_path = chunk.get("file_path", "")
        chunk.get("module_name", "")

        # Get the module chunk which has the imports list
        with self._conn.cursor() as cur:
            cur.execute("""
                SELECT imports FROM code_chunks
                WHERE file_path = %s AND chunk_type = 'module'
                LIMIT 1
            """, (file_path,))
            row = cur.fetchone()

        if not row or not row[0]:
            return

        imports = row[0]
        parts = [f"  {imp}" for imp in imports[:20]]

        self._try_add_section(
            ctx, f"Imports ({len(imports)})", "\n".join(parts),
            priority=10, source="code_chunks",
            max_chars=self._budget.imports_max,
        )

    # --- Data access helpers ---

    def _fetch_chunk_by_name(
        self, qualified_name: str, chunk_type: str | None = None,
    ) -> dict | None:
        """Fetch a code chunk by qualified_name."""
        with self._conn.cursor() as cur:
            if chunk_type:
                cur.execute("""
                    SELECT id, file_path, file_hash, chunk_type, qualified_name,
                           module_name, signature, docstring, body, decorators,
                           imports, line_start, line_end, parent_chunk_id, class_name
                    FROM code_chunks
                    WHERE qualified_name = %s AND chunk_type = %s
                    LIMIT 1
                """, (qualified_name, chunk_type))
            else:
                cur.execute("""
                    SELECT id, file_path, file_hash, chunk_type, qualified_name,
                           module_name, signature, docstring, body, decorators,
                           imports, line_start, line_end, parent_chunk_id, class_name
                    FROM code_chunks
                    WHERE qualified_name = %s
                    ORDER BY CASE chunk_type
                        WHEN 'method' THEN 1 WHEN 'function' THEN 2
                        WHEN 'class' THEN 3 ELSE 4 END
                    LIMIT 1
                """, (qualified_name,))

            row = cur.fetchone()
            if not row:
                return None

            cols = [
                "id", "file_path", "file_hash", "chunk_type", "qualified_name",
                "module_name", "signature", "docstring", "body", "decorators",
                "imports", "line_start", "line_end", "parent_chunk_id", "class_name",
            ]
            return dict(zip(cols, row))

    def _fetch_chunk_by_location(self, file_path: str, line: int) -> dict | None:
        """Fetch the most specific chunk containing a given line."""
        with self._conn.cursor() as cur:
            cur.execute("""
                SELECT id, file_path, file_hash, chunk_type, qualified_name,
                       module_name, signature, docstring, body, decorators,
                       imports, line_start, line_end, parent_chunk_id, class_name
                FROM code_chunks
                WHERE file_path = %s AND line_start <= %s AND line_end >= %s
                ORDER BY (line_end - line_start) ASC,
                         CASE chunk_type WHEN 'module' THEN 999 ELSE 0 END ASC
                LIMIT 1
            """, (file_path, line, line))

            row = cur.fetchone()
            if not row:
                return None

            cols = [
                "id", "file_path", "file_hash", "chunk_type", "qualified_name",
                "module_name", "signature", "docstring", "body", "decorators",
                "imports", "line_start", "line_end", "parent_chunk_id", "class_name",
            ]
            return dict(zip(cols, row))

    def _build_file_interface(self, file_path: str, max_chars: int) -> str | None:
        """
        Build a compact interface view of a file.
        For headers: full content (they're declarations).
        For implementation files: only signatures + docstrings.
        """
        suffix = Path(file_path).suffix.lower()
        is_header = suffix in (".h", ".hpp", ".hh", ".hxx")

        if is_header:
            # Headers are interfaces — include more content
            with self._conn.cursor() as cur:
                cur.execute("""
                    SELECT body FROM code_chunks
                    WHERE file_path = %s AND chunk_type = 'module'
                    LIMIT 1
                """, (file_path,))
                row = cur.fetchone()
                if row and row[0]:
                    return row[0][:max_chars]
            return None

        # For .cpp/.py files: only signatures
        with self._conn.cursor() as cur:
            cur.execute("""
                SELECT qualified_name, signature, docstring, chunk_type
                FROM code_chunks
                WHERE file_path = %s AND chunk_type != 'module'
                ORDER BY line_start
            """, (file_path,))
            rows = cur.fetchall()

        if not rows:
            return None

        parts = []
        chars = 0
        for qname, sig, doc, ctype in rows:
            line = sig or qname
            if doc:
                line += f"  // {doc[:80]}"
            if chars + len(line) > max_chars:
                parts.append("  ... [truncated]")
                break
            parts.append(f"  {line}")
            chars += len(line)

        return "\n".join(parts)

    def _find_header_for(self, file_path: str) -> str | None:
        """Find the corresponding header file for a .cpp file."""
        p = Path(file_path)
        if p.suffix.lower() in (".h", ".hpp", ".hh", ".hxx"):
            return file_path  # Already a header

        if p.suffix.lower() not in (".cpp", ".cc", ".cxx"):
            return None

        # Check for corresponding header in the same directory
        for ext in (".h", ".hpp", ".hh"):
            header = p.with_suffix(ext)
            # Check if we have chunks for this header
            with self._conn.cursor() as cur:
                cur.execute(
                    "SELECT 1 FROM code_chunks WHERE file_path = %s LIMIT 1",
                    (str(header),)
                )
                if cur.fetchone():
                    return str(header)

        return None

    def _get_neo4j_callers(self, qualified_name: str) -> list[str]:
        """Get functions that CALL the target."""
        try:
            from context_hydrator import _get_neo4j
            driver = _get_neo4j()
            with driver.session() as session:
                result = session.run("""
                    MATCH (caller:Function)-[:CALLS]->(target:Function {qualified_name: $qname})
                    RETURN caller.qualified_name as qname
                    LIMIT 10
                """, qname=qualified_name)
                return [r["qname"] for r in result]
        except Exception:
            return []

    def _get_neo4j_callees(self, qualified_name: str) -> list[str]:
        """Get functions that the target CALLS."""
        try:
            from context_hydrator import _get_neo4j
            driver = _get_neo4j()
            with driver.session() as session:
                result = session.run("""
                    MATCH (source:Function {qualified_name: $qname})-[:CALLS]->(target:Function)
                    RETURN target.qualified_name as qname
                    LIMIT 10
                """, qname=qualified_name)
                return [r["qname"] for r in result]
        except Exception:
            return []


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Assemble task-scoped context for Builder agents."
    )
    parser.add_argument("--task", required=True, help="Task description")
    parser.add_argument("--target", default=None, help="Target qualified_name")
    parser.add_argument("--target-file", default=None, help="Target file path")
    parser.add_argument("--target-line", type=int, default=None, help="Target line number")
    parser.add_argument("--error-log", default=None, help="Build error log file")
    parser.add_argument("--budget", type=int, default=24000, help="Total char budget")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--prompt", action="store_true", help="Output as prompt block")
    args = parser.parse_args()

    budget = TokenBudget(total_max_chars=args.budget)
    error_groups = None

    if args.error_log:
        from classify_compile_errors import classify_build_log
        log_text = Path(args.error_log).read_text(encoding="utf-8", errors="replace")
        result = classify_build_log(log_text)
        error_groups = result.get("groups", [])

    conn = psycopg.connect(DB_DSN)
    try:
        assembler = TaskContextAssembler(conn, budget=budget)
        ctx = assembler.assemble(
            args.task,
            target_qualified_name=args.target,
            target_file=args.target_file,
            target_line=args.target_line,
            error_groups=error_groups,
        )

        if args.prompt:
            print(ctx.to_prompt_block())
        elif args.json:
            print(json.dumps(ctx.to_dict(), indent=2, default=str))
        else:
            print(f"TASK: {ctx.task_description}")
            print(f"TARGET: {ctx.target_qualified_name}")
            print(f"TOTAL_CHARS={ctx.total_chars}")
            print(f"TOTAL_TOKENS≈{ctx.total_tokens_approx}")
            print(f"ASSEMBLY_MS={ctx.assembly_ms}")
            print(f"SECTIONS={len(ctx.sections)}")
            print()
            for s in sorted(ctx.sections, key=lambda s: s.priority):
                print(f"  [{s.priority}] {s.label} ({s.char_count} chars, {s.source})")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
