"""
parse_code_cpp.py — Tree-sitter-based C++ parser for the code index.

L7-S4 of the Level 7 "Autonomous Enterprise" implementation.

Parses C++ source files (.cpp, .h, .hpp, .cc, .cxx) into code chunks
using tree-sitter for accurate AST parsing. Produces the same chunk schema
as the Python parser, integrating seamlessly into the existing Qdrant,
PostgreSQL, and Neo4j pipelines.

Handles Unreal Engine conventions:
    - UCLASS, USTRUCT, UENUM macros (extracted as decorators)
    - UPROPERTY, UFUNCTION macros
    - GENERATED_BODY() markers
    - Namespaces as module-level scope
    - Method implementations outside class body (ClassName::MethodName)

Usage:
    python parse_code_cpp.py /path/to/source [--project-id my_project] [--dry-run]

Features:
    - Tree-sitter AST parsing (accurate, fast, no regex heuristics)
    - Extracts: classes, structs, functions, methods, namespaces
    - Handles out-of-class method definitions (Foo::Bar style)
    - Extracts #include directives as imports
    - Extracts comments preceding definitions as docstrings
    - Unreal Engine macro awareness (UCLASS, UPROPERTY, etc.)
    - Same database schema as Python parser — zero pipeline changes needed
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any

import psycopg
import tree_sitter_cpp as tscpp
from tree_sitter import Language, Parser

LOGGER = logging.getLogger("openclaw.parse_code_cpp")
logging.basicConfig(
    level=os.environ.get("OPENCLAW_LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from base_code_parser import BaseCodeParser

DB_DSN = (os.environ.get("OPENCLAW_MEMORY_DSN") or os.environ.get("OPENCLAW_MEMORY_DB_DSN") or "dbname=openclaw_memory")  # honor both canonical DSN env names

# Unreal Engine macros to extract as decorators
UE_CLASS_MACROS = {"UCLASS", "USTRUCT", "UENUM", "UINTERFACE"}
UE_MEMBER_MACROS = {"UPROPERTY", "UFUNCTION", "GENERATED_BODY"}

# C++ file extensions
CPP_EXTENSIONS = [".cpp", ".h", ".hpp", ".cc", ".cxx", ".hh", ".hxx"]


class CppParser(BaseCodeParser):
    """Tree-sitter-based C++ code parser."""

    def __init__(self):
        self._language = Language(tscpp.language())
        self._parser = Parser(self._language)

    @property
    def language_id(self) -> str:
        return "cpp"

    @property
    def file_extensions(self) -> list[str]:
        return CPP_EXTENSIONS

    def _preprocess_content(self, content: str) -> str:
        """
        Strip C++ export macros that confuse tree-sitter.
        Handles patterns like: class MYGAME_API AMyCharacter
        by removing the ALL_CAPS_API token between 'class'/'struct' and the type name.
        """
        import re
        # Remove DLL export macros (e.g., MYGAME_API, ENGINE_API, CORE_API)
        # These appear between class/struct keyword and the class name
        content = re.sub(
            r'\b(class|struct)\s+([A-Z][A-Z0-9_]*_API)\s+',
            r'\1 ',
            content,
        )
        return content

    def parse_file(self, file_path: str, content: str) -> list[dict[str, Any]]:
        """Parse a C++ file into code chunks."""
        try:
            preprocessed = self._preprocess_content(content)
            tree = self._parser.parse(preprocessed.encode("utf-8"))
        except Exception as e:
            LOGGER.warning("Parse error for %s: %s", file_path, e)
            return []

        # Use preprocessed content for AST traversal but original for hashing
        lines = preprocessed.splitlines()
        fhash = self.file_hash(content)
        module = self.module_name_from_path(file_path)
        chunks: list[dict[str, Any]] = []

        # --- Module chunk (entire file) ---
        includes = self._extract_includes(tree.root_node, preprocessed)
        file_comment = self._extract_leading_comment(tree.root_node, lines)

        chunks.append(self.make_chunk(
            file_path=file_path,
            file_hash=fhash,
            chunk_type="module",
            qualified_name=module,
            module_name=module,
            body=content,
            line_start=1,
            line_end=max(len(lines), 1),
            docstring=file_comment,
            imports=includes,
        ))

        # --- Walk top-level declarations ---
        self._walk_declarations(
            tree.root_node, file_path, fhash, module, lines, preprocessed, chunks,
            namespace_prefix=None, parent_class=None, parent_chunk_id=None,
        )

        return chunks

    def _walk_declarations(
        self,
        node,
        file_path: str,
        fhash: str,
        module: str,
        lines: list[str],
        content: str,
        chunks: list[dict],
        *,
        namespace_prefix: str | None,
        parent_class: str | None,
        parent_chunk_id: str | None,
    ) -> None:
        """Recursively walk AST nodes and extract chunks."""
        children = list(node.children)
        pending_ue_macros: list[str] = []

        for i, child in enumerate(children):
            ntype = child.type

            # Collect UE macros from ERROR nodes (UCLASS(), USTRUCT(), etc.)
            # tree-sitter doesn't understand these macros, so they appear as ERROR
            if ntype == "ERROR":
                macro_text = self._node_text(child, content).strip()
                for macro in UE_CLASS_MACROS:
                    if macro_text.startswith(macro):
                        pending_ue_macros.append(macro_text)
                        break
                continue

            if ntype == "namespace_definition":
                pending_ue_macros.clear()
                self._parse_namespace(
                    child, file_path, fhash, module, lines, content, chunks,
                    namespace_prefix=namespace_prefix,
                )

            elif ntype in ("class_specifier", "struct_specifier"):
                decorators = list(pending_ue_macros)
                pending_ue_macros.clear()
                self._parse_class(
                    child, file_path, fhash, module, lines, content, chunks,
                    namespace_prefix=namespace_prefix,
                    decorators=decorators or None,
                )

            elif ntype == "function_definition":
                pending_ue_macros.clear()
                self._parse_function(
                    child, file_path, fhash, module, lines, content, chunks,
                    namespace_prefix=namespace_prefix,
                    parent_class=parent_class,
                    parent_chunk_id=parent_chunk_id,
                )

            elif ntype == "declaration":
                pending_ue_macros.clear()
                # Could be a function declaration (prototype)
                # We skip pure declarations and only index definitions with bodies
                pass

            elif ntype == "template_declaration":
                decorators = ["template"] + list(pending_ue_macros)
                pending_ue_macros.clear()
                # Template wrapper — check the inner declaration
                for inner in child.children:
                    if inner.type in ("function_definition", "class_specifier", "struct_specifier"):
                        if inner.type == "function_definition":
                            self._parse_function(
                                inner, file_path, fhash, module, lines, content, chunks,
                                namespace_prefix=namespace_prefix,
                                parent_class=parent_class,
                                parent_chunk_id=parent_chunk_id,
                                decorators=decorators,
                            )
                        else:
                            self._parse_class(
                                inner, file_path, fhash, module, lines, content, chunks,
                                namespace_prefix=namespace_prefix,
                                decorators=decorators,
                            )
            else:
                # Non-matching node — clear pending macros
                if ntype not in (";", "comment"):
                    pending_ue_macros.clear()

    def _parse_namespace(
        self, node, file_path, fhash, module, lines, content, chunks,
        *, namespace_prefix,
    ) -> None:
        """Parse a namespace definition."""
        ns_name = None
        for child in node.children:
            if child.type == "namespace_identifier":
                ns_name = self._node_text(child, content)
                break

        if not ns_name:
            # Anonymous namespace — skip prefix
            ns_name = ""

        full_prefix = f"{namespace_prefix}::{ns_name}" if namespace_prefix and ns_name else (namespace_prefix or ns_name)

        # Recurse into the namespace body
        for child in node.children:
            if child.type == "declaration_list":
                self._walk_declarations(
                    child, file_path, fhash, module, lines, content, chunks,
                    namespace_prefix=full_prefix or None,
                    parent_class=None,
                    parent_chunk_id=None,
                )

    def _parse_class(
        self, node, file_path, fhash, module, lines, content, chunks,
        *, namespace_prefix=None, decorators=None,
    ) -> None:
        """Parse a class or struct definition."""
        class_name = None
        for child in node.children:
            if child.type == "type_identifier":
                class_name = self._node_text(child, content)
                break

        if not class_name:
            return  # Anonymous class/struct

        # Build qualified name
        if namespace_prefix:
            qname = f"{module}.{namespace_prefix}::{class_name}"
        else:
            qname = f"{module}.{class_name}"

        body_text = self._node_source(node, lines)
        docstring = self._extract_preceding_comment(node, lines)
        sig = f"{'struct' if node.type == 'struct_specifier' else 'class'} {class_name}"

        # Extract base classes
        for child in node.children:
            if child.type == "base_class_clause":
                sig += " : " + self._node_text(child, content).lstrip(": ")
                break

        # Extract UE macros as decorators
        all_decorators = list(decorators or [])
        all_decorators.extend(self._extract_ue_macros(node, content))

        chunk_id = self.make_chunk_id(file_path, "class", qname)
        chunks.append(self.make_chunk(
            file_path=file_path,
            file_hash=fhash,
            chunk_type="class",
            qualified_name=qname,
            module_name=module,
            body=body_text,
            line_start=node.start_point[0] + 1,
            line_end=node.end_point[0] + 1,
            signature=sig,
            docstring=docstring,
            decorators=all_decorators or None,
        ))

        # Parse methods inside the class body
        for child in node.children:
            if child.type == "field_declaration_list":
                for member in child.children:
                    if member.type == "function_definition":
                        self._parse_function(
                            member, file_path, fhash, module, lines, content, chunks,
                            namespace_prefix=namespace_prefix,
                            parent_class=class_name,
                            parent_chunk_id=chunk_id,
                        )
                    elif member.type == "declaration":
                        # Check for inline method declarations with bodies
                        for inner in member.children:
                            if inner.type == "function_definition":
                                self._parse_function(
                                    inner, file_path, fhash, module, lines, content, chunks,
                                    namespace_prefix=namespace_prefix,
                                    parent_class=class_name,
                                    parent_chunk_id=chunk_id,
                                )

    def _parse_function(
        self, node, file_path, fhash, module, lines, content, chunks,
        *, namespace_prefix=None, parent_class=None, parent_chunk_id=None,
        decorators=None,
    ) -> None:
        """Parse a function or method definition."""
        # Extract function name — handle qualified names (ClassName::MethodName)
        func_name = None
        resolved_class = parent_class

        declarator = self._find_child(node, "function_declarator")
        if not declarator:
            declarator = self._find_child(node, "pointer_declarator")
            if declarator:
                declarator = self._find_child(declarator, "function_declarator")

        if declarator:
            # Check for qualified identifier (Foo::Bar)
            qualified = self._find_child(declarator, "qualified_identifier")
            if qualified:
                func_name = self._node_text(qualified, content)
                # Extract class name from scope
                scope = self._find_child(qualified, "namespace_identifier")
                if not scope:
                    scope = self._find_child(qualified, "template_type")
                if scope:
                    resolved_class = self._node_text(scope, content)
                # Get just the function name part
                name_node = self._find_child(qualified, "identifier") or self._find_child(qualified, "destructor_name")
                if name_node:
                    func_name = self._node_text(name_node, content)
            else:
                # Try identifier, field_identifier (inline class methods), or destructor_name
                name_node = (
                    self._find_child(declarator, "identifier")
                    or self._find_child(declarator, "field_identifier")
                    or self._find_child(declarator, "destructor_name")
                )
                if name_node:
                    func_name = self._node_text(name_node, content)

        if not func_name:
            return  # Can't identify function

        # Build chunk_type and qualified_name
        is_method = resolved_class is not None
        chunk_type = "method" if is_method else "function"

        if is_method:
            qname = f"{module}.{resolved_class}.{func_name}"
        elif namespace_prefix:
            qname = f"{module}.{namespace_prefix}::{func_name}"
        else:
            qname = f"{module}.{func_name}"

        body_text = self._node_source(node, lines)
        sig = self._extract_signature(node, lines)
        docstring = self._extract_preceding_comment(node, lines)

        all_decorators = list(decorators or [])
        all_decorators.extend(self._extract_ue_macros(node, content))

        # Find parent class chunk if this is an out-of-class method definition
        if is_method and not parent_chunk_id:
            class_qname = f"{module}.{resolved_class}"
            parent_chunk_id = self.make_chunk_id(file_path, "class", class_qname)

        chunks.append(self.make_chunk(
            file_path=file_path,
            file_hash=fhash,
            chunk_type=chunk_type,
            qualified_name=qname,
            module_name=module,
            body=body_text,
            line_start=node.start_point[0] + 1,
            line_end=node.end_point[0] + 1,
            signature=sig,
            docstring=docstring,
            decorators=all_decorators or None,
            parent_chunk_id=parent_chunk_id,
            class_name=resolved_class,
        ))

    # --- Extraction helpers ---

    def _node_text(self, node, content: str) -> str:
        """Get the text of a node from the content bytes."""
        return content[node.start_byte:node.end_byte]

    def _node_source(self, node, lines: list[str]) -> str:
        """Get the source lines for a node."""
        start = node.start_point[0]
        end = node.end_point[0] + 1
        return "\n".join(lines[start:end])

    def _find_child(self, node, child_type: str):
        """Find the first child of a given type."""
        for child in node.children:
            if child.type == child_type:
                return child
        return None

    def _extract_includes(self, root, content: str) -> list[str]:
        """Extract #include directives as import names."""
        includes = []
        for child in root.children:
            if child.type == "preproc_include":
                path_node = self._find_child(child, "string_literal") or self._find_child(child, "system_lib_string")
                if path_node:
                    inc_path = self._node_text(path_node, content).strip('"<>')
                    includes.append(inc_path)
        return sorted(set(includes))

    def _extract_signature(self, func_node, lines: list[str]) -> str:
        """Extract the function signature (everything before the body)."""
        body_node = self._find_child(func_node, "compound_statement")
        if body_node:
            # Signature is from function start to just before the body
            sig_start = func_node.start_point[0]
            sig_end = body_node.start_point[0]
            sig_lines = lines[sig_start:sig_end + 1]
            sig = "\n".join(sig_lines).strip()
            # Remove the opening brace if it's on the sig line
            if sig.endswith("{"):
                sig = sig[:-1].strip()
            return sig
        return self._node_source(func_node, lines).split("\n")[0]

    def _extract_preceding_comment(self, node, lines: list[str]) -> str | None:
        """Extract comment block immediately preceding a node as docstring."""
        start_line = node.start_point[0]
        if start_line == 0:
            return None

        comment_lines = []
        for i in range(start_line - 1, max(start_line - 20, -1), -1):
            line = lines[i].strip()
            if line.startswith("//") or line.startswith("/*") or line.startswith("*") or line.endswith("*/"):
                comment_lines.insert(0, line)
            elif line == "":
                continue  # Allow blank lines between comment and definition
            else:
                break

        if not comment_lines:
            return None

        # Clean comment markers
        cleaned = []
        for line in comment_lines:
            line = line.lstrip("/").lstrip("*").lstrip(" ")
            if line.rstrip() == "":
                continue
            cleaned.append(line.rstrip())

        return "\n".join(cleaned) if cleaned else None

    def _extract_leading_comment(self, root, lines: list[str]) -> str | None:
        """Extract the file-level leading comment."""
        for child in root.children:
            if child.type == "comment":
                return self._extract_preceding_comment(child, lines)
            if child.type != "preproc_include":
                break
        return None

    def _extract_ue_macros(self, node, content: str) -> list[str]:
        """Extract Unreal Engine macros (UCLASS, UPROPERTY, etc.) as decorators."""
        macros = []
        for child in node.children:
            if child.type == "expression_statement":
                text = self._node_text(child, content).strip().rstrip(";")
                for macro in UE_CLASS_MACROS | UE_MEMBER_MACROS:
                    if text.startswith(macro):
                        macros.append(text)
                        break
            elif child.type == "field_declaration_list":
                for member in child.children:
                    if member.type == "expression_statement":
                        text = self._node_text(member, content).strip().rstrip(";")
                        for macro in UE_MEMBER_MACROS:
                            if text.startswith(macro):
                                macros.append(text)
                                break
        return macros


# ---------------------------------------------------------------------------
# Database operations (reuse from parse_code_ast)
# ---------------------------------------------------------------------------

def process_cpp_file(
    conn,
    parser: CppParser,
    file_path: str,
    *,
    project_id: str | None = None,
    force: bool = False,
) -> dict[str, int]:
    """Parse and upsert a single C++ file."""
    from parse_code_ast import (
        get_existing_file_hash,
        upsert_chunks,
        remove_stale_chunks,
    )

    stats = {"parsed": 0, "upserted": 0, "removed": 0, "skipped": 0}

    try:
        content = Path(file_path).read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        LOGGER.warning("Cannot read %s: %s", file_path, e)
        return stats

    current_hash = parser.file_hash(content)

    if not force:
        existing = get_existing_file_hash(conn, file_path)
        if existing == current_hash:
            stats["skipped"] = 1
            return stats

    chunks = parser.parse_file(file_path, content)
    if not chunks:
        return stats

    for chunk in chunks:
        chunk["project_id"] = project_id
        chunk["subproject_id"] = None

    stats["parsed"] = len(chunks)
    current_ids = {c["id"] for c in chunks}

    stats["upserted"] = upsert_chunks(conn, chunks)
    stats["removed"] = remove_stale_chunks(conn, file_path, current_ids)

    return stats


def discover_cpp_files(target: str) -> list[str]:
    """Find all C++ files under a directory."""
    target_path = Path(target).resolve()
    if target_path.is_file() and target_path.suffix.lower() in CPP_EXTENSIONS:
        return [str(target_path)]
    if target_path.is_dir():
        files = []
        target_depth = len(target_path.parts)
        for ext in CPP_EXTENSIONS:
            for f in sorted(target_path.rglob(f"*{ext}")):
                relative_parts = f.parts[target_depth:]
                if any(p.startswith(".") or p in ("__pycache__", "Intermediate", "Saved") for p in relative_parts):
                    continue
                files.append(str(f.resolve()))
        return sorted(set(files))
    return []


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Parse C++ files into code chunks for the Memory Engine."
    )
    parser.add_argument("target", help="Directory or C++ file to parse")
    parser.add_argument("--project-id", default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    files = discover_cpp_files(args.target)
    if not files:
        LOGGER.error("No C++ files found at: %s", args.target)
        sys.exit(1)

    LOGGER.info("Found %d C++ files to process", len(files))

    cpp_parser = CppParser()
    total = {"parsed": 0, "upserted": 0, "removed": 0, "skipped": 0, "files": 0, "errors": 0}

    if args.dry_run:
        for fp in files:
            try:
                content = Path(fp).read_text(encoding="utf-8", errors="replace")
                chunks = cpp_parser.parse_file(fp, content)
                total["parsed"] += len(chunks)
                total["files"] += 1
                for c in chunks:
                    print(f"  {c['chunk_type']:8s} {c['qualified_name']} (L{c['line_start']}-{c['line_end']})")
            except Exception as e:
                LOGGER.error("Error parsing %s: %s", fp, e)
                total["errors"] += 1
    else:
        conn = psycopg.connect(DB_DSN)
        try:
            for fp in files:
                try:
                    stats = process_cpp_file(conn, cpp_parser, fp, project_id=args.project_id, force=args.force)
                    for k in ("parsed", "upserted", "removed", "skipped"):
                        total[k] += stats[k]
                    total["files"] += 1
                    if stats["skipped"]:
                        LOGGER.debug("SKIP %s", fp)
                    else:
                        LOGGER.info("PARSED %s: %d chunks", Path(fp).name, stats["upserted"])
                except Exception as e:
                    LOGGER.error("Error processing %s: %s", fp, e)
                    total["errors"] += 1
        finally:
            conn.close()

    print(f"FILES={total['files']}")
    print(f"PARSED={total['parsed']}")
    print(f"UPSERTED={total['upserted']}")
    print(f"REMOVED={total['removed']}")
    print(f"SKIPPED={total['skipped']}")
    print(f"ERRORS={total['errors']}")


if __name__ == "__main__":
    main()
