"""
base_code_parser.py — Abstract base class for language-specific code parsers.

L7-S4 of the Level 7 "Autonomous Enterprise" implementation.

Defines the BaseCodeParser interface that all language parsers must implement.
Provides shared utilities for chunk ID generation, file hashing, and database
operations that are language-agnostic.

Existing Python parser (parse_code_ast.py) and new parsers (C++, etc.) can
both conform to this interface. The Python parser is NOT refactored to inherit
from this base (to avoid breaking a tested, approved codebase); instead, new
parsers use this interface and produce the same output schema.

Output contract: all parsers must produce chunk dicts with these keys:
    id, file_path, file_hash, chunk_type, qualified_name, module_name,
    signature, docstring, body, decorators, imports, line_start, line_end,
    parent_chunk_id, class_name
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class BaseCodeParser(ABC):
    """
    Abstract base class for language-specific code parsers.

    Subclasses must implement:
        - language_id: str property returning the language identifier
        - file_extensions: list of file extensions this parser handles
        - parse_file(file_path, content) -> list of chunk dicts
    """

    @property
    @abstractmethod
    def language_id(self) -> str:
        """Language identifier (e.g., 'python', 'cpp', 'csharp')."""
        ...

    @property
    @abstractmethod
    def file_extensions(self) -> list[str]:
        """File extensions this parser handles (e.g., ['.py'] or ['.cpp', '.h', '.hpp'])."""
        ...

    @abstractmethod
    def parse_file(self, file_path: str, content: str) -> list[dict[str, Any]]:
        """
        Parse a source file into code chunks.

        Args:
            file_path: Absolute path to the source file.
            content: Full text content of the file.

        Returns:
            List of chunk dicts conforming to the code_chunks table schema.
            Each dict must have all required keys (see module docstring).
        """
        ...

    def can_parse(self, file_path: str) -> bool:
        """Check if this parser handles the given file."""
        return Path(file_path).suffix.lower() in self.file_extensions

    # --- Shared utilities (same as parse_code_ast.py) ---

    @staticmethod
    def make_chunk_id(file_path: str, chunk_type: str, qualified_name: str) -> str:
        """Deterministic chunk ID from (file_path, chunk_type, qualified_name)."""
        raw = f"{file_path}::{chunk_type}::{qualified_name}"
        return "cc-" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]

    @staticmethod
    def file_hash(content: str) -> str:
        """SHA-256 of file content."""
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    @staticmethod
    def module_name_from_path(file_path: str) -> str:
        """Derive module name from file path (stem without extension)."""
        return Path(file_path).stem

    def make_chunk(
        self,
        *,
        file_path: str,
        file_hash: str,
        chunk_type: str,
        qualified_name: str,
        module_name: str,
        body: str,
        line_start: int,
        line_end: int,
        signature: str | None = None,
        docstring: str | None = None,
        decorators: list[str] | None = None,
        imports: list[str] | None = None,
        parent_chunk_id: str | None = None,
        class_name: str | None = None,
    ) -> dict[str, Any]:
        """
        Build a chunk dict conforming to the code_chunks schema.
        Computes the chunk ID automatically.
        """
        chunk_id = self.make_chunk_id(file_path, chunk_type, qualified_name)
        return {
            "id": chunk_id,
            "file_path": file_path,
            "file_hash": file_hash,
            "chunk_type": chunk_type,
            "qualified_name": qualified_name,
            "module_name": module_name,
            "signature": signature,
            "docstring": docstring,
            "body": body,
            "decorators": decorators,
            "imports": imports or [],
            "line_start": line_start,
            "line_end": line_end,
            "parent_chunk_id": parent_chunk_id,
            "class_name": class_name,
        }
