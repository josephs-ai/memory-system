"""
Tests for parse_code_ast.py — Slice 2 (S2) of the Zero-Reread Architecture.

Covers:
    - Module-level parsing
    - Top-level functions (sync and async)
    - Classes with methods
    - Nested functions
    - Decorators (simple, dotted, with arguments)
    - Import extraction (import, from...import, inline imports)
    - Signature extraction (single-line and multi-line)
    - Chunk ID determinism
    - SyntaxError handling (graceful skip)
    - Incremental parsing (file hash skip logic)
    - Stale chunk removal
    - Dry-run mode
    - Database upsert idempotency
"""

from __future__ import annotations

import hashlib
import os
import sys
import tempfile
import textwrap

import psycopg
import pytest

# Ensure script directory is on path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from parse_code_ast import (
    make_chunk_id,
    parse_file,
    _file_hash,
    _extract_imports,
    _get_decorator_names,
    discover_python_files,
    process_file,
    upsert_chunks,
    remove_stale_chunks,
    get_existing_file_hash,
)

DB_DSN = os.environ.get("OPENCLAW_MEMORY_DSN", "dbname=openclaw_memory")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SIMPLE_MODULE = textwrap.dedent('''\
    """A simple test module."""

    import os
    from pathlib import Path

    CONSTANT = 42

    def hello(name: str) -> str:
        """Say hello."""
        return f"Hello, {name}"

    def goodbye():
        return "Bye"
''')

ASYNC_FUNCTION = textwrap.dedent('''\
    import asyncio

    async def fetch_data(url: str, *, timeout: int = 30) -> dict:
        """Fetch data from URL."""
        async with aiohttp.ClientSession() as session:
            resp = await session.get(url, timeout=timeout)
            return await resp.json()
''')

CLASS_WITH_METHODS = textwrap.dedent('''\
    from dataclasses import dataclass

    class Calculator:
        """A simple calculator."""

        def __init__(self, precision: int = 2):
            self.precision = precision

        def add(self, a: float, b: float) -> float:
            """Add two numbers."""
            return round(a + b, self.precision)

        async def async_compute(self, expr: str) -> float:
            """Async computation."""
            return eval(expr)
''')

NESTED_FUNCTIONS = textwrap.dedent('''\
    def outer(x):
        """Outer function."""
        def inner(y):
            """Inner helper."""
            return x + y
        return inner(10)
''')

COMPLEX_DECORATORS = textwrap.dedent('''\
    import functools
    from app import router

    @staticmethod
    def plain_decorator():
        pass

    @router.get("/health")
    def dotted_decorator():
        pass

    @functools.lru_cache(maxsize=128)
    def decorator_with_args(n):
        return n * 2
''')

INLINE_IMPORTS = textwrap.dedent('''\
    def lazy_loader():
        import json
        from collections import OrderedDict
        return json.dumps(OrderedDict())
''')

MULTILINE_SIGNATURE = textwrap.dedent('''\
    def complex_function(
        arg1: str,
        arg2: int,
        *,
        keyword_only: bool = False,
        another: float = 3.14,
    ) -> dict[str, Any]:
        """A function with a multi-line signature."""
        return {"arg1": arg1}
''')

SYNTAX_ERROR_FILE = "def broken(\n"

EMPTY_FILE = ""

SINGLE_LINE_FILE = "x = 1\n"


# ---------------------------------------------------------------------------
# Unit Tests: Chunk ID
# ---------------------------------------------------------------------------

class TestChunkId:
    def test_deterministic(self):
        """Same inputs always produce the same ID."""
        id1 = make_chunk_id("/a/b.py", "function", "b.foo")
        id2 = make_chunk_id("/a/b.py", "function", "b.foo")
        assert id1 == id2

    def test_prefix(self):
        """Chunk IDs start with 'cc-'."""
        cid = make_chunk_id("/a/b.py", "function", "b.foo")
        assert cid.startswith("cc-")

    def test_different_inputs_different_ids(self):
        """Different inputs produce different IDs."""
        id1 = make_chunk_id("/a/b.py", "function", "b.foo")
        id2 = make_chunk_id("/a/b.py", "function", "b.bar")
        id3 = make_chunk_id("/a/c.py", "function", "c.foo")
        assert id1 != id2
        assert id1 != id3

    def test_chunk_type_matters(self):
        """Same name but different chunk_type produces different ID."""
        id1 = make_chunk_id("/a/b.py", "function", "b.Foo")
        id2 = make_chunk_id("/a/b.py", "class", "b.Foo")
        assert id1 != id2


# ---------------------------------------------------------------------------
# Unit Tests: Parse File
# ---------------------------------------------------------------------------

class TestParseFile:
    def test_simple_module(self):
        chunks = parse_file("/test/simple.py", SIMPLE_MODULE)
        types = {c["chunk_type"] for c in chunks}
        assert "module" in types
        assert "function" in types

        # Should have: 1 module + 2 functions
        assert len(chunks) == 3

        # Module chunk
        mod = [c for c in chunks if c["chunk_type"] == "module"][0]
        assert mod["qualified_name"] == "simple"
        assert mod["docstring"] == "A simple test module."
        assert "os" in mod["imports"]
        assert "pathlib.Path" in mod["imports"]
        assert mod["line_start"] == 1

        # Function chunks
        fns = {c["qualified_name"]: c for c in chunks if c["chunk_type"] == "function"}
        assert "simple.hello" in fns
        assert "simple.goodbye" in fns
        assert fns["simple.hello"]["docstring"] == "Say hello."
        assert "def hello" in fns["simple.hello"]["signature"]

    def test_async_function(self):
        chunks = parse_file("/test/async_mod.py", ASYNC_FUNCTION)
        fns = [c for c in chunks if c["chunk_type"] == "function"]
        assert len(fns) == 1
        fn = fns[0]
        assert fn["qualified_name"] == "async_mod.fetch_data"
        assert "async def" in fn["signature"]
        assert fn["docstring"] == "Fetch data from URL."

    def test_class_with_methods(self):
        chunks = parse_file("/test/calc.py", CLASS_WITH_METHODS)
        classes = [c for c in chunks if c["chunk_type"] == "class"]
        methods = [c for c in chunks if c["chunk_type"] == "method"]

        assert len(classes) == 1
        assert classes[0]["qualified_name"] == "calc.Calculator"
        assert classes[0]["docstring"] == "A simple calculator."

        assert len(methods) == 3  # __init__, add, async_compute
        method_names = {m["qualified_name"] for m in methods}
        assert "calc.Calculator.__init__" in method_names
        assert "calc.Calculator.add" in method_names
        assert "calc.Calculator.async_compute" in method_names

        # Methods should reference their parent class chunk
        for m in methods:
            assert m["parent_chunk_id"] == classes[0]["id"]
            assert m["class_name"] == "Calculator"

    def test_nested_functions(self):
        chunks = parse_file("/test/nested.py", NESTED_FUNCTIONS)
        fns = [c for c in chunks if c["chunk_type"] == "function"]

        assert len(fns) == 2  # outer + inner
        names = {f["qualified_name"] for f in fns}
        assert "nested.outer" in names
        assert "nested.outer.inner" in names

        inner = [f for f in fns if f["qualified_name"] == "nested.outer.inner"][0]
        outer = [f for f in fns if f["qualified_name"] == "nested.outer"][0]
        assert inner["parent_chunk_id"] == outer["id"]
        assert inner["docstring"] == "Inner helper."

    def test_decorators(self):
        chunks = parse_file("/test/deco.py", COMPLEX_DECORATORS)
        fns = {c["qualified_name"]: c for c in chunks if c["chunk_type"] == "function"}

        assert fns["deco.plain_decorator"]["decorators"] == ["staticmethod"]
        assert fns["deco.dotted_decorator"]["decorators"] == ["router.get"]
        assert fns["deco.decorator_with_args"]["decorators"] == ["functools.lru_cache"]

    def test_inline_imports(self):
        chunks = parse_file("/test/lazy.py", INLINE_IMPORTS)
        fn = [c for c in chunks if c["chunk_type"] == "function"][0]
        assert "json" in fn["imports"]
        assert "collections.OrderedDict" in fn["imports"]

    def test_multiline_signature(self):
        chunks = parse_file("/test/multi.py", MULTILINE_SIGNATURE)
        fn = [c for c in chunks if c["chunk_type"] == "function"][0]
        assert "def complex_function" in fn["signature"]
        assert "keyword_only" in fn["signature"]

    def test_syntax_error_graceful(self):
        """SyntaxError should return empty list, not crash."""
        chunks = parse_file("/test/broken.py", SYNTAX_ERROR_FILE)
        assert chunks == []

    def test_empty_file(self):
        """Empty file should produce a module chunk."""
        chunks = parse_file("/test/empty.py", EMPTY_FILE)
        # ast.parse("") succeeds — produces a Module with no body
        assert len(chunks) == 1
        assert chunks[0]["chunk_type"] == "module"

    def test_single_line_file(self):
        chunks = parse_file("/test/one.py", SINGLE_LINE_FILE)
        assert len(chunks) == 1
        assert chunks[0]["chunk_type"] == "module"

    def test_line_numbers_consistent(self):
        """Line numbers should be valid and within file bounds."""
        chunks = parse_file("/test/simple.py", SIMPLE_MODULE)
        total_lines = len(SIMPLE_MODULE.splitlines())
        for chunk in chunks:
            assert chunk["line_start"] >= 1
            assert chunk["line_end"] >= chunk["line_start"]
            assert chunk["line_end"] <= total_lines

    def test_file_hash_deterministic(self):
        """All chunks from same file should have the same file_hash."""
        chunks = parse_file("/test/simple.py", SIMPLE_MODULE)
        hashes = {c["file_hash"] for c in chunks}
        assert len(hashes) == 1
        assert hashes.pop() == _file_hash(SIMPLE_MODULE)

    def test_module_name_from_path(self):
        chunks = parse_file("/some/deep/path/my_module.py", "x = 1\n")
        assert chunks[0]["module_name"] == "my_module"


# ---------------------------------------------------------------------------
# Unit Tests: Import Extraction
# ---------------------------------------------------------------------------

class TestImportExtraction:
    def test_module_level_imports(self):
        import ast as stdlib_ast
        tree = stdlib_ast.parse(SIMPLE_MODULE)
        imports = _extract_imports(tree)
        assert "os" in imports
        assert "pathlib.Path" in imports

    def test_no_imports(self):
        import ast as stdlib_ast
        tree = stdlib_ast.parse("x = 1\n")
        imports = _extract_imports(tree)
        assert imports == []


# ---------------------------------------------------------------------------
# Unit Tests: File Discovery
# ---------------------------------------------------------------------------

class TestFileDiscovery:
    def test_single_file(self, tmp_path):
        f = tmp_path / "test.py"
        f.write_text("x = 1\n")
        result = discover_python_files(str(f))
        assert len(result) == 1
        assert result[0] == str(f.resolve())

    def test_directory(self, tmp_path):
        (tmp_path / "a.py").write_text("x = 1\n")
        (tmp_path / "b.py").write_text("y = 2\n")
        (tmp_path / "sub").mkdir()
        (tmp_path / "sub" / "c.py").write_text("z = 3\n")
        result = discover_python_files(str(tmp_path))
        assert len(result) == 3

    def test_skips_pycache(self, tmp_path):
        (tmp_path / "a.py").write_text("x = 1\n")
        pycache = tmp_path / "__pycache__"
        pycache.mkdir()
        (pycache / "a.cpython-313.pyc.py").write_text("")  # shouldn't be found
        result = discover_python_files(str(tmp_path))
        assert len(result) == 1

    def test_skips_hidden_dirs(self, tmp_path):
        (tmp_path / "a.py").write_text("x = 1\n")
        hidden = tmp_path / ".hidden"
        hidden.mkdir()
        (hidden / "b.py").write_text("y = 2\n")
        result = discover_python_files(str(tmp_path))
        assert len(result) == 1

    def test_non_python_file(self, tmp_path):
        f = tmp_path / "readme.txt"
        f.write_text("hello")
        result = discover_python_files(str(f))
        assert result == []


# ---------------------------------------------------------------------------
# Integration Tests: Database operations
# ---------------------------------------------------------------------------

@pytest.fixture
def db_conn():
    """Provide a database connection and clean up test data after."""
    conn = psycopg.connect(DB_DSN)
    yield conn
    # Clean up any test chunks
    with conn.cursor() as cur:
        cur.execute("DELETE FROM code_chunks WHERE file_path LIKE '/test/%'")
    conn.commit()
    conn.close()


class TestDatabaseOps:
    def test_upsert_and_query(self, db_conn):
        chunks = parse_file("/test/simple.py", SIMPLE_MODULE)
        count = upsert_chunks(db_conn, chunks)
        assert count == len(chunks)

        with db_conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM code_chunks WHERE file_path = '/test/simple.py'")
            assert cur.fetchone()[0] == len(chunks)

    def test_upsert_idempotent(self, db_conn):
        chunks = parse_file("/test/simple.py", SIMPLE_MODULE)
        upsert_chunks(db_conn, chunks)
        upsert_chunks(db_conn, chunks)  # second run

        with db_conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM code_chunks WHERE file_path = '/test/simple.py'")
            assert cur.fetchone()[0] == len(chunks)

    def test_stale_removal(self, db_conn):
        # Insert chunks for a file
        chunks = parse_file("/test/simple.py", SIMPLE_MODULE)
        upsert_chunks(db_conn, chunks)

        # Now "re-parse" with only 1 chunk remaining
        keep_ids = {chunks[0]["id"]}
        removed = remove_stale_chunks(db_conn, "/test/simple.py", keep_ids)
        assert removed == len(chunks) - 1

        with db_conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM code_chunks WHERE file_path = '/test/simple.py'")
            assert cur.fetchone()[0] == 1

    def test_file_hash_skip(self, db_conn):
        chunks = parse_file("/test/simple.py", SIMPLE_MODULE)
        upsert_chunks(db_conn, chunks)

        # Should return the stored hash
        stored = get_existing_file_hash(db_conn, "/test/simple.py")
        assert stored == _file_hash(SIMPLE_MODULE)

        # Non-existent file returns None
        assert get_existing_file_hash(db_conn, "/test/nonexistent.py") is None

    def test_process_file_incremental(self, db_conn, tmp_path):
        f = tmp_path / "incr.py"
        f.write_text(SIMPLE_MODULE)

        # First run: should parse
        stats = process_file(db_conn, str(f.resolve()))
        assert stats["parsed"] > 0
        assert stats["skipped"] == 0

        # Second run: should skip (unchanged)
        stats2 = process_file(db_conn, str(f.resolve()))
        assert stats2["skipped"] == 1
        assert stats2["parsed"] == 0

        # Force run: should re-parse
        stats3 = process_file(db_conn, str(f.resolve()), force=True)
        assert stats3["parsed"] > 0
        assert stats3["skipped"] == 0

    def test_constraint_enforcement(self, db_conn):
        """CHECK constraints should reject bad data."""
        with db_conn.cursor() as cur:
            with pytest.raises(psycopg.errors.CheckViolation):
                cur.execute(
                    """INSERT INTO code_chunks
                       (id, file_path, file_hash, chunk_type, qualified_name, body, line_start, line_end)
                       VALUES ('test-bad', '/test/x.py', 'h', 'INVALID', 'x', 'x', 1, 1)"""
                )
        db_conn.rollback()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
