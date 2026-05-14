"""
Tests for classify_compile_errors.py — L8-S1 of Level 8.

Covers:
    - Clang error line parsing
    - MSVC error line parsing
    - Error categorization (all categories)
    - Template instantiation chain handling
    - Macro expansion error handling (UE macros)
    - Linker error parsing
    - Error grouping and cascade detection
    - Chunk ID resolution from database
    - Linker symbol → chunk resolution
    - Full pipeline integration
    - Complex multi-error UE build log
"""

from __future__ import annotations

import os
import sys
import textwrap

import psycopg
import pytest

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from classify_compile_errors import (
    CompileError,
    ErrorCategory,
    ErrorGroup,
    classify_build_log,
    classify_error,
    group_errors,
    parse_build_log,
    parse_line,
    resolve_chunk_ids,
    resolve_linker_symbols,
)

DB_DSN = os.environ.get("OPENCLAW_MEMORY_DSN", "dbname=openclaw_memory")


# ---------------------------------------------------------------------------
# Realistic UE build log samples
# ---------------------------------------------------------------------------

CLANG_SIMPLE_ERROR = (
    "/Users/dev/MyGame/Source/MyGame/MyCharacter.cpp:47:5: "
    "error: use of undeclared identifier 'Helth'"
)

CLANG_WARNING = (
    "/Users/dev/MyGame/Source/MyGame/MyCharacter.cpp:23:12: "
    "warning: implicit conversion loses integer precision: 'int64_t' to 'int32_t'"
)

MSVC_ERROR = (
    "D:\\MyGame\\Source\\MyGame\\MyCharacter.cpp(47): "
    "error C2065: 'Helth': undeclared identifier"
)

CLANG_TEMPLATE_CHAIN = textwrap.dedent("""\
    /Users/dev/MyGame/Source/MyGame/AbilitySystem.cpp:128:15: error: no matching function for call to 'TArray<FAbilityData>::Add'
    /Users/dev/UE_5.7/Engine/Source/Runtime/Core/Public/Containers/Array.h:1247:7: note: candidate function not viable: no known conversion from 'FAbilityData *' to 'const FAbilityData &' for 1st argument
    /Users/dev/UE_5.7/Engine/Source/Runtime/Core/Public/Containers/Array.h:1253:7: note: candidate function not viable: requires 0 arguments, but 1 was provided
""")

CLANG_MACRO_ERROR = textwrap.dedent("""\
    /Users/dev/MyGame/Source/MyGame/MyCharacter.h:15:5: error: expected member name or ';' after declaration specifiers
    /Users/dev/MyGame/Source/MyGame/MyCharacter.h:14:5: note: expanded from macro 'UPROPERTY'
""")

CLANG_MISSING_INCLUDE = (
    "/Users/dev/MyGame/Source/MyGame/CombatSystem.cpp:3:10: "
    "fatal error: 'CombatSystem.h' file not found"
)

CLANG_CASCADE_LOG = textwrap.dedent("""\
    /Users/dev/MyGame/Source/MyGame/CombatSystem.cpp:3:10: fatal error: 'CombatTypes.h' file not found
    /Users/dev/MyGame/Source/MyGame/CombatSystem.cpp:15:5: error: use of undeclared identifier 'FCombatData'
    /Users/dev/MyGame/Source/MyGame/CombatSystem.cpp:22:12: error: use of undeclared identifier 'EDamageType'
    /Users/dev/MyGame/Source/MyGame/CombatSystem.cpp:30:8: error: unknown type name 'FCombatResult'
""")

LINKER_ERROR = (
    'ld.lld: error: undefined reference to "AMyCharacter::Attack(float)"'
)

UE_BUILD_NOISE = textwrap.dedent("""\
    [1/47] Compiling MyCharacter.cpp
    [2/47] Compiling CombatSystem.cpp
    Building MyGameEditor...
    Total time: 12.34s
""")

COMPLEX_UE_LOG = textwrap.dedent("""\
    [1/15] Compiling Module.MyGame.cpp
    /Users/dev/MyGame/Source/MyGame/MyCharacter.cpp:47:5: error: use of undeclared identifier 'Helth'
    /Users/dev/MyGame/Source/MyGame/AbilitySystem.cpp:128:15: error: no matching function for call to 'ProcessAbility'
    /Users/dev/MyGame/Source/MyGame/AbilitySystem.cpp:128:15: note: candidate function not viable: requires 2 arguments, but 3 were provided
    /Users/dev/MyGame/Source/MyGame/AbilitySystem.h:45:5: note: 'ProcessAbility' declared here
    /Users/dev/MyGame/Source/MyGame/EnemyAI.cpp:92:20: error: no member named 'GetTargetLocation' in 'AMyCharacter'
    /Users/dev/MyGame/Source/MyGame/RoomManager.cpp:33:10: fatal error: 'RoomTypes.h' file not found
    /Users/dev/MyGame/Source/MyGame/RoomManager.cpp:55:5: error: use of undeclared identifier 'FRoomData'
    /Users/dev/MyGame/Source/MyGame/RoomManager.cpp:62:12: error: unknown type name 'ERoomTransition'
    [2/15] Linking MyGameEditor
    ld.lld: error: undefined reference to "AEnemyBase::OnDeath()"
    Total time: 5.67s
""")

# Template error that should map to a specific chunk via line range
TEMPLATE_MACRO_COMPLEX = textwrap.dedent("""\
    /Users/dev/MyGame/Source/MyGame/MyCharacter.h:25:5: error: expected member name or ';' after declaration specifiers
    /Users/dev/MyGame/Source/MyGame/MyCharacter.h:24:5: note: expanded from macro 'UPROPERTY'
    /Users/dev/UE_5.7/Engine/Source/Runtime/CoreUObject/Public/UObject/ObjectMacros.h:1847:42: note: expanded from macro 'UPROPERTY'
    /Users/dev/MyGame/Source/MyGame/MyCharacter.h:32:5: error: no matching constructor for initialization of 'TArray<FInventoryItem>'
    /Users/dev/UE_5.7/Engine/Source/Runtime/Core/Public/Containers/Array.h:98:2: note: candidate constructor not viable
    /Users/dev/UE_5.7/Engine/Source/Runtime/Core/Public/Containers/Array.h:104:2: note: candidate constructor not viable
    /Users/dev/UE_5.7/Engine/Source/Runtime/Core/Public/Containers/Array.h:112:2: note: candidate constructor not viable
""")


# ---------------------------------------------------------------------------
# Tests: Line Parsing
# ---------------------------------------------------------------------------

class TestLineParsing:
    def test_clang_error(self):
        err = parse_line(CLANG_SIMPLE_ERROR)
        assert err is not None
        assert err.file_path == "/Users/dev/MyGame/Source/MyGame/MyCharacter.cpp"
        assert err.line == 47
        assert err.column == 5
        assert err.severity == "error"
        assert "undeclared identifier" in err.message

    def test_clang_warning(self):
        err = parse_line(CLANG_WARNING)
        assert err is not None
        assert err.severity == "warning"
        assert err.line == 23

    def test_msvc_error(self):
        err = parse_line(MSVC_ERROR)
        assert err is not None
        assert err.file_path == "D:\\MyGame\\Source\\MyGame\\MyCharacter.cpp"
        assert err.line == 47
        assert err.severity == "error"

    def test_clang_fatal(self):
        err = parse_line(CLANG_MISSING_INCLUDE)
        assert err is not None
        assert err.severity == "fatal error"
        assert "file not found" in err.message.lower()

    def test_noise_filtered(self):
        for line in UE_BUILD_NOISE.splitlines():
            assert parse_line(line) is None

    def test_linker_error(self):
        err = parse_line(LINKER_ERROR)
        assert err is not None
        assert err.severity == "error"


# ---------------------------------------------------------------------------
# Tests: Classification
# ---------------------------------------------------------------------------

class TestClassification:
    def test_undeclared_identifier(self):
        err = parse_line(CLANG_SIMPLE_ERROR)
        assert classify_error(err) == ErrorCategory.UNDECLARED_IDENTIFIER

    def test_missing_include(self):
        err = parse_line(CLANG_MISSING_INCLUDE)
        assert classify_error(err) == ErrorCategory.MISSING_INCLUDE

    def test_argument_mismatch(self):
        err = CompileError(
            file_path="test.cpp", line=1, column=1, severity="error",
            message="no matching function for call to 'foo'", raw_text=""
        )
        assert classify_error(err) == ErrorCategory.ARGUMENT_MISMATCH

    def test_missing_member(self):
        err = CompileError(
            file_path="test.cpp", line=1, column=1, severity="error",
            message="no member named 'GetHealth' in 'AMyCharacter'", raw_text=""
        )
        assert classify_error(err) == ErrorCategory.MISSING_MEMBER

    def test_linker_category(self):
        err = CompileError(
            file_path="", line=0, column=None, severity="error",
            message='undefined reference to "Foo::Bar()"', raw_text=""
        )
        assert classify_error(err) == ErrorCategory.LINKER_ERROR

    def test_syntax_error(self):
        err = CompileError(
            file_path="test.cpp", line=1, column=1, severity="error",
            message="expected ';' after expression", raw_text=""
        )
        assert classify_error(err) == ErrorCategory.SYNTAX_ERROR

    def test_access_control(self):
        err = CompileError(
            file_path="test.cpp", line=1, column=1, severity="error",
            message="'Health' is a private member of 'AMyCharacter'", raw_text=""
        )
        assert classify_error(err) == ErrorCategory.ACCESS_CONTROL

    def test_redefinition(self):
        err = CompileError(
            file_path="test.cpp", line=1, column=1, severity="error",
            message="redefinition of 'AMyCharacter'", raw_text=""
        )
        assert classify_error(err) == ErrorCategory.REDEFINITION


# ---------------------------------------------------------------------------
# Tests: Grouping & Cascades
# ---------------------------------------------------------------------------

class TestGrouping:
    def test_template_chain_grouped(self):
        errors = parse_build_log(CLANG_TEMPLATE_CHAIN)
        groups = group_errors(errors)
        assert len(groups) == 1
        assert groups[0].root_error.severity == "error"
        assert len(groups[0].related_errors) == 2  # 2 notes
        assert groups[0].root_error.notes  # notes attached

    def test_macro_expansion_grouped(self):
        errors = parse_build_log(CLANG_MACRO_ERROR)
        groups = group_errors(errors)
        assert len(groups) == 1
        assert groups[0].category == ErrorCategory.MACRO_EXPANSION

    def test_cascade_detection(self):
        errors = parse_build_log(CLANG_CASCADE_LOG)
        groups = group_errors(errors)
        # First error is the root cause (missing include)
        assert groups[0].category == ErrorCategory.MISSING_INCLUDE
        assert not groups[0].is_cascade
        # Subsequent errors in same file are cascades
        cascades = [g for g in groups if g.is_cascade]
        assert len(cascades) >= 2

    def test_complex_log_grouping(self):
        errors = parse_build_log(COMPLEX_UE_LOG)
        groups = group_errors(errors)
        # Should have multiple distinct error groups
        assert len(groups) >= 4
        # At least one cascade (RoomManager errors after missing include)
        cascades = [g for g in groups if g.is_cascade]
        assert len(cascades) >= 1

    def test_template_macro_complex(self):
        """Complex template/macro chain maps to correct root cause."""
        errors = parse_build_log(TEMPLATE_MACRO_COMPLEX)
        groups = group_errors(errors)
        # First group: macro expansion error
        assert groups[0].category == ErrorCategory.MACRO_EXPANSION
        # Second group: template/argument mismatch
        assert len(groups) >= 2


# ---------------------------------------------------------------------------
# Tests: Full Pipeline
# ---------------------------------------------------------------------------

class TestFullPipeline:
    def test_simple_log(self):
        result = classify_build_log(CLANG_SIMPLE_ERROR, resolve_chunks=False)
        assert result["summary"]["errors"] == 1
        assert result["summary"]["error_groups"] == 1
        assert "undeclared_identifier" in result["summary"]["category_breakdown"]

    def test_complex_log_stats(self):
        result = classify_build_log(COMPLEX_UE_LOG, resolve_chunks=False)
        s = result["summary"]
        assert s["errors"] >= 4
        assert s["cascades"] >= 1
        assert s["unique_root_causes"] < s["errors"]

    def test_fix_strategies_present(self):
        result = classify_build_log(COMPLEX_UE_LOG, resolve_chunks=False)
        for group in result["groups"]:
            assert group["fix_strategy"] != "" or group["category"] == "unknown"

    def test_empty_log(self):
        result = classify_build_log("", resolve_chunks=False)
        assert result["summary"]["total_diagnostics"] == 0
        assert result["summary"]["error_groups"] == 0

    def test_noise_only_log(self):
        result = classify_build_log(UE_BUILD_NOISE, resolve_chunks=False)
        assert result["summary"]["total_diagnostics"] == 0


# ---------------------------------------------------------------------------
# Tests: Database Integration (chunk resolution)
# ---------------------------------------------------------------------------

@pytest.fixture
def db_conn():
    conn = psycopg.connect(DB_DSN)
    yield conn
    # Clean up test data
    with conn.cursor() as cur:
        cur.execute("DELETE FROM code_chunks WHERE file_path LIKE '/test/classify/%'")
    conn.commit()
    conn.close()


@pytest.fixture
def test_chunks(db_conn):
    """Insert test chunks that match our error file paths."""
    from parse_code_ast import upsert_chunks
    from base_code_parser import BaseCodeParser

    chunks = [
        {
            "id": BaseCodeParser.make_chunk_id("/test/classify/MyCharacter.cpp", "method", "MyCharacter.AMyCharacter.Attack"),
            "file_path": "/test/classify/MyCharacter.cpp",
            "file_hash": "testhash1",
            "chunk_type": "method",
            "qualified_name": "MyCharacter.AMyCharacter.Attack",
            "module_name": "MyCharacter",
            "signature": "void AMyCharacter::Attack(float Damage)",
            "docstring": None,
            "body": "void AMyCharacter::Attack(float Damage) { Health -= Damage; }",
            "decorators": None,
            "imports": [],
            "line_start": 40,
            "line_end": 55,
            "parent_chunk_id": None,
            "class_name": "AMyCharacter",
        },
        {
            "id": BaseCodeParser.make_chunk_id("/test/classify/MyCharacter.cpp", "class", "MyCharacter.AMyCharacter"),
            "file_path": "/test/classify/MyCharacter.cpp",
            "file_hash": "testhash1",
            "chunk_type": "class",
            "qualified_name": "MyCharacter.AMyCharacter",
            "module_name": "MyCharacter",
            "signature": "class AMyCharacter : public ACharacter",
            "docstring": None,
            "body": "class AMyCharacter : public ACharacter { ... }",
            "decorators": None,
            "imports": [],
            "line_start": 10,
            "line_end": 80,
            "parent_chunk_id": None,
            "class_name": None,
        },
        {
            "id": BaseCodeParser.make_chunk_id("/test/classify/MyCharacter.cpp", "module", "MyCharacter"),
            "file_path": "/test/classify/MyCharacter.cpp",
            "file_hash": "testhash1",
            "chunk_type": "module",
            "qualified_name": "MyCharacter",
            "module_name": "MyCharacter",
            "signature": None,
            "docstring": None,
            "body": "// full file",
            "decorators": None,
            "imports": [],
            "line_start": 1,
            "line_end": 100,
            "parent_chunk_id": None,
            "class_name": None,
        },
    ]

    upsert_chunks(db_conn, chunks)
    return chunks


class TestChunkResolution:
    def test_line_to_chunk_exact(self, db_conn, test_chunks):
        """Error at line 47 of MyCharacter.cpp maps to Attack method (lines 40-55)."""
        err = CompileError(
            file_path="/test/classify/MyCharacter.cpp",
            line=47, column=5, severity="error",
            message="use of undeclared identifier 'Helth'",
            raw_text="",
        )
        resolved = resolve_chunk_ids(db_conn, [err])
        assert resolved == 1
        assert err.chunk_id is not None
        assert err.chunk_qualified_name == "MyCharacter.AMyCharacter.Attack"
        assert err.chunk_type == "method"

    def test_prefers_specific_chunk(self, db_conn, test_chunks):
        """When multiple chunks contain the line, prefer the most specific (smallest span)."""
        # Line 47 is inside both the method (40-55) and the class (10-80)
        err = CompileError(
            file_path="/test/classify/MyCharacter.cpp",
            line=47, column=5, severity="error",
            message="test", raw_text="",
        )
        resolve_chunk_ids(db_conn, [err])
        # Should prefer method over class over module
        assert err.chunk_type == "method"

    def test_class_level_error(self, db_conn, test_chunks):
        """Error at line 12 (inside class but not inside any method) maps to class chunk."""
        err = CompileError(
            file_path="/test/classify/MyCharacter.cpp",
            line=12, column=1, severity="error",
            message="test", raw_text="",
        )
        resolve_chunk_ids(db_conn, [err])
        assert err.chunk_type == "class"

    def test_linker_symbol_resolution(self, db_conn, test_chunks):
        """Linker error for 'AMyCharacter::Attack(float)' resolves to the Attack chunk."""
        err = CompileError(
            file_path="", line=0, column=None, severity="error",
            message='undefined reference to "AMyCharacter::Attack(float)"',
            raw_text="",
            category=ErrorCategory.LINKER_ERROR,
        )
        group = ErrorGroup(root_error=err, category=ErrorCategory.LINKER_ERROR)
        resolved = resolve_linker_symbols(db_conn, [group])
        assert resolved == 1
        assert err.chunk_qualified_name == "MyCharacter.AMyCharacter.Attack"

    def test_full_pipeline_with_db(self, db_conn, test_chunks):
        """Full pipeline resolves chunk IDs from a realistic build log."""
        log = "/test/classify/MyCharacter.cpp:47:5: error: use of undeclared identifier 'Helth'"
        result = classify_build_log(log, dsn=DB_DSN)
        grp = result["groups"][0]
        assert grp["root_error"]["chunk_id"] is not None
        assert grp["root_error"]["chunk_name"] == "MyCharacter.AMyCharacter.Attack"
        assert grp["root_error"]["chunk_type"] == "method"
        assert len(result["summary"]["chunks_affected"]) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
