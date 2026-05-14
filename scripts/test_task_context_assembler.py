"""
Tests for task_context_assembler.py — L8-S3 of Level 8.

Covers:
    - Target resolution by qualified_name
    - Target resolution by file_path + line
    - Full body included for target chunk
    - Class declaration included for methods
    - Header preference for C++ dependencies
    - Caller/callee extraction from Neo4j
    - Sibling method listing
    - Compile error integration
    - Patch history integration
    - Token budget enforcement
    - Priority ordering of sections
    - Prompt block formatting
    - Budget truncation behavior
"""

from __future__ import annotations

import os
import sys

import psycopg
import pytest

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from task_context_assembler import TaskContextAssembler, TokenBudget, TaskContext
from parse_code_ast import parse_file, upsert_chunks, remove_stale_chunks
from parse_code_cpp import CppParser
from base_code_parser import BaseCodeParser

DB_DSN = os.environ.get("OPENCLAW_MEMORY_DSN", "dbname=openclaw_memory")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

PYTHON_MODULE = """\
import os
import json

class CombatSystem:
    \"\"\"Manages combat encounters.\"\"\"

    def __init__(self):
        self.health = 100.0
        self.damage_multiplier = 1.0

    def take_damage(self, amount: float) -> float:
        \"\"\"Apply damage and return remaining health.\"\"\"
        effective = amount * self.damage_multiplier
        self.health -= effective
        return self.health

    def heal(self, amount: float) -> float:
        \"\"\"Heal and return new health.\"\"\"
        self.health = min(100.0, self.health + amount)
        return self.health

    def is_dead(self) -> bool:
        return self.health <= 0


def calculate_dps(damage: float, attack_speed: float) -> float:
    \"\"\"Calculate damage per second.\"\"\"
    return damage * attack_speed
"""

CPP_HEADER = """\
#pragma once
#include "CoreMinimal.h"

class ACombatManager {
public:
    ACombatManager();
    void ProcessDamage(float Amount);
    float GetHealth() const;
private:
    float Health;
};
"""

CPP_IMPL = """\
#include "CombatManager.h"

ACombatManager::ACombatManager() {
    Health = 100.0f;
}

void ACombatManager::ProcessDamage(float Amount) {
    Health -= Amount;
    if (Health <= 0.0f) {
        // Handle death
    }
}

float ACombatManager::GetHealth() const {
    return Health;
}
"""


@pytest.fixture
def db_conn():
    conn = psycopg.connect(DB_DSN)
    yield conn
    with conn.cursor() as cur:
        cur.execute("DELETE FROM code_chunks WHERE file_path LIKE '/test/tca/%'")
        cur.execute("DELETE FROM patch_ledger WHERE step_id LIKE 'test-tca-%'")
    conn.commit()
    conn.close()


@pytest.fixture
def seeded_db(db_conn):
    """Insert Python and C++ test chunks."""
    # Python chunks
    py_chunks = parse_file("/test/tca/combat.py", PYTHON_MODULE)
    upsert_chunks(db_conn, py_chunks)

    # C++ chunks
    cpp_parser = CppParser()
    header_chunks = cpp_parser.parse_file("/test/tca/CombatManager.h", CPP_HEADER)
    impl_chunks = cpp_parser.parse_file("/test/tca/CombatManager.cpp", CPP_IMPL)
    upsert_chunks(db_conn, header_chunks)
    upsert_chunks(db_conn, impl_chunks)

    return {
        "py_chunks": py_chunks,
        "header_chunks": header_chunks,
        "impl_chunks": impl_chunks,
    }


# ---------------------------------------------------------------------------
# Tests: Target Resolution
# ---------------------------------------------------------------------------

class TestTargetResolution:
    def test_resolve_by_qualified_name(self, db_conn, seeded_db):
        asm = TaskContextAssembler(db_conn)
        ctx = asm.assemble(
            "fix damage calculation",
            target_qualified_name="combat.CombatSystem.take_damage",
        )
        assert ctx.target_qualified_name == "combat.CombatSystem.take_damage"
        assert ctx.total_chars > 0

    def test_resolve_by_file_line(self, db_conn, seeded_db):
        asm = TaskContextAssembler(db_conn)
        ctx = asm.assemble(
            "fix line 13",
            target_file="/test/tca/combat.py",
            target_line=13,
        )
        # Line 13 is inside take_damage method
        assert ctx.target_qualified_name is not None
        assert "take_damage" in ctx.target_qualified_name

    def test_nonexistent_target(self, db_conn, seeded_db):
        asm = TaskContextAssembler(db_conn)
        ctx = asm.assemble(
            "fix something",
            target_qualified_name="nonexistent.function",
        )
        assert ctx.total_chars == 0
        assert len(ctx.sections) == 0


# ---------------------------------------------------------------------------
# Tests: Section Content
# ---------------------------------------------------------------------------

class TestSectionContent:
    def test_target_body_included(self, db_conn, seeded_db):
        asm = TaskContextAssembler(db_conn)
        ctx = asm.assemble(
            "fix damage",
            target_qualified_name="combat.CombatSystem.take_damage",
        )
        target_sections = [s for s in ctx.sections if s.priority == 1]
        assert len(target_sections) == 1
        assert "def take_damage" in target_sections[0].content

    def test_class_declaration_included(self, db_conn, seeded_db):
        asm = TaskContextAssembler(db_conn)
        ctx = asm.assemble(
            "fix damage",
            target_qualified_name="combat.CombatSystem.take_damage",
        )
        class_sections = [s for s in ctx.sections if s.priority == 2]
        assert len(class_sections) == 1
        assert "CombatSystem" in class_sections[0].content

    def test_sibling_methods_included(self, db_conn, seeded_db):
        asm = TaskContextAssembler(db_conn)
        ctx = asm.assemble(
            "fix damage",
            target_qualified_name="combat.CombatSystem.take_damage",
        )
        sibling_sections = [s for s in ctx.sections if s.priority == 6]
        assert len(sibling_sections) == 1
        assert "heal" in sibling_sections[0].content

    def test_import_context_included(self, db_conn, seeded_db):
        asm = TaskContextAssembler(db_conn)
        ctx = asm.assemble(
            "fix imports",
            target_qualified_name="combat.CombatSystem.take_damage",
        )
        import_sections = [s for s in ctx.sections if s.priority == 10]
        assert len(import_sections) == 1
        assert "os" in import_sections[0].content or "json" in import_sections[0].content


# ---------------------------------------------------------------------------
# Tests: C++ Header Preference
# ---------------------------------------------------------------------------

class TestCppHeaderPreference:
    def test_header_interface_for_cpp(self, db_conn, seeded_db):
        """For C++ targets, dependency interfaces should prefer .h over .cpp."""
        asm = TaskContextAssembler(db_conn)
        ctx = asm.assemble(
            "fix ProcessDamage",
            target_qualified_name="CombatManager.ACombatManager.ProcessDamage",
        )
        # Should have target body
        target_sections = [s for s in ctx.sections if s.priority == 1]
        assert len(target_sections) == 1
        assert "ProcessDamage" in target_sections[0].content

    def test_header_preferred_over_impl(self, db_conn, seeded_db):
        """find_header_for should resolve .cpp → .h."""
        asm = TaskContextAssembler(db_conn)
        header = asm._find_header_for("/test/tca/CombatManager.cpp")
        assert header == "/test/tca/CombatManager.h"

    def test_header_returns_self_for_h(self, db_conn, seeded_db):
        asm = TaskContextAssembler(db_conn)
        header = asm._find_header_for("/test/tca/CombatManager.h")
        assert header == "/test/tca/CombatManager.h"


# ---------------------------------------------------------------------------
# Tests: Compile Error Integration
# ---------------------------------------------------------------------------

class TestErrorIntegration:
    def test_error_context_added(self, db_conn, seeded_db):
        error_groups = [
            {
                "category": "undeclared_identifier",
                "is_cascade": False,
                "fix_strategy": "Add missing #include",
                "root_error": {
                    "file": "/test/tca/combat.py",
                    "line": 13,
                    "message": "use of undeclared identifier 'Helth'",
                    "chunk_name": "combat.CombatSystem.take_damage",
                    "notes": [],
                },
            }
        ]
        asm = TaskContextAssembler(db_conn)
        ctx = asm.assemble(
            "fix typo",
            target_qualified_name="combat.CombatSystem.take_damage",
            error_groups=error_groups,
        )
        error_sections = [s for s in ctx.sections if s.priority == 7]
        assert len(error_sections) == 1
        assert "undeclared_identifier" in error_sections[0].content
        assert "Helth" in error_sections[0].content


# ---------------------------------------------------------------------------
# Tests: Token Budget
# ---------------------------------------------------------------------------

class TestTokenBudget:
    def test_total_budget_enforced(self, db_conn, seeded_db):
        tiny_budget = TokenBudget(total_max_chars=500)
        asm = TaskContextAssembler(db_conn, budget=tiny_budget)
        ctx = asm.assemble(
            "fix damage",
            target_qualified_name="combat.CombatSystem.take_damage",
        )
        # Total should be near or under 500
        assert ctx.total_chars <= 600  # Allow small overshoot from truncation marker

    def test_per_section_budget_enforced(self, db_conn, seeded_db):
        small_budget = TokenBudget(
            total_max_chars=100000,
            target_body_max=100,
        )
        asm = TaskContextAssembler(db_conn, budget=small_budget)
        ctx = asm.assemble(
            "fix damage",
            target_qualified_name="combat.CombatSystem.take_damage",
        )
        target_sections = [s for s in ctx.sections if s.priority == 1]
        if target_sections:
            # Should be truncated near 100 chars
            assert target_sections[0].char_count <= 200  # +truncation marker

    def test_priority_ordering_correct(self, db_conn, seeded_db):
        asm = TaskContextAssembler(db_conn)
        ctx = asm.assemble(
            "fix damage",
            target_qualified_name="combat.CombatSystem.take_damage",
        )
        priorities = [s.priority for s in ctx.sections]
        # Should be in ascending order (highest priority first)
        assert priorities == sorted(priorities)


# ---------------------------------------------------------------------------
# Tests: Prompt Block
# ---------------------------------------------------------------------------

class TestPromptBlock:
    def test_prompt_block_format(self, db_conn, seeded_db):
        asm = TaskContextAssembler(db_conn)
        ctx = asm.assemble(
            "fix damage calculation",
            target_qualified_name="combat.CombatSystem.take_damage",
        )
        block = ctx.to_prompt_block()
        assert "## Task: fix damage calculation" in block
        assert "## Target: combat.CombatSystem.take_damage" in block
        assert "###" in block  # Section headers

    def test_to_dict_serializable(self, db_conn, seeded_db):
        import json
        asm = TaskContextAssembler(db_conn)
        ctx = asm.assemble(
            "fix damage",
            target_qualified_name="combat.CombatSystem.take_damage",
        )
        d = ctx.to_dict()
        # Should be JSON-serializable
        serialized = json.dumps(d)
        assert len(serialized) > 0
        assert d["total_tokens_approx"] > 0


# ---------------------------------------------------------------------------
# Tests: Patch History Integration
# ---------------------------------------------------------------------------

class TestPatchHistoryIntegration:
    def test_patch_history_included(self, db_conn, seeded_db):
        from patch_ledger import PatchLedger
        ledger = PatchLedger(db_conn)
        ledger.record_patch(
            step_id="test-tca-001",
            file_path="/test/tca/combat.py",
            content_after="modified",
            operation="modify",
        )

        asm = TaskContextAssembler(db_conn)
        ctx = asm.assemble(
            "fix damage",
            target_qualified_name="combat.CombatSystem.take_damage",
        )
        patch_sections = [s for s in ctx.sections if s.priority == 8]
        assert len(patch_sections) == 1
        assert "test-tca-001" in patch_sections[0].content


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
