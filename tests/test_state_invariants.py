from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from orchestrator.state_invariants import validate_work_item_state


def test_valid_done_state():
    errors = validate_work_item_state(
        status="done",
        next_action="close",
        assigned_role=None,
    )
    assert errors == []


def test_invalid_done_with_assigned_role():
    errors = validate_work_item_state(
        status="done",
        next_action="close",
        assigned_role="tester",
    )
    assert errors
    assert any("assigned_role must be null" in e for e in errors)


def test_invalid_next_action_for_done():
    errors = validate_work_item_state(
        status="done",
        next_action="test",
        assigned_role=None,
    )
    assert errors
    assert any("Invalid next_action" in e for e in errors)


def test_valid_testing_state():
    errors = validate_work_item_state(
        status="testing",
        next_action="test",
        assigned_role="tester",
    )
    assert errors == []


def test_invalid_testing_role():
    errors = validate_work_item_state(
        status="testing",
        next_action="test",
        assigned_role="reviewer",
    )
    assert errors
    assert any("tester" in e for e in errors)
