from __future__ import annotations

from orchestrator.state_invariants import validate_work_item_state


def test_system_designing_allows_system_designer() -> None:
    errors = validate_work_item_state(
        status="system_designing",
        next_action="system_design",
        assigned_role="system_designer",
    )
    assert errors == []


def test_system_designing_rejects_wrong_role() -> None:
    errors = validate_work_item_state(
        status="system_designing",
        next_action="system_design",
        assigned_role="builder",
    )
    assert any("system_designer" in e for e in errors)
