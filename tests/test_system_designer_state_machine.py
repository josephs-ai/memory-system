from __future__ import annotations

from orchestrator.state_machine import resolve_role_for_work_item, transition_status_on_dispatch


def test_system_designer_role_resolution() -> None:
    assert resolve_role_for_work_item("queued", "system_design") == "system_designer"
    assert resolve_role_for_work_item("ready", "system_design") == "system_designer"


def test_system_designer_dispatch_transition() -> None:
    assert transition_status_on_dispatch("system_designer") == "system_designing"
