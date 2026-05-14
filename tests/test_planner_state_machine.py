from orchestrator.state_machine import (
    resolve_role_for_work_item,
    transition_status_on_dispatch,
    resolve_post_plan_transition,
)


def test_plan_role_resolution():
    assert resolve_role_for_work_item("queued", "plan") == "planner"
    assert resolve_role_for_work_item("ready", "plan") == "planner"


def test_plan_dispatch_status():
    assert transition_status_on_dispatch("planner") == "planning"


def test_plan_direct_execute_transition():
    status, next_action = resolve_post_plan_transition("direct_execute")
    assert status == "ready"
    assert next_action == "build"


def test_plan_needs_human_transition():
    status, next_action = resolve_post_plan_transition("needs_human")
    assert status == "awaiting_approval"
    assert next_action == "approve"


def test_plan_decompose_transition():
    status, next_action = resolve_post_plan_transition("decompose")
    assert status == "waiting_on_children"
    assert next_action == "wait"
