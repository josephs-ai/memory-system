from orchestrator.state_invariants import validate_work_item_state


def test_valid_planning_state():
    errors = validate_work_item_state(
        status="planning",
        next_action="plan",
        assigned_role="planner",
    )
    assert errors == []


def test_valid_waiting_on_children_state():
    errors = validate_work_item_state(
        status="waiting_on_children",
        next_action="wait",
        assigned_role=None,
    )
    assert errors == []


def test_invalid_planning_with_wrong_role():
    errors = validate_work_item_state(
        status="planning",
        next_action="plan",
        assigned_role="builder",
    )
    assert errors
    assert any("planner" in e for e in errors)


def test_invalid_waiting_on_children_with_assigned_role():
    errors = validate_work_item_state(
        status="waiting_on_children",
        next_action="wait",
        assigned_role="planner",
    )
    assert errors
    assert any("assigned_role must be null" in e for e in errors)
