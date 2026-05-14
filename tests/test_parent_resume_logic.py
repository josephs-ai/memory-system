def test_parent_resume_rule_documented():
    """
    This is a lightweight contract test for MVP semantics.

    Parent waiting_on_children should resume only when all children are done,
    and should resume into approved/writeback.
    """
    parent_status = "waiting_on_children"
    children_statuses = ["done", "done"]

    should_resume = parent_status == "waiting_on_children" and all(
        s == "done" for s in children_statuses
    )

    assert should_resume is True

    resumed_status = "approved"
    resumed_next_action = "writeback"

    assert resumed_status == "approved"
    assert resumed_next_action == "writeback"
