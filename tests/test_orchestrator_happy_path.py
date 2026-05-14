from __future__ import annotations

from orchestrator.state_machine import (
    resolve_post_build_transition,
    resolve_post_review_transition,
    resolve_post_test_transition,
)


def test_happy_path_transitions():
    build_status, build_next = resolve_post_build_transition(requires_review=True)
    assert build_status == "built_pending_review"
    assert build_next == "review"

    review_status, review_next, review_retry, review_escalation = resolve_post_review_transition(
        passed=True,
        requires_test=True,
    )
    assert review_status == "review_passed_test_needed"
    assert review_next == "test"
    assert review_retry is False
    assert review_escalation is None

    test_status, test_next, test_retry, test_escalation = resolve_post_test_transition(
        passed=True
    )
    assert test_status == "approved"
    assert test_next == "writeback"
    assert test_retry is False
    assert test_escalation is None
