from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from orchestrator.state_machine import (
    resolve_post_review_transition,
    resolve_post_test_transition,
)


def test_review_failure_goes_back_to_build():
    status, next_action, increment_retry, escalation_reason = resolve_post_review_transition(
        passed=False,
        requires_test=True,
    )
    assert status == "review_failed"
    assert next_action == "build"
    assert increment_retry is True
    assert escalation_reason == "review_failed_requires_rebuild"


def test_review_success_without_test_goes_to_approved():
    status, next_action, increment_retry, escalation_reason = resolve_post_review_transition(
        passed=True,
        requires_test=False,
    )
    assert status == "approved"
    assert next_action == "writeback"
    assert increment_retry is False
    assert escalation_reason is None


def test_test_failure_goes_back_to_build():
    status, next_action, increment_retry, escalation_reason = resolve_post_test_transition(
        passed=False
    )
    assert status == "test_failed"
    assert next_action == "build"
    assert increment_retry is True
    assert escalation_reason == "test_failed_requires_rebuild"


def test_test_success_goes_to_approved():
    status, next_action, increment_retry, escalation_reason = resolve_post_test_transition(
        passed=True
    )
    assert status == "approved"
    assert next_action == "writeback"
    assert increment_retry is False
    assert escalation_reason is None
