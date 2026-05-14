from __future__ import annotations

from orchestrator.policies.role_routing_policy import (
    is_executable_specialist_role,
    resolve_allowed_specialist_tokens,
)


def test_system_designer_is_executable() -> None:
    assert is_executable_specialist_role("system_designer") is True


def test_system_designer_accepted_for_cross_system_feature() -> None:
    accepted = resolve_allowed_specialist_tokens(
        planner_tokens=["system_designer", "technical_designer"],
        kind="feature",
        risk_level="medium",
        tags=["cross_system"],
        classification="decompose",
    )
    assert "system_designer" in accepted
    assert "technical_designer" in accepted


def test_system_designer_not_accepted_for_low_risk_local_feature() -> None:
    accepted = resolve_allowed_specialist_tokens(
        planner_tokens=["system_designer", "technical_designer"],
        kind="feature",
        risk_level="low",
        tags=["local_fix"],
        classification="decompose",
    )
    assert "system_designer" not in accepted
    assert "technical_designer" not in accepted


def test_system_designer_not_accepted_for_non_feature() -> None:
    accepted = resolve_allowed_specialist_tokens(
        planner_tokens=["system_designer"],
        kind="bugfix",
        risk_level="high",
        tags=["cross_system"],
        classification="decompose",
    )
    assert "system_designer" not in accepted
