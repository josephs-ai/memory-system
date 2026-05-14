from orchestrator.policies.model_routing_policy import resolve_model_routing_tier
from orchestrator.policies.role_routing_policy import (
    is_executable_specialist_role,
    normalize_specialist_recommendations,
)


def test_technical_designer_is_policy_recognized():
    out = normalize_specialist_recommendations(["technical_designer", "artist"])
    assert out == ["technical_designer"]


def test_technical_designer_is_executable_specialist_role():
    assert is_executable_specialist_role("technical_designer") is True


def test_technical_designer_routes_to_design_strong():
    assert resolve_model_routing_tier(role="technical_designer") == "design_strong"
