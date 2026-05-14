from orchestrator.policies.model_routing_policy import resolve_model_routing_tier
from orchestrator.provider_config import get_technical_designer_target


def test_technical_designer_routes_to_design_strong():
    assert resolve_model_routing_tier(role="technical_designer") == "design_strong"


def test_technical_designer_has_default_target():
    target = get_technical_designer_target("design_strong")
    assert target == "orchestrator_role_targets.adapter_technical_designer"
