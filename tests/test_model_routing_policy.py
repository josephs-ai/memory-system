from orchestrator.policies.model_routing_policy import resolve_model_routing_tier


def test_planner_routes_to_reasoning_max():
    assert resolve_model_routing_tier(role="planner") == "reasoning_max"


def test_builder_routes_to_coding_strong():
    assert resolve_model_routing_tier(role="builder") == "coding_strong"


def test_reviewer_routes_to_review_strong():
    assert resolve_model_routing_tier(role="reviewer") == "review_strong"


def test_tester_low_risk_routes_to_validation_cheap():
    assert (
        resolve_model_routing_tier(role="tester", risk_level="medium")
        == "validation_cheap"
    )


def test_tester_high_risk_routes_to_validation_strong():
    assert (
        resolve_model_routing_tier(role="tester", risk_level="high")
        == "validation_strong"
    )


def test_writeback_routes_to_writeback_cheap():
    assert resolve_model_routing_tier(role="writeback") == "writeback_cheap"


def test_technical_designer_routes_to_design_strong():
    assert resolve_model_routing_tier(role="technical_designer") == "design_strong"
