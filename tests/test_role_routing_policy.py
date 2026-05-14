from orchestrator.policies.role_routing_policy import (
    normalize_specialist_recommendations,
    resolve_allowed_specialist_tokens,
    should_accept_specialist_recommendation,
)


def test_normalize_specialist_recommendations_filters_unknowns():
    out = normalize_specialist_recommendations(
        ["technical_designer", "artist", "", "technical_designer"]
    )
    assert out == ["technical_designer"]


def test_accepts_technical_designer_for_feature_decompose():
    ok = should_accept_specialist_recommendation(
        token="technical_designer",
        kind="feature",
        risk_level="medium",
        tags=["cross_system"],
        classification="decompose",
    )
    assert ok is True


def test_rejects_technical_designer_for_bugfix():
    ok = should_accept_specialist_recommendation(
        token="technical_designer",
        kind="bugfix",
        risk_level="medium",
        tags=[],
        classification="decompose",
    )
    assert ok is False


def test_resolve_allowed_specialist_tokens_core_fallback():
    out = resolve_allowed_specialist_tokens(
        planner_tokens=["technical_designer", "artist"],
        kind="feature",
        risk_level="medium",
        tags=["cross_system"],
        classification="decompose",
    )
    assert out == ["technical_designer"]
