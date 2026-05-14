from orchestrator.policies.planner_invocation_policy import (
    resolve_initial_next_action,
    should_invoke_planner,
)


def test_small_scoped_feature_still_invokes_planner_in_game_dev():
    item = {
        "memory_mode_effective": "game_dev",
        "kind": "feature",
        "risk_level": "medium",
        "title": "Small movement fix",
        "description": "Tight scoped fix.",
        "subproject_id": None,
        "tags": ["game_dev"],
    }
    assert should_invoke_planner(item) is True
    assert resolve_initial_next_action(item) == "plan"


def test_non_game_dev_skips_planner():
    item = {
        "memory_mode_effective": "normal",
        "kind": "feature",
        "risk_level": "medium",
        "title": "Small movement fix",
        "description": "Tight scoped fix.",
        "subproject_id": None,
        "tags": [],
    }
    assert should_invoke_planner(item) is False
    assert resolve_initial_next_action(item) == "build"
