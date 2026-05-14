def test_planner_result_can_carry_routing_tier():
    planner_result = {"classification": "needs_human"}
    planner_result["routing_tier"] = "reasoning_max"
    assert planner_result["routing_tier"] == "reasoning_max"


def test_worker_result_can_carry_routing_tier():
    metadata = {}
    metadata["routing_tier"] = "coding_strong"
    assert metadata["routing_tier"] == "coding_strong"
