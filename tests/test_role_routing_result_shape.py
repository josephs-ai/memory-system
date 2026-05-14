def test_planner_result_can_carry_accepted_specialist_roles():
    planner_result = {
        "specialist_roles_requested": ["technical_designer"],
    }

    accepted_specialists = ["technical_designer"]
    planner_result["accepted_specialist_roles"] = accepted_specialists

    assert planner_result["accepted_specialist_roles"] == ["technical_designer"]
