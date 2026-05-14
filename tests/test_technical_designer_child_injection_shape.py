def test_technical_designer_child_injection_shape():
    planner_result = {
        "accepted_specialist_roles": ["technical_designer"],
        "proposed_child_items": [
            {
                "child_key": "child_1",
                "title": "Implement combat slice",
                "description": "builder slice",
                "kind": "feature",
                "priority": 50,
                "risk_level": "medium",
                "recommended_role": "builder",
                "tags": ["game_dev"],
                "memory_refs": ["mem://project/overview"],
                "depends_on_child_key": None,
            }
        ],
    }

    child_items = list(planner_result["proposed_child_items"])
    if "technical_designer" in planner_result["accepted_specialist_roles"] and child_items:
        first_child = child_items[0]
        td_child = {
            "child_key": "technical_designer_spec",
            "title": f"Technical design spec for: {first_child.get('title')}",
            "description": "Create a bounded implementation-facing technical design spec before builder work begins.",
            "kind": "feature",
            "priority": first_child.get("priority", 50),
            "risk_level": first_child.get("risk_level", "medium"),
            "recommended_role": "technical_designer",
            "tags": list(first_child.get("tags", []) or []),
            "memory_refs": list(first_child.get("memory_refs", []) or []),
            "depends_on_child_key": None,
        }
        child_items.insert(0, td_child)
        for child in child_items[1:]:
            if child.get("recommended_role") == "builder" and not child.get("depends_on_child_key"):
                child["depends_on_child_key"] = "technical_designer_spec"

    assert child_items[0]["recommended_role"] == "technical_designer"
    assert child_items[1]["depends_on_child_key"] == "technical_designer_spec"
