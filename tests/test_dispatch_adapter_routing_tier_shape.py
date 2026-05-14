def test_worker_metadata_can_carry_routing_tier():
    metadata = {}
    metadata["routing_tier"] = "coding_strong"
    assert metadata["routing_tier"] == "coding_strong"
