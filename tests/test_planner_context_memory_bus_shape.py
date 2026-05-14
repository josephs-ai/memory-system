from orchestrator.memory_bus.payload_normalizer import normalize_memory_bus_payload


def test_planner_context_memory_bus_shape_fields_exist():
    payload = normalize_memory_bus_payload({
        "guardrails": ["g1"],
        "project_summary": "p",
        "subproject_summary": "s",
        "feature_summaries": ["f1"],
        "relevant_memories": ["m1"],
        "memory_refs": ["mem1"],
        "context_source": "postgres_canonical_initial",
    })

    assert sorted(payload.keys()) == sorted([
        "guardrails",
        "project_summary",
        "subproject_summary",
        "feature_summaries",
        "relevant_memories",
        "memory_refs",
        "context_source",
    ])
