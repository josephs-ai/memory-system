from orchestrator.memory_bus.payload_normalizer import normalize_memory_bus_payload


def test_normalize_empty_payload():
    out = normalize_memory_bus_payload(None)
    assert out == {
        "guardrails": [],
        "project_summary": None,
        "subproject_summary": None,
        "feature_summaries": [],
        "relevant_memories": [],
        "memory_refs": [],
        "context_source": None,
    }


def test_normalize_string_guardrails():
    out = normalize_memory_bus_payload({
        "guardrails": "do not break movement state machine",
        "project_summary": "proj",
        "feature_summaries": ["feat1"],
        "relevant_memories": ["mem1"],
        "memory_refs": ["ref1"],
        "context_source": "postgres",
    })
    assert out["guardrails"] == ["do not break movement state machine"]
    assert out["project_summary"] == "proj"
    assert out["feature_summaries"] == ["feat1"]
    assert out["relevant_memories"] == ["mem1"]
    assert out["memory_refs"] == ["ref1"]
    assert out["context_source"] == "postgres"
