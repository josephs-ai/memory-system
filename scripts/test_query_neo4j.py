"""Regression tests for query_neo4j.py."""
from __future__ import annotations

from query_neo4j import _run_text_search


class FakeSession:
    def __init__(self):
        self.calls = []

    # Match neo4j.Session.run's problematic API shape: first positional arg is
    # named `query`, so passing query=... as a kwarg raises TypeError before the
    # method body. This test catches that exact regression.
    def run(self, query, **kwargs):  # noqa: A002 - mirrors third-party API
        self.calls.append((query, kwargs))
        return []


def test_text_search_does_not_pass_reserved_query_kwarg():
    session = FakeSession()
    rows = _run_text_search(
        session,
        query="memory startup",
        limit=5,
        project_id=None,
        subproject_id=None,
        workflow_id=None,
        pipeline_id=None,
    )

    assert rows == []
    cypher, params = session.calls[0]
    assert "$q" in cypher
    assert "$query" not in cypher
    assert params["q"] == "memory startup"
    assert "query" not in params
    assert params["limit"] == 5
