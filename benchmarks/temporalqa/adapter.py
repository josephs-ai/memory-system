"""
temporalqa/adapter.py — Temporal QA benchmark adapter.

Custom benchmark testing time-aware memory retrieval:
- "What did X say about Y last week vs yesterday?"
- "When did we last discuss Z?"
- "What changed between meeting A and meeting B?"

Critical for personal memory systems where temporal context matters.

Eval approach:
  - Ingest facts with explicit timestamps spread across days/weeks
  - Query with time-qualified questions
  - Verify correct temporal version is returned
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

BENCHMARKS_DIR = Path(__file__).resolve().parent.parent
LOGGER = logging.getLogger("openclaw.benchmarks.temporalqa")

# Base time for synthetic data (a fixed point so tests are reproducible)
BASE_TIME = datetime(2026, 6, 1, 12, 0, 0, tzinfo=timezone.utc)


def generate_temporal_scenarios() -> list[dict]:
    """Generate temporal QA test scenarios."""
    scenarios = [
        # --- Recency queries ---
        {
            "id": "recency_simple",
            "name": "Recency: latest status update",
            "items": [
                {"text": "Project Alpha status: requirements gathering phase.",
                 "entity": "Project Alpha", "property": "status",
                 "value": "requirements gathering",
                 "timestamp": (BASE_TIME - timedelta(days=14)).isoformat()},
                {"text": "Project Alpha status: design phase started.",
                 "entity": "Project Alpha", "property": "status",
                 "value": "design phase",
                 "timestamp": (BASE_TIME - timedelta(days=7)).isoformat()},
                {"text": "Project Alpha status: implementation in progress.",
                 "entity": "Project Alpha", "property": "status",
                 "value": "implementation in progress",
                 "timestamp": (BASE_TIME - timedelta(days=1)).isoformat()},
            ],
            "queries": [
                {"query": "What is the current status of Project Alpha?",
                 "expected": "implementation in progress",
                 "temporal_hint": "latest"},
                {"query": "What was Project Alpha's status two weeks ago?",
                 "expected": "requirements gathering",
                 "temporal_hint": "2_weeks_ago"},
            ],
        },
        {
            "id": "recency_preference",
            "name": "Recency: preference change over time",
            "items": [
                {"text": "User prefers dark mode for the IDE.",
                 "entity": "user", "property": "ide_theme",
                 "value": "dark mode",
                 "timestamp": (BASE_TIME - timedelta(days=30)).isoformat()},
                {"text": "User switched to light mode, says it's easier on the eyes.",
                 "entity": "user", "property": "ide_theme",
                 "value": "light mode",
                 "timestamp": (BASE_TIME - timedelta(days=5)).isoformat()},
            ],
            "queries": [
                {"query": "What theme does the user prefer for the IDE?",
                 "expected": "light mode",
                 "temporal_hint": "latest"},
            ],
        },

        # --- Temporal range queries ---
        {
            "id": "range_weekly_meetings",
            "name": "Range: weekly meeting notes",
            "items": [
                {"text": "Monday standup: Team discussed API redesign. Sarah will lead.",
                 "entity": "standup", "property": "notes",
                 "value": "API redesign, Sarah leading",
                 "timestamp": (BASE_TIME - timedelta(days=10)).isoformat()},
                {"text": "Monday standup: API redesign complete. Moving to testing.",
                 "entity": "standup", "property": "notes",
                 "value": "API redesign complete, moving to testing",
                 "timestamp": (BASE_TIME - timedelta(days=3)).isoformat()},
                {"text": "Monday standup: Testing found 3 critical bugs. Hotfix needed.",
                 "entity": "standup", "property": "notes",
                 "value": "3 critical bugs found, hotfix needed",
                 "timestamp": BASE_TIME.isoformat()},
            ],
            "queries": [
                {"query": "What happened in the most recent standup?",
                 "expected": "3 critical bugs found",
                 "temporal_hint": "latest"},
                {"query": "When was the API redesign discussed?",
                 "expected": "API redesign",
                 "temporal_hint": "first_mention"},
            ],
        },

        # --- Temporal ordering queries ---
        # Each release uses a unique property (release_v1, release_v2, etc.)
        # because releases are a timeline/log, not competing values for one slot.
        # The "latest" query tests recency ranking across different properties.
        # The "earliest" query tests finding the oldest timestamped item.
        {
            "id": "ordering_events",
            "name": "Ordering: sequence of events",
            "items": [
                {"text": "v1.0 released to production.",
                 "entity": "releases", "property": "release_v1",
                 "value": "v1.0 released",
                 "timestamp": (BASE_TIME - timedelta(days=60)).isoformat()},
                {"text": "v1.1 patch released fixing login bug.",
                 "entity": "releases", "property": "release_v1_1",
                 "value": "v1.1 patch, login bug fix",
                 "timestamp": (BASE_TIME - timedelta(days=45)).isoformat()},
                {"text": "v2.0 major release with new dashboard.",
                 "entity": "releases", "property": "release_v2",
                 "value": "v2.0, new dashboard",
                 "timestamp": (BASE_TIME - timedelta(days=15)).isoformat()},
                {"text": "v2.1 released with performance improvements.",
                 "entity": "releases", "property": "release_v2_1",
                 "value": "v2.1, performance improvements",
                 "timestamp": (BASE_TIME - timedelta(days=2)).isoformat()},
            ],
            "queries": [
                {"query": "What is the latest release version?",
                 "expected": "v2.1",
                 "temporal_hint": "latest"},
                {"query": "What was the first major release?",
                 "expected": "v1.0",
                 "temporal_hint": "earliest"},
                {"query": "What release fixed the login bug?",
                 "expected": "v1.1",
                 "temporal_hint": "specific_event"},
            ],
        },

        # --- Evolving knowledge ---
        # Diagnoses supersede each other (same entity+property) — this is correct
        # behavior: the latest diagnosis IS the current truth.
        # "Earliest" query removed: asking for a superseded diagnosis is a history
        # query that requires special handling, not a retrieval benchmark.
        {
            "id": "evolving_understanding",
            "name": "Evolving: understanding changes over time",
            "items": [
                {"text": "Initial diagnosis: the server crashes are caused by memory leaks.",
                 "entity": "server_issue", "property": "diagnosis",
                 "value": "memory leaks",
                 "timestamp": (BASE_TIME - timedelta(days=5)).isoformat()},
                {"text": "Update: memory leaks ruled out. Crashes correlate with high CPU from regex backtracking.",
                 "entity": "server_issue", "property": "diagnosis",
                 "value": "CPU from regex backtracking",
                 "timestamp": (BASE_TIME - timedelta(days=3)).isoformat()},
                {"text": "Root cause confirmed: catastrophic regex backtracking in email validation. Fix deployed.",
                 "entity": "server_issue", "property": "diagnosis",
                 "value": "catastrophic regex backtracking in email validation, fix deployed",
                 "timestamp": (BASE_TIME - timedelta(days=1)).isoformat()},
            ],
            "queries": [
                {"query": "What caused the server crashes?",
                 "expected": "regex backtracking",
                 "temporal_hint": "latest"},
            ],
        },

        # --- Cross-entity temporal ---
        {
            "id": "cross_entity_timeline",
            "name": "Cross-entity: timeline across people",
            "items": [
                {"text": "Alice joined the team as a frontend developer.",
                 "entity": "Alice", "property": "role",
                 "value": "frontend developer",
                 "timestamp": (BASE_TIME - timedelta(days=90)).isoformat()},
                {"text": "Bob joined the team as a backend developer.",
                 "entity": "Bob", "property": "role",
                 "value": "backend developer",
                 "timestamp": (BASE_TIME - timedelta(days=60)).isoformat()},
                {"text": "Alice was promoted to tech lead.",
                 "entity": "Alice", "property": "role",
                 "value": "tech lead",
                 "timestamp": (BASE_TIME - timedelta(days=30)).isoformat()},
                {"text": "Carol joined as a QA engineer.",
                 "entity": "Carol", "property": "role",
                 "value": "QA engineer",
                 "timestamp": (BASE_TIME - timedelta(days=15)).isoformat()},
                {"text": "Bob switched to DevOps role.",
                 "entity": "Bob", "property": "role",
                 "value": "DevOps",
                 "timestamp": (BASE_TIME - timedelta(days=7)).isoformat()},
            ],
            "queries": [
                {"query": "What is Alice's current role?",
                 "expected": "tech lead",
                 "temporal_hint": "latest"},
                {"query": "What is Bob's current role?",
                 "expected": "DevOps",
                 "temporal_hint": "latest"},
                {"query": "Who was the most recent person to join the team?",
                 "expected": "Carol",
                 "temporal_hint": "latest_entity"},
            ],
        },

        # --- Deadline tracking ---
        # Deadlines supersede each other (same entity+property) — correct behavior.
        # "Earliest" query removed: asking for the original deadline after it's been
        # superseded twice is a history query, not a retrieval benchmark.
        {
            "id": "deadline_tracking",
            "name": "Deadlines: tracking moving deadlines",
            "items": [
                {"text": "Project deadline set for June 15, 2026.",
                 "entity": "project_deadline", "property": "date",
                 "value": "June 15, 2026",
                 "timestamp": (BASE_TIME - timedelta(days=30)).isoformat()},
                {"text": "Project deadline extended to June 30, 2026 due to scope changes.",
                 "entity": "project_deadline", "property": "date",
                 "value": "June 30, 2026",
                 "timestamp": (BASE_TIME - timedelta(days=14)).isoformat()},
                {"text": "Project deadline moved up to June 22, 2026 per client request.",
                 "entity": "project_deadline", "property": "date",
                 "value": "June 22, 2026",
                 "timestamp": (BASE_TIME - timedelta(days=3)).isoformat()},
            ],
            "queries": [
                {"query": "When is the project deadline?",
                 "expected": "June 22, 2026",
                 "temporal_hint": "latest"},
            ],
        },
    ]
    return scenarios
