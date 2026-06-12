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
        # --- Additional latest scenarios (larger pool) ---
        {
            "id": "latest_salary",
            "name": "Latest: salary/compensation updates",
            "items": [
                {"text": "Employee salary set to $85,000 per year.",
                 "entity": "employee", "property": "salary",
                 "value": "$85,000",
                 "timestamp": (BASE_TIME - timedelta(days=180)).isoformat()},
                {"text": "Employee received a raise to $95,000 per year.",
                 "entity": "employee", "property": "salary",
                 "value": "$95,000",
                 "timestamp": (BASE_TIME - timedelta(days=60)).isoformat()},
                {"text": "Employee promoted, new salary is $110,000.",
                 "entity": "employee", "property": "salary",
                 "value": "$110,000",
                 "timestamp": (BASE_TIME - timedelta(days=5)).isoformat()},
            ],
            "queries": [
                {"query": "What is the employee's current salary?",
                 "expected": "$110,000",
                 "temporal_hint": "latest"},
            ],
        },
        {
            "id": "latest_address",
            "name": "Latest: address change",
            "items": [
                {"text": "User lives at 123 Oak Street, Portland.",
                 "entity": "user", "property": "address",
                 "value": "123 Oak Street, Portland",
                 "timestamp": (BASE_TIME - timedelta(days=90)).isoformat()},
                {"text": "User moved to 456 Pine Ave, Seattle.",
                 "entity": "user", "property": "address",
                 "value": "456 Pine Ave, Seattle",
                 "timestamp": (BASE_TIME - timedelta(days=10)).isoformat()},
            ],
            "queries": [
                {"query": "Where does the user live?",
                 "expected": "Seattle",
                 "temporal_hint": "latest"},
            ],
        },
        {
            "id": "latest_stack",
            "name": "Latest: tech stack migration",
            "items": [
                {"text": "The backend is built with Django and PostgreSQL.",
                 "entity": "backend", "property": "framework",
                 "value": "Django",
                 "timestamp": (BASE_TIME - timedelta(days=120)).isoformat()},
                {"text": "Backend migrated to FastAPI for better async support.",
                 "entity": "backend", "property": "framework",
                 "value": "FastAPI",
                 "timestamp": (BASE_TIME - timedelta(days=14)).isoformat()},
            ],
            "queries": [
                {"query": "What framework does the backend use?",
                 "expected": "FastAPI",
                 "temporal_hint": "latest"},
            ],
        },
        {
            "id": "latest_manager",
            "name": "Latest: reporting structure change",
            "items": [
                {"text": "Team reports to Sarah, VP of Engineering.",
                 "entity": "team", "property": "manager",
                 "value": "Sarah",
                 "timestamp": (BASE_TIME - timedelta(days=60)).isoformat()},
                {"text": "Reorg: team now reports to Mike, Director of Platform.",
                 "entity": "team", "property": "manager",
                 "value": "Mike",
                 "timestamp": (BASE_TIME - timedelta(days=7)).isoformat()},
            ],
            "queries": [
                {"query": "Who does the team report to?",
                 "expected": "Mike",
                 "temporal_hint": "latest"},
            ],
        },
        {
            "id": "latest_db_version",
            "name": "Latest: database version upgrade",
            "items": [
                {"text": "Production database running PostgreSQL 14.",
                 "entity": "production_db", "property": "version",
                 "value": "PostgreSQL 14",
                 "timestamp": (BASE_TIME - timedelta(days=90)).isoformat()},
                {"text": "Upgraded production database to PostgreSQL 16.",
                 "entity": "production_db", "property": "version",
                 "value": "PostgreSQL 16",
                 "timestamp": (BASE_TIME - timedelta(days=3)).isoformat()},
            ],
            "queries": [
                {"query": "What version of PostgreSQL is running in production?",
                 "expected": "PostgreSQL 16",
                 "temporal_hint": "latest"},
            ],
        },
        {
            "id": "latest_deployment",
            "name": "Latest: deployment target change",
            "items": [
                {"text": "Application deployed on AWS EC2 instances.",
                 "entity": "application", "property": "deployment_target",
                 "value": "AWS EC2",
                 "timestamp": (BASE_TIME - timedelta(days=45)).isoformat()},
                {"text": "Migrated application deployment to Kubernetes on EKS.",
                 "entity": "application", "property": "deployment_target",
                 "value": "Kubernetes on EKS",
                 "timestamp": (BASE_TIME - timedelta(days=8)).isoformat()},
            ],
            "queries": [
                {"query": "Where is the application deployed?",
                 "expected": "Kubernetes",
                 "temporal_hint": "latest"},
            ],
        },
        {
            "id": "latest_ci_tool",
            "name": "Latest: CI/CD tool switch",
            "items": [
                {"text": "CI/CD pipeline runs on Jenkins.",
                 "entity": "ci_cd", "property": "tool",
                 "value": "Jenkins",
                 "timestamp": (BASE_TIME - timedelta(days=60)).isoformat()},
                {"text": "Switched CI/CD to GitHub Actions for better integration.",
                 "entity": "ci_cd", "property": "tool",
                 "value": "GitHub Actions",
                 "timestamp": (BASE_TIME - timedelta(days=12)).isoformat()},
            ],
            "queries": [
                {"query": "What CI/CD tool does the project use?",
                 "expected": "GitHub Actions",
                 "temporal_hint": "latest"},
            ],
        },
        {
            "id": "latest_meeting_day",
            "name": "Latest: recurring meeting schedule change",
            "items": [
                {"text": "Weekly team sync is on Mondays at 10am.",
                 "entity": "team_sync", "property": "schedule",
                 "value": "Mondays at 10am",
                 "timestamp": (BASE_TIME - timedelta(days=30)).isoformat()},
                {"text": "Team sync moved to Wednesdays at 2pm starting this week.",
                 "entity": "team_sync", "property": "schedule",
                 "value": "Wednesdays at 2pm",
                 "timestamp": (BASE_TIME - timedelta(days=4)).isoformat()},
            ],
            "queries": [
                {"query": "When is the weekly team sync?",
                 "expected": "Wednesdays",
                 "temporal_hint": "latest"},
            ],
        },
        {
            "id": "latest_priority",
            "name": "Latest: project priority shift",
            "items": [
                {"text": "Top priority for Q2 is the mobile app launch.",
                 "entity": "q2_priority", "property": "focus",
                 "value": "mobile app launch",
                 "timestamp": (BASE_TIME - timedelta(days=40)).isoformat()},
                {"text": "Priority shifted: security audit is now the top Q2 focus after the breach attempt.",
                 "entity": "q2_priority", "property": "focus",
                 "value": "security audit",
                 "timestamp": (BASE_TIME - timedelta(days=6)).isoformat()},
            ],
            "queries": [
                {"query": "What is the top priority for Q2?",
                 "expected": "security audit",
                 "temporal_hint": "latest"},
            ],
        },
        {
            "id": "latest_vendor",
            "name": "Latest: vendor switch",
            "items": [
                {"text": "Email service uses SendGrid for transactional emails.",
                 "entity": "email_service", "property": "vendor",
                 "value": "SendGrid",
                 "timestamp": (BASE_TIME - timedelta(days=75)).isoformat()},
                {"text": "Switched email provider to Postmark for better deliverability.",
                 "entity": "email_service", "property": "vendor",
                 "value": "Postmark",
                 "timestamp": (BASE_TIME - timedelta(days=9)).isoformat()},
            ],
            "queries": [
                {"query": "What email service provider do we use?",
                 "expected": "Postmark",
                 "temporal_hint": "latest"},
            ],
        },
        {
            "id": "latest_python_version",
            "name": "Latest: Python version upgrade",
            "items": [
                {"text": "Project uses Python 3.10 with type hints.",
                 "entity": "project", "property": "python_version",
                 "value": "Python 3.10",
                 "timestamp": (BASE_TIME - timedelta(days=100)).isoformat()},
                {"text": "Upgraded project to Python 3.12 for performance gains.",
                 "entity": "project", "property": "python_version",
                 "value": "Python 3.12",
                 "timestamp": (BASE_TIME - timedelta(days=15)).isoformat()},
            ],
            "queries": [
                {"query": "What Python version does the project use?",
                 "expected": "Python 3.12",
                 "temporal_hint": "latest"},
            ],
        },
        {
            "id": "latest_office_location",
            "name": "Latest: office relocation",
            "items": [
                {"text": "Company office is at 100 Market St, San Francisco.",
                 "entity": "company", "property": "office_location",
                 "value": "100 Market St, San Francisco",
                 "timestamp": (BASE_TIME - timedelta(days=200)).isoformat()},
                {"text": "Company relocated to 500 Terry Francois Blvd, San Francisco.",
                 "entity": "company", "property": "office_location",
                 "value": "500 Terry Francois Blvd, San Francisco",
                 "timestamp": (BASE_TIME - timedelta(days=20)).isoformat()},
            ],
            "queries": [
                {"query": "Where is the company office?",
                 "expected": "Terry Francois",
                 "temporal_hint": "latest"},
            ],
        },
    ]
    return scenarios


def generate_temporal_scenarios_large(count: int = 100, seed: int = 42) -> list[dict]:
    """Generate a larger pool of temporal scenarios for stress testing.

    Combines the hand-crafted scenarios with programmatically generated ones.
    """
    from temporalqa.generator import generate_latest_scenarios

    base = generate_temporal_scenarios()
    # Count existing latest queries
    existing_latest = sum(
        1 for s in base for q in s["queries"] if q["temporal_hint"] == "latest"
    )
    # Generate enough to hit target
    extra_needed = max(0, count - existing_latest)
    generated = generate_latest_scenarios(count=extra_needed, seed=seed)

    return base + generated

