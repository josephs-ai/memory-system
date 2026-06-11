"""
contradiction/adapter.py — Contradiction Detection benchmark adapter.

Tests whether the memory system can detect and resolve conflicting information.

Eval approach:
  - Insert a fact, then insert a contradicting fact
  - Query and verify the system returns the correct/latest version
  - Measure whether the system flags contradictions
  - Test implicit vs explicit contradictions
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

BENCHMARKS_DIR = Path(__file__).resolve().parent.parent
LOGGER = logging.getLogger("openclaw.benchmarks.contradiction")

BASE_TIME = datetime(2026, 6, 1, 12, 0, 0, tzinfo=timezone.utc)


def generate_contradiction_scenarios() -> list[dict]:
    """Generate contradiction detection test scenarios."""
    scenarios = [
        # --- Direct contradictions ---
        {
            "id": "direct_location",
            "name": "Direct: location change",
            "category": "direct",
            "items": [
                {"text": "Sarah lives in New York City.",
                 "entity": "Sarah", "property": "location", "value": "New York City",
                 "timestamp": (BASE_TIME - timedelta(days=30)).isoformat()},
                {"text": "Sarah moved to Los Angeles last month.",
                 "entity": "Sarah", "property": "location", "value": "Los Angeles",
                 "timestamp": (BASE_TIME - timedelta(days=2)).isoformat()},
            ],
            "queries": [
                {"query": "Where does Sarah live?",
                 "correct_answer": "Los Angeles",
                 "contradicted_answer": "New York City",
                 "should_prefer_latest": True},
            ],
        },
        {
            "id": "direct_status",
            "name": "Direct: project status contradiction",
            "category": "direct",
            "items": [
                {"text": "The migration project is on track for the June deadline.",
                 "entity": "migration_project", "property": "status", "value": "on track",
                 "timestamp": (BASE_TIME - timedelta(days=7)).isoformat()},
                {"text": "The migration project is delayed by two weeks due to infrastructure issues.",
                 "entity": "migration_project", "property": "status", "value": "delayed two weeks",
                 "timestamp": (BASE_TIME - timedelta(days=1)).isoformat()},
            ],
            "queries": [
                {"query": "What is the status of the migration project?",
                 "correct_answer": "delayed",
                 "contradicted_answer": "on track",
                 "should_prefer_latest": True},
            ],
        },
        {
            "id": "direct_numeric",
            "name": "Direct: numeric fact change",
            "category": "direct",
            "items": [
                {"text": "The team has 8 members.",
                 "entity": "team", "property": "size", "value": "8 members",
                 "timestamp": (BASE_TIME - timedelta(days=14)).isoformat()},
                {"text": "Two people left the team, now we have 6 members.",
                 "entity": "team", "property": "size", "value": "6 members",
                 "timestamp": (BASE_TIME - timedelta(days=3)).isoformat()},
            ],
            "queries": [
                {"query": "How many people are on the team?",
                 "correct_answer": "6",
                 "contradicted_answer": "8",
                 "should_prefer_latest": True},
            ],
        },

        # --- Implicit contradictions ---
        {
            "id": "implicit_role",
            "name": "Implicit: role implies different status",
            "category": "implicit",
            "items": [
                {"text": "Maria is a university student studying computer science.",
                 "entity": "Maria", "property": "status", "value": "university student",
                 "timestamp": (BASE_TIME - timedelta(days=365)).isoformat()},
                {"text": "Maria started her new job as a software engineer at Google.",
                 "entity": "Maria", "property": "status", "value": "software engineer at Google",
                 "timestamp": (BASE_TIME - timedelta(days=10)).isoformat()},
            ],
            "queries": [
                {"query": "Is Maria a student?",
                 "correct_answer": "software engineer at Google",
                 "contradicted_answer": "student",
                 "should_prefer_latest": True},
            ],
        },
        {
            "id": "implicit_capability",
            "name": "Implicit: capability contradiction",
            "category": "implicit",
            "items": [
                {"text": "The system only supports PostgreSQL as a database backend.",
                 "entity": "system", "property": "database", "value": "only PostgreSQL",
                 "timestamp": (BASE_TIME - timedelta(days=60)).isoformat()},
                {"text": "Added MySQL and SQLite support to the system alongside PostgreSQL.",
                 "entity": "system", "property": "database", "value": "PostgreSQL, MySQL, SQLite",
                 "timestamp": (BASE_TIME - timedelta(days=5)).isoformat()},
            ],
            "queries": [
                {"query": "What databases does the system support?",
                 "correct_answer": "PostgreSQL, MySQL, SQLite",
                 "contradicted_answer": "only PostgreSQL",
                 "should_prefer_latest": True},
            ],
        },

        # --- Partial contradictions (some info stays valid) ---
        {
            "id": "partial_update",
            "name": "Partial: some facts remain, some change",
            "category": "partial",
            "items": [
                {"text": "The API endpoint is at api.example.com, requires API key auth, rate limited to 100 req/min.",
                 "entity": "api", "property": "config",
                 "value": "api.example.com, API key, 100 req/min",
                 "timestamp": (BASE_TIME - timedelta(days=20)).isoformat()},
                {"text": "API rate limit increased to 500 req/min. Endpoint and auth unchanged.",
                 "entity": "api", "property": "rate_limit",
                 "value": "500 req/min",
                 "timestamp": (BASE_TIME - timedelta(days=2)).isoformat()},
            ],
            "queries": [
                {"query": "What is the API rate limit?",
                 "correct_answer": "500 req/min",
                 "contradicted_answer": "100 req/min",
                 "should_prefer_latest": True},
                {"query": "What is the API endpoint?",
                 "correct_answer": "api.example.com",
                 "contradicted_answer": None,
                 "should_prefer_latest": False},  # Old info still valid
            ],
        },

        # --- No contradiction (consistent info) ---
        {
            "id": "no_contradiction",
            "name": "Control: consistent facts (no contradiction)",
            "category": "control",
            "items": [
                {"text": "Python is a programming language created by Guido van Rossum.",
                 "entity": "Python", "property": "creator", "value": "Guido van Rossum",
                 "timestamp": (BASE_TIME - timedelta(days=100)).isoformat()},
                {"text": "Guido van Rossum started developing Python in the late 1980s.",
                 "entity": "Python", "property": "creation_date", "value": "late 1980s",
                 "timestamp": (BASE_TIME - timedelta(days=50)).isoformat()},
            ],
            "queries": [
                {"query": "Who created Python?",
                 "correct_answer": "Guido van Rossum",
                 "contradicted_answer": None,
                 "should_prefer_latest": False},
            ],
        },

        # --- Multi-source contradictions ---
        {
            "id": "multi_source",
            "name": "Multi-source: different sources disagree",
            "category": "multi_source",
            "items": [
                {"text": "According to the product team, the launch date is July 1st.",
                 "entity": "launch", "property": "date", "value": "July 1st",
                 "timestamp": (BASE_TIME - timedelta(days=5)).isoformat()},
                {"text": "The CEO announced the launch will be July 15th.",
                 "entity": "launch", "property": "date", "value": "July 15th",
                 "timestamp": (BASE_TIME - timedelta(days=3)).isoformat()},
                {"text": "Email from marketing: launch event planned for July 1st.",
                 "entity": "launch", "property": "date", "value": "July 1st",
                 "timestamp": (BASE_TIME - timedelta(days=1)).isoformat()},
            ],
            "queries": [
                {"query": "When is the product launch?",
                 "correct_answer": "July",  # Any mention of July is partial credit
                 "contradicted_answer": None,
                 "should_prefer_latest": True,
                 "note": "Ambiguous — multiple conflicting sources"},
            ],
        },

        # --- Negation ---
        {
            "id": "negation",
            "name": "Negation: explicit retraction",
            "category": "negation",
            "items": [
                {"text": "The bug is caused by a race condition in the queue consumer.",
                 "entity": "bug_123", "property": "cause", "value": "race condition in queue consumer",
                 "timestamp": (BASE_TIME - timedelta(days=4)).isoformat()},
                {"text": "The race condition theory was wrong. The bug is actually caused by a deadlock in the connection pool.",
                 "entity": "bug_123", "property": "cause", "value": "deadlock in connection pool",
                 "timestamp": (BASE_TIME - timedelta(days=1)).isoformat()},
            ],
            "queries": [
                {"query": "What causes bug 123?",
                 "correct_answer": "deadlock in connection pool",
                 "contradicted_answer": "race condition",
                 "should_prefer_latest": True},
            ],
        },
    ]
    return scenarios
