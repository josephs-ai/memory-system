"""
crud_rag/adapter.py — CRUD-RAG benchmark adapter.

Tests Create, Read, Update, Delete operations on a memory store.
This is a custom benchmark since there's no standard CRUD-RAG dataset —
we generate realistic memory operation sequences and verify correctness.

Eval approach:
  - Create: Insert facts, verify they're retrievable
  - Read: Query for specific facts, measure precision/recall
  - Update: Modify existing facts, verify old version is superseded
  - Delete: Remove facts, verify they're no longer retrievable
  - Conflict: Insert contradictory facts, verify latest wins
"""

from __future__ import annotations

import logging
from pathlib import Path

BENCHMARKS_DIR = Path(__file__).resolve().parent.parent
LOGGER = logging.getLogger("openclaw.benchmarks.crud_rag")


def generate_crud_scenarios() -> list[dict]:
    """Generate CRUD test scenarios.

    Each scenario is a sequence of operations with expected outcomes.
    """
    scenarios = [
        # --- CREATE scenarios ---
        {
            "id": "create_single_fact",
            "name": "Create: single fact insertion",
            "ops": [
                {"op": "create", "text": "Alice works as a data scientist at Google.",
                 "entity": "Alice", "property": "job", "value": "data scientist at Google"},
            ],
            "queries": [
                {"query": "Where does Alice work?", "expected_answer": "data scientist at Google",
                 "should_find": True},
            ],
        },
        {
            "id": "create_multiple_facts",
            "name": "Create: multiple facts for same entity",
            "ops": [
                {"op": "create", "text": "Bob lives in Tokyo, Japan.",
                 "entity": "Bob", "property": "location", "value": "Tokyo, Japan"},
                {"op": "create", "text": "Bob speaks fluent Japanese and English.",
                 "entity": "Bob", "property": "languages", "value": "Japanese and English"},
                {"op": "create", "text": "Bob works as a translator.",
                 "entity": "Bob", "property": "job", "value": "translator"},
            ],
            "queries": [
                {"query": "Where does Bob live?", "expected_answer": "Tokyo, Japan", "should_find": True},
                {"query": "What languages does Bob speak?", "expected_answer": "Japanese and English", "should_find": True},
                {"query": "What does Bob do for work?", "expected_answer": "translator", "should_find": True},
            ],
        },
        {
            "id": "create_distinct_entities",
            "name": "Create: facts about different entities",
            "ops": [
                {"op": "create", "text": "Carol is a marathon runner from Kenya.",
                 "entity": "Carol", "property": "hobby", "value": "marathon runner from Kenya"},
                {"op": "create", "text": "Dave is a chess grandmaster from Russia.",
                 "entity": "Dave", "property": "hobby", "value": "chess grandmaster from Russia"},
                {"op": "create", "text": "Eve is a concert pianist from Austria.",
                 "entity": "Eve", "property": "hobby", "value": "concert pianist from Austria"},
            ],
            "queries": [
                {"query": "What sport does Carol do?", "expected_answer": "marathon runner", "should_find": True},
                {"query": "What is Dave known for?", "expected_answer": "chess grandmaster", "should_find": True},
                {"query": "What instrument does Eve play?", "expected_answer": "concert pianist", "should_find": True},
                {"query": "Who is a professional swimmer?", "expected_answer": "", "should_find": False},
            ],
        },

        # --- UPDATE scenarios ---
        {
            "id": "update_simple",
            "name": "Update: simple fact replacement",
            "ops": [
                {"op": "create", "text": "Frank lives in Boston.",
                 "entity": "Frank", "property": "location", "value": "Boston"},
                {"op": "update", "text": "Frank moved to San Francisco.",
                 "entity": "Frank", "property": "location", "value": "San Francisco",
                 "supersedes_value": "Boston"},
            ],
            "queries": [
                {"query": "Where does Frank live?", "expected_answer": "San Francisco",
                 "should_find": True, "should_not_find": "Boston"},
            ],
        },
        {
            "id": "update_job_change",
            "name": "Update: job change",
            "ops": [
                {"op": "create", "text": "Grace works as a junior developer at a startup.",
                 "entity": "Grace", "property": "job", "value": "junior developer at a startup"},
                {"op": "update", "text": "Grace was promoted to senior engineer at Microsoft.",
                 "entity": "Grace", "property": "job", "value": "senior engineer at Microsoft",
                 "supersedes_value": "junior developer"},
            ],
            "queries": [
                {"query": "What is Grace's current job?", "expected_answer": "senior engineer at Microsoft",
                 "should_find": True},
            ],
        },
        {
            "id": "update_multiple_revisions",
            "name": "Update: multiple revisions of same fact",
            "ops": [
                {"op": "create", "text": "Hank's favorite color is blue.",
                 "entity": "Hank", "property": "favorite_color", "value": "blue"},
                {"op": "update", "text": "Hank's favorite color is now green.",
                 "entity": "Hank", "property": "favorite_color", "value": "green",
                 "supersedes_value": "blue"},
                {"op": "update", "text": "Hank's favorite color changed to purple.",
                 "entity": "Hank", "property": "favorite_color", "value": "purple",
                 "supersedes_value": "green"},
            ],
            "queries": [
                {"query": "What is Hank's favorite color?", "expected_answer": "purple",
                 "should_find": True, "should_not_find": "blue"},
            ],
        },

        # --- DELETE scenarios ---
        {
            "id": "delete_single",
            "name": "Delete: remove a fact",
            "ops": [
                {"op": "create", "text": "Iris has a pet iguana named Spike.",
                 "entity": "Iris", "property": "pet", "value": "iguana named Spike"},
                {"op": "delete", "entity": "Iris", "property": "pet"},
            ],
            "queries": [
                {"query": "What pet does Iris have?", "expected_answer": "",
                 "should_find": False},
            ],
        },
        {
            "id": "delete_selective",
            "name": "Delete: remove one fact, keep others",
            "ops": [
                {"op": "create", "text": "Jack is a pilot.",
                 "entity": "Jack", "property": "job", "value": "pilot"},
                {"op": "create", "text": "Jack lives in Miami.",
                 "entity": "Jack", "property": "location", "value": "Miami"},
                {"op": "create", "text": "Jack has two children.",
                 "entity": "Jack", "property": "family", "value": "two children"},
                {"op": "delete", "entity": "Jack", "property": "job"},
            ],
            "queries": [
                {"query": "What does Jack do for work?", "expected_answer": "",
                 "should_find": False},
                {"query": "Where does Jack live?", "expected_answer": "Miami",
                 "should_find": True},
                {"query": "Does Jack have children?", "expected_answer": "two children",
                 "should_find": True},
            ],
        },

        # --- READ (retrieval quality) scenarios ---
        {
            "id": "read_specific_vs_general",
            "name": "Read: specific query among many facts",
            "ops": [
                {"op": "create", "text": "Karen studied medicine at Johns Hopkins University.",
                 "entity": "Karen", "property": "education", "value": "medicine at Johns Hopkins"},
                {"op": "create", "text": "Karen completed her residency at Mayo Clinic.",
                 "entity": "Karen", "property": "training", "value": "residency at Mayo Clinic"},
                {"op": "create", "text": "Karen specializes in pediatric cardiology.",
                 "entity": "Karen", "property": "specialty", "value": "pediatric cardiology"},
                {"op": "create", "text": "Karen published 15 research papers.",
                 "entity": "Karen", "property": "publications", "value": "15 research papers"},
                {"op": "create", "text": "Karen volunteers at a free clinic on weekends.",
                 "entity": "Karen", "property": "volunteering", "value": "free clinic on weekends"},
            ],
            "queries": [
                {"query": "Where did Karen study?", "expected_answer": "Johns Hopkins", "should_find": True},
                {"query": "What is Karen's medical specialty?", "expected_answer": "pediatric cardiology", "should_find": True},
                {"query": "How many papers has Karen published?", "expected_answer": "15 research papers", "should_find": True},
            ],
        },
        {
            "id": "read_disambiguation",
            "name": "Read: disambiguating similar entities",
            "ops": [
                {"op": "create", "text": "Leo the lion tamer works at the circus.",
                 "entity": "Leo (lion tamer)", "property": "job", "value": "lion tamer at circus"},
                {"op": "create", "text": "Leo the software developer works at Amazon.",
                 "entity": "Leo (developer)", "property": "job", "value": "software developer at Amazon"},
                {"op": "create", "text": "Leo the chef runs an Italian restaurant.",
                 "entity": "Leo (chef)", "property": "job", "value": "chef at Italian restaurant"},
            ],
            "queries": [
                {"query": "Which Leo works at Amazon?", "expected_answer": "software developer", "should_find": True},
                {"query": "Which Leo works at the circus?", "expected_answer": "lion tamer", "should_find": True},
            ],
        },

        # --- CONFLICT / CONTRADICTION scenarios ---
        {
            "id": "conflict_temporal",
            "name": "Conflict: temporal contradiction (latest should win)",
            "ops": [
                {"op": "create", "text": "As of January 2024, the project uses Python 3.10.",
                 "entity": "project", "property": "python_version", "value": "Python 3.10",
                 "timestamp": "2024-01-15T00:00:00Z"},
                {"op": "create", "text": "As of March 2024, the project upgraded to Python 3.12.",
                 "entity": "project", "property": "python_version", "value": "Python 3.12",
                 "timestamp": "2024-03-15T00:00:00Z"},
            ],
            "queries": [
                {"query": "What Python version does the project use?",
                 "expected_answer": "Python 3.12", "should_find": True},
            ],
        },
        {
            "id": "conflict_contradictory",
            "name": "Conflict: directly contradictory facts",
            "ops": [
                {"op": "create", "text": "The meeting is on Tuesday at 3pm.",
                 "entity": "meeting", "property": "schedule", "value": "Tuesday at 3pm"},
                {"op": "create", "text": "The meeting was rescheduled to Thursday at 2pm.",
                 "entity": "meeting", "property": "schedule", "value": "Thursday at 2pm",
                 "supersedes_value": "Tuesday at 3pm"},
            ],
            "queries": [
                {"query": "When is the meeting?", "expected_answer": "Thursday at 2pm",
                 "should_find": True, "should_not_find": "Tuesday"},
            ],
        },
    ]
    return scenarios
