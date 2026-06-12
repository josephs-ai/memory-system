"""
crud_rag/generator.py — Programmatic generator for CRUD-RAG scenarios.

Generates randomized update/conflict scenarios from templates.
"""
from __future__ import annotations

import random

# (entity, property, old_text, old_value, new_text, new_value, query, expected, should_not_find)
UPDATE_TEMPLATES = [
    ("user", "city", "User lives in Boston.", "Boston",
     "User moved to Denver.", "Denver",
     "Where does the user live?", "Denver", "Boston"),
    ("server", "cpu", "Server has 8 CPU cores.", "8 cores",
     "Server upgraded to 32 CPU cores.", "32 cores",
     "How many CPU cores does the server have?", "32", "8"),
    ("project", "budget", "Project budget is $50,000.", "$50,000",
     "Project budget increased to $80,000.", "$80,000",
     "What is the project budget?", "$80,000", "$50,000"),
    ("team", "lead", "Team lead is Alex.", "Alex",
     "Team lead changed to Jordan.", "Jordan",
     "Who is the team lead?", "Jordan", "Alex"),
    ("app", "version", "App is on version 2.3.", "version 2.3",
     "App updated to version 3.0.", "version 3.0",
     "What version is the app on?", "3.0", "2.3"),
    ("user", "plan", "User is on the free plan.", "free plan",
     "User upgraded to the pro plan.", "pro plan",
     "What plan is the user on?", "pro", "free"),
    ("office", "hours", "Office hours are 9am-5pm.", "9am-5pm",
     "Office hours changed to 10am-6pm.", "10am-6pm",
     "What are the office hours?", "10am-6pm", "9am-5pm"),
    ("server", "ram", "Server has 16GB RAM.", "16GB RAM",
     "Server upgraded to 64GB RAM.", "64GB RAM",
     "How much RAM does the server have?", "64GB", "16GB"),
    ("user", "role", "User role is contributor.", "contributor",
     "User promoted to admin role.", "admin",
     "What is the user's role?", "admin", "contributor"),
    ("project", "deadline", "Project due March 15.", "March 15",
     "Project deadline extended to April 30.", "April 30",
     "When is the project due?", "April 30", "March 15"),
    ("app", "theme", "App uses blue theme.", "blue theme",
     "App theme changed to dark mode.", "dark mode",
     "What theme does the app use?", "dark mode", "blue"),
    ("team", "count", "Team has 5 developers.", "5 developers",
     "Team expanded to 9 developers.", "9 developers",
     "How many developers on the team?", "9", "5"),
    ("user", "subscription", "User subscribes monthly.", "monthly",
     "User switched to annual subscription.", "annual",
     "What subscription does the user have?", "annual", "monthly"),
    ("server", "location", "Server in us-east-1.", "us-east-1",
     "Server migrated to eu-west-1.", "eu-west-1",
     "What region is the server in?", "eu-west-1", "us-east-1"),
    ("project", "priority", "Project priority is medium.", "medium",
     "Project escalated to critical priority.", "critical",
     "What is the project priority?", "critical", "medium"),
    ("api", "rate_limit", "API rate limit is 100 req/min.", "100 req/min",
     "API rate limit increased to 500 req/min.", "500 req/min",
     "What is the API rate limit?", "500", "100"),
    ("user", "language", "User interface set to English.", "English",
     "User changed interface to Spanish.", "Spanish",
     "What language is the interface set to?", "Spanish", "English"),
    ("service", "port", "Service runs on port 8080.", "port 8080",
     "Service moved to port 3000.", "port 3000",
     "What port does the service run on?", "3000", "8080"),
    ("team", "sprint_length", "Sprint length is 1 week.", "1 week",
     "Sprint length changed to 2 weeks.", "2 weeks",
     "How long is the sprint?", "2 weeks", "1 week"),
    ("db", "engine", "Database engine is InnoDB.", "InnoDB",
     "Switched database engine to RocksDB.", "RocksDB",
     "What database engine is used?", "RocksDB", "InnoDB"),
]


def generate_update_scenarios(count: int = 100, seed: int = 42) -> list[dict]:
    """Generate CRUD update/conflict scenarios from templates."""
    rng = random.Random(seed)
    templates = list(UPDATE_TEMPLATES)
    rng.shuffle(templates)

    scenarios = []
    for i in range(count):
        t = templates[i % len(templates)]
        entity, prop, old_text, old_val, new_text, new_val, query, expected, should_not = t

        scenarios.append({
            "id": f"gen_update_{i:03d}",
            "name": f"Generated update #{i}: {entity}.{prop}",
            "ops": [
                {"op": "create", "text": old_text,
                 "entity": entity, "property": prop, "value": old_val},
                {"op": "update", "text": new_text,
                 "entity": entity, "property": prop, "value": new_val,
                 "supersedes_value": old_val},
            ],
            "queries": [
                {"query": query, "expected_answer": expected,
                 "should_find": True, "should_not_find": should_not},
            ],
        })

    return scenarios
