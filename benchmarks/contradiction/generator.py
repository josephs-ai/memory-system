"""
contradiction/generator.py — Programmatic generator for contradiction scenarios.

Generates randomized direct contradiction scenarios from templates.
"""
from __future__ import annotations

import random
from datetime import datetime, timedelta, timezone

BASE_TIME = datetime(2026, 6, 1, 12, 0, 0, tzinfo=timezone.utc)

# (entity, property, old_text, old_value, new_text, new_value, query, expected_substr)
DIRECT_TEMPLATES = [
    ("user", "language", "User's primary language is Python.", "Python",
     "User switched to Rust as their primary language.", "Rust",
     "What programming language does the user prefer?", "Rust"),
    ("server", "provider", "Server hosted on AWS.", "AWS",
     "Migrated server to Google Cloud.", "Google Cloud",
     "Where is the server hosted?", "Google Cloud"),
    ("team", "methodology", "Team follows waterfall methodology.", "waterfall",
     "Team adopted agile/scrum methodology.", "agile/scrum",
     "What development methodology does the team use?", "agile"),
    ("app", "database", "App uses MySQL for data storage.", "MySQL",
     "Migrated app database to PostgreSQL.", "PostgreSQL",
     "What database does the app use?", "PostgreSQL"),
    ("user", "timezone", "User works in EST timezone.", "EST",
     "User relocated, now works in PST timezone.", "PST",
     "What timezone does the user work in?", "PST"),
    ("project", "repo", "Project hosted on GitLab.", "GitLab",
     "Moved project repository to GitHub.", "GitHub",
     "Where is the project repository?", "GitHub"),
    ("company", "ceo", "Company CEO is John Smith.", "John Smith",
     "New CEO appointed: Jane Doe.", "Jane Doe",
     "Who is the company CEO?", "Jane Doe"),
    ("system", "cache", "System uses Redis for caching.", "Redis",
     "Replaced Redis with Memcached for caching.", "Memcached",
     "What caching system is used?", "Memcached"),
    ("deploy", "strategy", "Deployments use blue-green strategy.", "blue-green",
     "Switched to canary deployments for safer rollouts.", "canary",
     "What deployment strategy is used?", "canary"),
    ("api", "format", "API returns XML responses.", "XML",
     "API migrated to JSON response format.", "JSON",
     "What format does the API use?", "JSON"),
    ("infra", "iac_tool", "Infrastructure managed with CloudFormation.", "CloudFormation",
     "Switched infrastructure to Pulumi.", "Pulumi",
     "What IaC tool manages the infrastructure?", "Pulumi"),
    ("team", "chat_tool", "Team communicates via Slack.", "Slack",
     "Team moved to Microsoft Teams.", "Microsoft Teams",
     "What chat tool does the team use?", "Teams"),
    ("service", "protocol", "Service uses REST API.", "REST",
     "Service migrated to gRPC for performance.", "gRPC",
     "What protocol does the service use?", "gRPC"),
    ("backup", "location", "Backups stored on local NAS.", "local NAS",
     "Backups moved to S3 Glacier for cost savings.", "S3 Glacier",
     "Where are backups stored?", "S3 Glacier"),
    ("auth", "provider", "Authentication handled by Auth0.", "Auth0",
     "Switched auth to Keycloak for self-hosting.", "Keycloak",
     "What handles authentication?", "Keycloak"),
    ("docs", "platform", "Documentation on Confluence.", "Confluence",
     "Migrated docs to Notion.", "Notion",
     "Where is the documentation?", "Notion"),
    ("ci", "runner", "CI runs on self-hosted Jenkins runners.", "Jenkins runners",
     "Moved CI to GitHub Actions runners.", "GitHub Actions runners",
     "What CI runners are used?", "GitHub Actions"),
    ("frontend", "styling", "Frontend styled with Bootstrap CSS.", "Bootstrap",
     "Switched styling to Tailwind CSS.", "Tailwind CSS",
     "What CSS framework styles the frontend?", "Tailwind"),
    ("user", "os", "User's machine runs macOS.", "macOS",
     "User switched to Linux (Fedora) as daily driver.", "Linux Fedora",
     "What OS does the user run?", "Linux"),
    ("db", "hosting", "Database hosted on RDS.", "RDS",
     "Moved database to self-managed on EC2.", "self-managed on EC2",
     "How is the database hosted?", "self-managed"),
    ("container", "runtime", "Containers run on Docker.", "Docker",
     "Switched container runtime to Podman.", "Podman",
     "What container runtime is used?", "Podman"),
    ("network", "firewall", "Network protected by pfSense.", "pfSense",
     "Replaced firewall with OPNsense.", "OPNsense",
     "What firewall protects the network?", "OPNsense"),
    ("project", "board", "Project tracked in Jira.", "Jira",
     "Moved project tracking to Linear.", "Linear",
     "What project management tool is used?", "Linear"),
    ("api", "gateway", "API gateway is Kong.", "Kong",
     "Switched API gateway to Traefik.", "Traefik",
     "What API gateway is used?", "Traefik"),
    ("user", "browser", "User's primary browser is Chrome.", "Chrome",
     "User switched to Firefox for privacy.", "Firefox",
     "What browser does the user use?", "Firefox"),
    ("observability", "tracing", "Distributed tracing with Jaeger.", "Jaeger",
     "Migrated tracing to Grafana Tempo.", "Grafana Tempo",
     "What distributed tracing tool is used?", "Tempo"),
    ("team", "meeting_tool", "Team video calls on Zoom.", "Zoom",
     "Switched video calls to Google Meet.", "Google Meet",
     "What tool does the team use for video calls?", "Google Meet"),
    ("service", "runtime", "Service runs on Node.js 18.", "Node.js 18",
     "Upgraded service to Node.js 22.", "Node.js 22",
     "What Node.js version runs the service?", "Node.js 22"),
    ("user", "shell", "User uses bash shell.", "bash",
     "User switched to zsh with oh-my-zsh.", "zsh",
     "What shell does the user use?", "zsh"),
    ("data", "warehouse", "Data warehouse is Snowflake.", "Snowflake",
     "Migrated data warehouse to ClickHouse.", "ClickHouse",
     "What data warehouse is used?", "ClickHouse"),
]


def generate_direct_scenarios(count: int = 100, seed: int = 42) -> list[dict]:
    """Generate `count` direct contradiction scenarios from templates."""
    rng = random.Random(seed)
    templates = list(DIRECT_TEMPLATES)
    rng.shuffle(templates)

    scenarios = []
    for i in range(count):
        t = templates[i % len(templates)]
        entity, prop, old_text, old_val, new_text, new_val, query, expected = t

        old_days = rng.randint(20, 120)
        new_days = rng.randint(1, 10)

        scenarios.append({
            "id": f"gen_direct_{i:03d}",
            "name": f"Generated direct #{i}: {entity}.{prop}",
            "category": "direct",
            "items": [
                {"text": old_text, "entity": entity, "property": prop,
                 "value": old_val,
                 "timestamp": (BASE_TIME - timedelta(days=old_days)).isoformat()},
                {"text": new_text, "entity": entity, "property": prop,
                 "value": new_val,
                 "timestamp": (BASE_TIME - timedelta(days=new_days)).isoformat()},
            ],
            "queries": [
                {"query": query, "correct_answer": expected},
            ],
        })

    return scenarios
