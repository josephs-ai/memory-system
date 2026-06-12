"""
temporalqa/generator.py — Programmatic generator for temporal QA scenarios.

Generates randomized "latest" scenarios from templates to scale the benchmark
beyond hand-crafted examples.
"""
from __future__ import annotations

import random
from datetime import datetime, timedelta, timezone

BASE_TIME = datetime(2026, 6, 1, 12, 0, 0, tzinfo=timezone.utc)

# Templates: (entity, property, old_text, old_value, new_text, new_value, query, expected_substr)
LATEST_TEMPLATES = [
    ("user", "email_provider", "User's email is on Gmail.", "Gmail",
     "User switched to ProtonMail for privacy.", "ProtonMail",
     "What email provider does the user use?", "ProtonMail"),
    ("server", "os", "Server runs Ubuntu 20.04 LTS.", "Ubuntu 20.04",
     "Server upgraded to Ubuntu 24.04 LTS.", "Ubuntu 24.04",
     "What OS does the server run?", "Ubuntu 24.04"),
    ("api", "auth_method", "API uses basic auth for authentication.", "basic auth",
     "API switched to OAuth 2.0 with JWT tokens.", "OAuth 2.0",
     "How does the API handle authentication?", "OAuth"),
    ("database", "backup_schedule", "Database backups run weekly on Sundays.", "weekly on Sundays",
     "Backup schedule changed to daily at 2am.", "daily at 2am",
     "When do database backups run?", "daily"),
    ("website", "cdn", "Website uses CloudFlare CDN.", "CloudFlare",
     "Switched CDN to Fastly for edge compute.", "Fastly",
     "What CDN does the website use?", "Fastly"),
    ("monitoring", "tool", "Infrastructure monitoring uses Nagios.", "Nagios",
     "Migrated monitoring to Datadog for better dashboards.", "Datadog",
     "What monitoring tool is used?", "Datadog"),
    ("team", "standup_time", "Daily standup is at 9am.", "9am",
     "Standup moved to 10:30am to accommodate remote team.", "10:30am",
     "What time is the daily standup?", "10:30"),
    ("project", "license", "Project uses MIT license.", "MIT",
     "License changed to Apache 2.0 for patent protection.", "Apache 2.0",
     "What license does the project use?", "Apache"),
    ("user", "editor", "User codes in VS Code.", "VS Code",
     "User switched to Neovim for speed.", "Neovim",
     "What code editor does the user use?", "Neovim"),
    ("cluster", "orchestrator", "Cluster managed by Docker Swarm.", "Docker Swarm",
     "Migrated cluster orchestration to Kubernetes.", "Kubernetes",
     "What orchestration tool manages the cluster?", "Kubernetes"),
    ("frontend", "framework", "Frontend built with React 17.", "React 17",
     "Frontend migrated to Svelte for smaller bundles.", "Svelte",
     "What frontend framework is used?", "Svelte"),
    ("storage", "provider", "File storage on Google Cloud Storage.", "Google Cloud Storage",
     "Moved file storage to MinIO for on-prem.", "MinIO",
     "Where are files stored?", "MinIO"),
    ("team", "size", "Engineering team has 12 members.", "12 members",
     "After hiring sprint, team grew to 18 members.", "18 members",
     "How many people are on the engineering team?", "18"),
    ("service", "language", "Service written in Java.", "Java",
     "Service rewritten in Go for better concurrency.", "Go",
     "What language is the service written in?", "Go"),
    ("build", "tool", "Build system uses Maven.", "Maven",
     "Switched build tool to Gradle for flexibility.", "Gradle",
     "What build tool does the project use?", "Gradle"),
    ("queue", "broker", "Message queue uses RabbitMQ.", "RabbitMQ",
     "Migrated message queue to Apache Kafka.", "Apache Kafka",
     "What message broker is used?", "Kafka"),
    ("dns", "provider", "DNS managed by GoDaddy.", "GoDaddy",
     "DNS moved to Cloudflare for DNSSEC support.", "Cloudflare",
     "Who manages the DNS?", "Cloudflare"),
    ("logging", "platform", "Logs shipped to ELK stack.", "ELK stack",
     "Switched logging to Grafana Loki for cost savings.", "Grafana Loki",
     "What logging platform is used?", "Loki"),
    ("secrets", "manager", "Secrets stored in AWS Secrets Manager.", "AWS Secrets Manager",
     "Migrated secrets to HashiCorp Vault for multi-cloud.", "HashiCorp Vault",
     "Where are secrets stored?", "Vault"),
    ("proxy", "software", "Reverse proxy is Nginx.", "Nginx",
     "Switched to Caddy for automatic HTTPS.", "Caddy",
     "What reverse proxy is used?", "Caddy"),
    ("cache", "system", "Caching layer uses Memcached.", "Memcached",
     "Replaced Memcached with Redis for data structures.", "Redis",
     "What caching system is used?", "Redis"),
    ("testing", "framework", "Tests written with unittest.", "unittest",
     "Migrated test suite to pytest for better fixtures.", "pytest",
     "What testing framework is used?", "pytest"),
    ("container", "registry", "Container images on Docker Hub.", "Docker Hub",
     "Moved container registry to GitHub Container Registry.", "GitHub Container Registry",
     "Where are container images stored?", "GitHub Container"),
    ("search", "engine", "Search powered by Elasticsearch.", "Elasticsearch",
     "Replaced search with Meilisearch for simplicity.", "Meilisearch",
     "What search engine is used?", "Meilisearch"),
    ("vpn", "provider", "Team VPN is OpenVPN.", "OpenVPN",
     "Switched VPN to WireGuard for performance.", "WireGuard",
     "What VPN does the team use?", "WireGuard"),
    ("analytics", "tool", "Analytics tracked with Google Analytics.", "Google Analytics",
     "Switched to Plausible Analytics for privacy.", "Plausible",
     "What analytics tool is used?", "Plausible"),
    ("user", "phone", "User's phone number is 555-0100.", "555-0100",
     "User updated phone to 555-0200.", "555-0200",
     "What is the user's phone number?", "555-0200"),
    ("sprint", "goal", "Sprint goal is to finish the auth module.", "auth module",
     "Sprint goal updated to ship the billing integration.", "billing integration",
     "What is the current sprint goal?", "billing"),
    ("infra", "terraform_version", "Infrastructure uses Terraform 1.3.", "Terraform 1.3",
     "Upgraded to Terraform 1.7 for import blocks.", "Terraform 1.7",
     "What version of Terraform is used?", "1.7"),
    ("payments", "processor", "Payments processed by Stripe.", "Stripe",
     "Switched payment processing to Adyen for global coverage.", "Adyen",
     "What payment processor is used?", "Adyen"),
]


def generate_latest_scenarios(count: int = 100, seed: int = 42) -> list[dict]:
    """Generate `count` latest-type temporal QA scenarios from templates."""
    rng = random.Random(seed)

    # Shuffle and cycle through templates
    templates = list(LATEST_TEMPLATES)
    rng.shuffle(templates)

    scenarios = []
    for i in range(count):
        t = templates[i % len(templates)]
        entity, prop, old_text, old_val, new_text, new_val, query, expected = t

        # Randomize time gaps
        old_days = rng.randint(20, 180)
        new_days = rng.randint(1, 15)

        scenarios.append({
            "id": f"gen_latest_{i:03d}",
            "name": f"Generated latest #{i}: {entity}.{prop}",
            "items": [
                {"text": old_text, "entity": entity, "property": prop,
                 "value": old_val,
                 "timestamp": (BASE_TIME - timedelta(days=old_days)).isoformat()},
                {"text": new_text, "entity": entity, "property": prop,
                 "value": new_val,
                 "timestamp": (BASE_TIME - timedelta(days=new_days)).isoformat()},
            ],
            "queries": [
                {"query": query, "expected": expected, "temporal_hint": "latest"},
            ],
        })

    return scenarios
