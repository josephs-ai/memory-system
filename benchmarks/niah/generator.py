"""
niah/generator.py — Needle-in-a-Haystack synthetic benchmark generator.

Generates synthetic memory stores of varying sizes with "needles" (specific
facts) planted at different positions (depths) among "haystack" distractors.

Needle format:
    "The secret code for <entity> is <value>."
    "The birthdate of <entity> is <value>."
    etc.

Haystack items are plausible but non-needle facts about various topics.
"""

from __future__ import annotations

import random
import string
import uuid
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Needle templates
# ---------------------------------------------------------------------------

NEEDLE_TEMPLATES = [
    "The secret passphrase for {entity} is '{value}'.",
    "The registration number of {entity} is {value}.",
    "The unique identifier assigned to {entity} is {value}.",
    "The access code for {entity} account is {value}.",
    "The project codename for {entity} is '{value}'.",
    "The meeting room reserved for {entity} is {value}.",
    "The API key prefix for {entity} service is {value}.",
    "The employee badge number for {entity} is {value}.",
    "The serial number of {entity} device is {value}.",
    "The version tag for {entity} release is {value}.",
]

NEEDLE_QUERY_TEMPLATES = [
    "What is the secret passphrase for {entity}?",
    "What is the registration number of {entity}?",
    "What is the unique identifier assigned to {entity}?",
    "What is the access code for {entity} account?",
    "What is the project codename for {entity}?",
    "What is the meeting room reserved for {entity}?",
    "What is the API key prefix for {entity} service?",
    "What is the employee badge number for {entity}?",
    "What is the serial number of {entity} device?",
    "What is the version tag for {entity} release?",
]

# Haystack templates — plausible facts unrelated to needles
HAYSTACK_TEMPLATES = [
    "{entity} prefers to work in the morning hours.",
    "{entity} has been using the system since last year.",
    "The team {entity} belongs to focuses on infrastructure.",
    "{entity} submitted a report on {topic} last quarter.",
    "Recent analysis shows {entity} usage has increased by {pct}%.",
    "The server {entity} was last restarted on a Tuesday.",
    "Configuration for {entity} follows the standard template.",
    "{entity} has {count} active connections at peak load.",
    "The {entity} module was refactored in the last sprint.",
    "Documentation for {entity} was updated recently.",
    "The backup schedule for {entity} runs every 6 hours.",
    "{entity} integrates with the main authentication service.",
    "Load balancing for {entity} uses round-robin strategy.",
    "The {entity} database has {count} thousand records.",
    "Monitoring alerts for {entity} are routed to the ops team.",
    "{entity} supports both REST and GraphQL interfaces.",
    "The cache TTL for {entity} is set to 300 seconds.",
    "{entity} logs are retained for 90 days by policy.",
    "The {entity} service mesh uses mutual TLS.",
    "Circuit breaker for {entity} triggers at 50% error rate.",
]

ENTITY_POOL = [
    "Project Alpha", "Project Beta", "Project Gamma", "Project Delta",
    "Project Epsilon", "Project Zeta", "Project Eta", "Project Theta",
    "Service Omega", "Service Lambda", "Service Sigma", "Service Tau",
    "Module Kappa", "Module Phi", "Module Chi", "Module Psi",
    "User Alice", "User Bob", "User Carol", "User Dave",
    "User Eve", "User Frank", "User Grace", "User Hank",
    "Team Phoenix", "Team Orion", "Team Vega", "Team Lyra",
    "System Nexus", "System Apex", "System Core", "System Edge",
]

TOPIC_POOL = ["performance", "security", "reliability", "scalability", "cost"]
PCT_POOL = [12, 18, 23, 31, 45, 67]
COUNT_POOL = [50, 100, 250, 500, 1000, 5000]


def _random_value() -> str:
    """Generate a unique, memorable value string."""
    chars = random.choices(string.ascii_uppercase + string.digits, k=8)
    return "".join(chars[:4]) + "-" + "".join(chars[4:])


@dataclass
class Needle:
    needle_id: str
    entity: str
    value: str
    text: str          # The memory item text
    query: str         # The query to retrieve it
    template_idx: int  # Which template was used


@dataclass
class NIAHConfig:
    store_size: int = 100           # Total items in store (including needle)
    num_needles: int = 10           # Number of needles to plant
    needle_positions: list[str] = field(
        default_factory=lambda: ["start", "middle", "end"]
    )
    distractor_similarity: str = "low"  # "low" | "medium" | "high"
    seed: int = 42


def generate_haystack_item(idx: int) -> str:
    template = random.choice(HAYSTACK_TEMPLATES)
    entity = random.choice(ENTITY_POOL)
    topic = random.choice(TOPIC_POOL)
    pct = random.choice(PCT_POOL)
    count = random.choice(COUNT_POOL)
    return template.format(entity=entity, topic=topic, pct=pct, count=count)


def generate_needle(entity: str, template_idx: int) -> Needle:
    value = _random_value()
    tmpl = NEEDLE_TEMPLATES[template_idx % len(NEEDLE_TEMPLATES)]
    qtmpl = NEEDLE_QUERY_TEMPLATES[template_idx % len(NEEDLE_QUERY_TEMPLATES)]
    text = tmpl.format(entity=entity, value=value)
    query = qtmpl.format(entity=entity)
    return Needle(
        needle_id=f"needle_{uuid.uuid4().hex[:8]}",
        entity=entity,
        value=value,
        text=text,
        query=query,
        template_idx=template_idx,
    )


def build_niah_dataset(config: NIAHConfig) -> dict:
    """
    Build the full NIAH dataset: a list of memory texts plus needle specs.

    Returns:
        {
            "config": {...},
            "items": [{"id": str, "text": str, "is_needle": bool, "needle_id": str|None}],
            "needles": [Needle],
        }
    """
    random.seed(config.seed)

    # Pick distinct entities for needles
    needle_entities = random.sample(ENTITY_POOL, min(config.num_needles, len(ENTITY_POOL)))

    needles: list[Needle] = []
    for i, entity in enumerate(needle_entities):
        needles.append(generate_needle(entity, i))

    num_haystack = config.store_size - config.num_needles
    if num_haystack < 0:
        num_haystack = 0

    haystack_texts = [generate_haystack_item(i) for i in range(num_haystack)]

    # Determine insertion positions for needles based on needle_positions config
    items = []
    haystack_idx = 0
    total = num_haystack + len(needles)

    # Assign each needle a position index
    needle_positions_map: dict[str, int] = {}
    for needle in needles:
        pos_type = random.choice(config.needle_positions)
        if pos_type == "start":
            pos = random.randint(0, total // 5)
        elif pos_type == "middle":
            pos = random.randint(total // 3, 2 * total // 3)
        else:  # end
            pos = random.randint(4 * total // 5, total - 1)
        needle_positions_map[needle.needle_id] = min(pos, total - 1)

    # Sort needles by position
    sorted_needles = sorted(needles, key=lambda n: needle_positions_map[n.needle_id])
    needle_queue = list(sorted_needles)
    needle_pos_queue = [needle_positions_map[n.needle_id] for n in needle_queue]

    global_idx = 0
    nq_ptr = 0

    while global_idx < total:
        # Check if a needle should be inserted here
        while nq_ptr < len(needle_queue) and needle_pos_queue[nq_ptr] <= global_idx:
            n = needle_queue[nq_ptr]
            items.append({
                "id": n.needle_id,
                "text": n.text,
                "is_needle": True,
                "needle_id": n.needle_id,
            })
            nq_ptr += 1
            global_idx += 1

        if haystack_idx < len(haystack_texts):
            items.append({
                "id": f"hay_{uuid.uuid4().hex[:8]}",
                "text": haystack_texts[haystack_idx],
                "is_needle": False,
                "needle_id": None,
            })
            haystack_idx += 1
            global_idx += 1
        elif nq_ptr < len(needle_queue):
            # Flush remaining needles
            n = needle_queue[nq_ptr]
            items.append({
                "id": n.needle_id,
                "text": n.text,
                "is_needle": True,
                "needle_id": n.needle_id,
            })
            nq_ptr += 1
            global_idx += 1
        else:
            break

    return {
        "config": {
            "store_size": config.store_size,
            "num_needles": config.num_needles,
            "needle_positions": config.needle_positions,
            "distractor_similarity": config.distractor_similarity,
            "seed": config.seed,
            "actual_items": len(items),
        },
        "items": items,
        "needles": needles,
    }
