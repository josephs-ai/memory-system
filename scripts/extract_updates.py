import re
import json
import hashlib
import argparse
from datetime import datetime, timezone
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("input_path")
parser.add_argument("--source-agent", default="unknown")
parser.add_argument("--source-session", default="unknown")
parser.add_argument("--source-chunk", default=None)
args = parser.parse_args()

path = Path(args.input_path)
text = path.read_text(encoding="utf-8", errors="ignore")

now = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def make_id(source_name: str, fact_text: str) -> str:
    h = hashlib.sha1(f"{source_name}|{fact_text}".encode("utf-8")).hexdigest()[:12]
    return f"mem-{h}"


def clean_line(s: str) -> str:
    s = s.strip()
    s = re.sub(r"^(USER|ASSISTANT):\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"^(did u know that|did you know that)\s+", "", s, flags=re.IGNORECASE)
    s = re.sub(r"^(just so you know)\s+", "", s, flags=re.IGNORECASE)
    return s.strip()


def normalize_fact_text(s: str) -> str:
    s = clean_line(s)
    s = re.sub(r"\s+", " ", s).strip()
    if s:
        s = s[0].upper() + s[1:]
    return s


def is_question_like(s: str) -> bool:
    t = s.lower().strip()
    return (
        t.endswith("?")
        or t.startswith("who ")
        or t.startswith("what ")
        or t.startswith("hello")
        or t.startswith("hey")
        or t.startswith("did u know")
        or t.startswith("did you know")
        or "what should we call" in t
        or "who am i" in t
        or "who are you" in t
    )


def is_low_value_line(s: str) -> bool:
    t = s.lower().strip()
    low_patterns = [
        r"^hello$",
        r"^hello\?$",
        r"^hey[!.]?$",
        r"blank slate",
        r"just woke up",
        r"who am i",
        r"who are you",
        r"what should my name be",
        r"what should we call me",
        r"what should i call you",
        r"good to know",
        r"we got distracted",
        r"properly introduced",
        r"i'm here",
        r"i didnt",
        r"i didn't",
    ]
    return any(re.search(p, t) for p in low_patterns)


def durable_claim_score(s: str) -> int:
    t = s.lower().strip()
    score = 0

    good_patterns = [
        (r"\bnow uses\b", 5),
        (r"\buses\b", 3),
        (r"\bsupports\b", 4),
        (r"\binstead of\b", 5),
        (r"\bpolicy\b", 4),
        (r"\bmemory\b", 4),
        (r"\bcanonical\b", 4),
        (r"\bregistry\b", 5),
        (r"\bactive items\b", 4),
        (r"\bstructured\b", 3),
        (r"\bconfigured\b", 3),
        (r"\bcapability\b", 3),
        (r"\bworks through\b", 4),
        (r"\bdecision\b", 3),
        (r"\barchitecture\b", 4),
        (r"\bcheckpoint\b", 3),
        (r"\bdehydration\b", 3),
        (r"\btranscript\b", 3),
        (r"\bpreference\b", 3),
        (r"\bprefers\b", 4),
        (r"\bcurrent\b", 2),
        (r"\buses a structured registry\b", 6),
    ]

    for pattern, pts in good_patterns:
        if re.search(pattern, t):
            score += pts

    bad_patterns = [
        (r"\?$", -6),
        (r"\bhello\b", -5),
        (r"\bhey\b", -4),
        (r"\bwho am i\b", -8),
        (r"\bwho are you\b", -8),
        (r"\bwhat should\b", -6),
        (r"\bcall me\b", -5),
        (r"\bcall you\b", -5),
        (r"\bblank slate\b", -6),
        (r"\bjust woke up\b", -6),
    ]

    for pattern, pts in bad_patterns:
        if re.search(pattern, t):
            score += pts

    if not is_question_like(s) and len(t.split()) >= 6:
        score += 2

    return score


def classify_item(fact_text: str, source_name: str):
    t = fact_text.lower()

    memory_type = "fact"
    scope = "stable"
    entity = "openclaw"
    prop = None
    value = None
    confidence = 0.80
    tags = []
    notes = "Structured candidate extracted from chunk."

    if "browser" in t or "devtools" in t or "cdp" in t:
        entity = "browser"
        tags += ["browser"]
        confidence = max(confidence, 0.90)

    if "browser-control" in t or "browser control" in t:
        prop = "capability"
        value = "browser_control_available"
        tags += ["capability"]

    if "cli" in t:
        tags += ["cli"]

    if "tool" in t and "restricted" in t:
        prop = "tool_access_mode"
        value = "restricted_or_cli_fallback"
        tags += ["tool-access"]
        confidence = max(confidence, 0.91)

    if "transcript reader" in t or "dehydrator" in t or "cross-agent" in t:
        entity = "checkpoint_pipeline"
        memory_type = "learned_fix"
        prop = "processing_capability"
        value = "cross_agent_supported"
        tags += ["checkpoint", "transcript", "cross-agent"]
        confidence = max(confidence, 0.88)

    if "preference" in t or "joseph prefers" in t or "prefers" in t:
        memory_type = "preference"
        entity = "user_preference"
        scope = "stable"
        confidence = max(confidence, 0.90)
        tags += ["preference"]

    if "memory" in t or "canonical" in t or "registry" in t:
        entity = "checkpoint_pipeline"
        tags += ["memory-system"]

        if "now uses" in t or "instead of" in t:
            memory_type = "decision"
            confidence = max(confidence, 0.92)
            tags += ["architecture", "decision"]

        if "registry" in t:
            prop = prop or "canonical_memory_backend"
            value = value or "structured_registry"
            tags += ["registry"]

        if "markdown bullets" in t or "markdown bullet" in t:
            value = "structured_registry_not_markdown_inference"
            tags += ["markdown-inference-removed"]

    return {
        "id": make_id(source_name, fact_text),
        "text": fact_text,
        "memory_type": memory_type,
        "scope": scope,
        "entity": entity,
        "property": prop,
        "value": value,
        "status": "active",
        "confidence": round(confidence, 2),
        "source_agent": args.source_agent,
        "source_session": args.source_session,
        "source_chunk": args.source_chunk or source_name,
        "first_seen": now,
        "last_confirmed": now,
        "supersedes": None,
        "tags": sorted(set(tags)),
        "notes": notes,
    }


facts = []

# Keep only a small set of proven high-value exact rules.
known_rules = [
    (
        r"functionality to control browser|browser automation capabilities|native `browser` tool|chrome devtools protocol|execute the `openclaw browser` cli commands",
        "OpenClaw has browser-control capability through browser automation features and CLI/browser-tool pathways.",
    ),
    (
        r"transcript reader supports|--all-agents|--agent|--file",
        "Transcript reader supports direct file mode, per-agent latest-session mode, and cross-agent discovery mode.",
    ),
    (
        r"dehydration strips|sender wrappers|heartbeat boilerplate|system gateway spam|tool results|assistant tool-call content",
        "Transcript dehydration removes sender wrappers, heartbeat/system noise, tool results, and assistant tool-call content.",
    ),
    (
        r"gateway\.md first|worker\.md first|ranks gateway\.md first|ranks worker\.md first",
        "Hybrid retrieval tuning now ranks gateway memory first for gateway queries and worker memory first for worker/browser port queries.",
    ),
    (
        r"visible browser.*9224|forwarded devtools port 9224",
        "Visible browser state is managed through the forwarded DevTools port 9224.",
    ),
    (
        r"prefers memory automation upgrades to be built before optional polish",
        "Joseph prefers memory automation upgrades to be built before optional polish.",
    ),
]

for pattern, fact in known_rules:
    if re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL):
        facts.append(fact)

# Generic fallback for unseen durable claims.
if not facts:
    candidate_lines = []
    for raw_line in text.splitlines():
        line = normalize_fact_text(raw_line)
        if not line:
            continue
        if is_low_value_line(line):
            continue
        if is_question_like(line):
            continue

        score = durable_claim_score(line)
        if score >= 6:
            candidate_lines.append((score, line))

    candidate_lines.sort(key=lambda x: x[0], reverse=True)

    seen_lines = set()
    for score, line in candidate_lines[:3]:
        canonical = normalize_fact_text(line)
        if canonical not in seen_lines:
            facts.append(canonical)
            seen_lines.add(canonical)

final = []
seen = set()
for fact in facts:
    canonical = normalize_fact_text(fact)
    if canonical and canonical not in seen:
        final.append(canonical)
        seen.add(canonical)

if not final:
    print("NONE")
else:
    for fact in final:
        item = classify_item(fact, path.name)
        print(json.dumps(item, ensure_ascii=False))
