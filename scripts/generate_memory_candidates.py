import re
import json
import hashlib
import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("input_path")
parser.add_argument("--source-agent", default="unknown")
parser.add_argument("--source-session", default="unknown")
parser.add_argument("--source-chunk", default=None)
args = parser.parse_args()

path = Path(args.input_path)
text = path.read_text(encoding="utf-8", errors="ignore")


def make_candidate_id(source_name: str, text: str) -> str:
    h = hashlib.sha1(f"{source_name}|{text}".encode("utf-8")).hexdigest()[:12]
    return f"cand-{h}"


def clean_line(s: str) -> str:
    s = s.strip()
    s = re.sub(r"^(USER|ASSISTANT):\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"^(did u know that|did you know that)\s+", "", s, flags=re.IGNORECASE)
    s = re.sub(r"^(just so you know)\s+", "", s, flags=re.IGNORECASE)
    return s.strip()


def normalize_text(s: str) -> str:
    s = clean_line(s)
    s = re.sub(r"\s+", " ", s).strip()
    if s:
        s = s[0].upper() + s[1:]
    return s


def is_question_like(s: str) -> bool:
    t = s.lower().strip()
    question_starts = (
        "who ", "what ", "when ", "where ", "why ", "how ",
        "hello", "hey"
    )
    return (
        t.endswith("?")
        or t.startswith(question_starts)
        or "who am i" in t
        or "who are you" in t
        or "what should we call" in t
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
        r"i am here",
    ]
    return any(re.search(p, t) for p in low_patterns)


def is_speculative(s: str) -> bool:
    t = s.lower().strip()
    speculative_patterns = [
        r"\bmaybe\b",
        r"\bmight\b",
        r"\bcould\b",
        r"\bperhaps\b",
        r"\bprobably\b",
        r"\bi think\b",
        r"\btry\b",
        r"\bfor example\b",
        r"\bexample\b",
    ]
    return any(re.search(p, t) for p in speculative_patterns)


def candidate_score(s: str) -> tuple[int, list[str]]:
    t = s.lower().strip()
    score = 0
    reasons = []

    # general durable-statement cues only
    positive = [
        (r"\bnow\b", 2, "temporal_update"),
        (r"\buses\b", 3, "uses"),
        (r"\bsupports\b", 3, "supports"),
        (r"\bprefers\b", 5, "preference"),
        (r"\bpreference\b", 4, "preference"),
        (r"\bconfigured\b", 3, "configured"),
        (r"\bchanged\b", 3, "changed"),
        (r"\breplaced\b", 3, "replaced"),
        (r"\binstead of\b", 4, "contrast"),
        (r"\benabled\b", 3, "enabled"),
        (r"\bdisabled\b", 3, "disabled"),
        (r"\bcurrent\b", 2, "current_state"),
        (r"\bworks through\b", 3, "mechanism"),
        (r"\bis managed through\b", 3, "mechanism"),
        (r"\bpolicy\b", 3, "policy"),
        (r"\bdecision\b", 3, "decision"),
        (r"\bobserved\b", 2, "observation"),
        (r"\bone-off\b", 2, "one_off"),
        (r"\btest output\b", 3, "test_output"),
        (r"\bcheckpoint run\b", 2, "checkpoint_run"),
        (r"\bmismatch\b", 2, "mismatch"),
    ]

    negative = [
        (r"\?$", -6, "question"),
        (r"\bhello\b", -5, "greeting"),
        (r"\bhey\b", -4, "greeting"),
        (r"\bwho am i\b", -8, "startup_chatter"),
        (r"\bwho are you\b", -8, "startup_chatter"),
        (r"\bwhat should\b", -6, "question"),
        (r"\bcall me\b", -5, "startup_chatter"),
        (r"\bcall you\b", -5, "startup_chatter"),
    ]

    for pattern, pts, reason in positive:
        if re.search(pattern, t):
            score += pts
            reasons.append(reason)

    for pattern, pts, reason in negative:
        if re.search(pattern, t):
            score += pts
            reasons.append(reason)

    if not is_question_like(s) and len(t.split()) >= 6:
        score += 2
        reasons.append("declarative_length")

    if is_speculative(s):
        score -= 4
        reasons.append("speculative")

    return score, reasons


candidates = []
seen = set()

for raw_line in text.splitlines():
    line = normalize_text(raw_line)
    if not line:
        continue
    if is_low_value_line(line):
        continue
    if is_question_like(line):
        continue

    score, reasons = candidate_score(line)
    if score < 5:
        continue

    if line in seen:
        continue
    seen.add(line)

    candidates.append({
        "candidate_id": make_candidate_id(path.name, line),
        "raw_text": raw_line.strip(),
        "normalized_text": line,
        "source_agent": args.source_agent,
        "source_session": args.source_session,
        "source_chunk": args.source_chunk or path.name,
        "candidate_score": score,
        "candidate_reasons": reasons,
    })

if not candidates:
    print("NONE")
else:
    for cand in candidates:
        print(json.dumps(cand, ensure_ascii=False))

