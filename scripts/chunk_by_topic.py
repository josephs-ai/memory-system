import re
import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True)
parser.add_argument("--outdir", required=True)
args = parser.parse_args()

input_path = Path(args.input)
outdir = Path(args.outdir)
outdir.mkdir(parents=True, exist_ok=True)

text = input_path.read_text(encoding="utf-8", errors="ignore")
lines = [line.rstrip() for line in text.splitlines() if line.strip()]

def clean_role_prefix(line: str) -> str:
    return re.sub(r"^(USER|ASSISTANT):\s*", "", line, flags=re.IGNORECASE).strip()

def normalize_line(line: str) -> str:
    t = clean_role_prefix(line)
    t = re.sub(r"^(did u know that|did you know that)\s+", "", t, flags=re.IGNORECASE)
    return t.strip()

def is_header(line: str) -> bool:
    return line.strip().startswith("=====")

def is_startup_instruction(line: str) -> bool:
    t = normalize_line(line).lower()
    return (
        "a new session was started via /new or /reset" in t
        or "execute your session startup sequence" in t
        or "read the required files before responding" in t
    )

def is_chatter(line: str) -> bool:
    t = normalize_line(line).lower()
    return (
        "who am i" in t
        or "who are you" in t
        or "what should my name be" in t
        or "what should we call me" in t
        or "what should i call you" in t
        or t.startswith("hello")
        or t.startswith("hey")
        or "blank slate" in t
        or "just woke up" in t
        or "get properly introduced" in t
        or "good to know" in t
        or "i'm here" in t
        or "i am here" in t
        or "we got distracted" in t
    )

def is_durable_statement(line: str) -> bool:
    t = normalize_line(line).lower()
    if t.endswith("?"):
        return False
    durable_cues = [
        " now ",
        " now uses ",
        " uses ",
        " supports ",
        " instead of ",
        " changed ",
        " replaced ",
        " policy ",
        " decision ",
        " registry ",
        " canonical ",
        " memory ",
        " prefers ",
        " is managed through ",
        " works through ",
    ]
    padded = f" {t} "
    return any(cue in padded for cue in durable_cues)

def classify_line(line: str) -> str:
    if is_header(line):
        return "header"
    if is_startup_instruction(line):
        return "startup"
    if is_chatter(line):
        return "chatter"
    if is_durable_statement(line):
        return "durable"
    return "other"

chunks = []
current = []
current_kind = None
current_norms = set()

for line in lines:
    kind = classify_line(line)
    norm = normalize_line(line)

    if kind == "header":
        continue

    should_split = False
    if current:
        if kind != current_kind:
            should_split = True
        elif kind == "durable" and norm not in current_norms:
            should_split = True

    if should_split:
        chunks.append((current_kind, current))
        current = [line]
        current_kind = kind
        current_norms = {norm}
    else:
        current.append(line)
        current_kind = kind
        current_norms.add(norm)

if current:
    chunks.append((current_kind, current))

base = input_path.stem
for i, (kind, chunk) in enumerate(chunks, start=1):
    out = outdir / f"{base}.topic{i:02d}.{kind}.txt"
    out.write_text("\n".join(chunk) + "\n", encoding="utf-8")

print(f"chunks: {len(chunks)}")
for i, (kind, chunk) in enumerate(chunks, start=1):
    print(f"{i:02d}: kind={kind} lines={len(chunk)}")
