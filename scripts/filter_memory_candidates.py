import re
import json
import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("input_path")
args = parser.parse_args()

path = Path(args.input_path)


def load_jsonl(path: Path):
    items = []
    if not path.exists():
        return items
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s or s == "NONE":
                continue
            try:
                items.append(json.loads(s))
            except Exception:
                pass
    return items


def should_suppress(text: str) -> tuple[bool, str]:
    t = text.lower().strip()

    suppression_patterns = [
        (r"a new session was started via /new or /reset", "startup_instruction"),
        (r"execute your session startup sequence", "startup_instruction"),
        (r"read the required files before responding", "startup_instruction"),
        (r"greet the user in your configured persona", "startup_instruction"),
        (r"do not mention internal steps, files, tools, or reasoning", "system_instruction"),
        (r"keep it to 1-3 sentences", "system_instruction"),
        (r"mention the default model", "system_instruction"),
        (r"be yourself - use your defined voice", "system_instruction"),
        (r"ask what they want to do", "system_instruction"),
        (r"example", "example_text"),
        (r"for example", "example_text"),
        (r"maybe try", "speculative"),
    ]

    for pattern, reason in suppression_patterns:
        if re.search(pattern, t):
            return True, reason

    return False, ""


items = load_jsonl(path)

if not items:
    print("NONE")
else:
    kept = []
    for item in items:
        suppress, reason = should_suppress(item.get("normalized_text", ""))
        if not suppress:
            kept.append(item)

    if not kept:
        print("NONE")
    else:
        for item in kept:
            print(json.dumps(item, ensure_ascii=False))
