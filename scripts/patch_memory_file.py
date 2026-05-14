"""
Memory system utility: patch memory file.

Key functions: load_json, ensure_section, section_range, append_under_heading
"""
import json
import argparse
from pathlib import Path

def load_json(path: str):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def ensure_section(lines, heading):
    header = f"## {heading}"
    for i, line in enumerate(lines):
        if line.strip() == header:
            return i
    if lines and lines[-1].strip() != "":
        lines.append("")
    lines.append(header)
    return len(lines) - 1

def section_range(lines, heading):
    start = None
    end = len(lines)
    target = f"## {heading}"

    for i, line in enumerate(lines):
        if line.strip() == target:
            start = i
            break

    if start is None:
        return None, None

    for j in range(start + 1, len(lines)):
        if lines[j].startswith("## "):
            end = j
            break

    return start, end

def append_under_heading(lines, heading, bullet_text):
    ensure_section(lines, heading)
    start, end = section_range(lines, heading)
    insert_at = end
    if insert_at > 0 and lines[insert_at - 1].strip() != "":
        lines.insert(insert_at, "")
        insert_at += 1
    lines.insert(insert_at - 1 if insert_at > start + 1 else insert_at, f"- {bullet_text}")

def replace_bullet(lines, old_text, new_text):
    old_line = f"- {old_text}"
    for i, line in enumerate(lines):
        if line.strip() == old_line:
            lines[i] = f"- {new_text}"
            return True
    return False

def remove_bullet(lines, old_text):
    old_line = f"- {old_text}"
    for i, line in enumerate(lines):
        if line.strip() == old_line:
            lines.pop(i)
            return True
    return False

def append_superseded(lines, old_text):
    ensure_section(lines, "Superseded")
    start, end = section_range(lines, "Superseded")
    insert_at = end
    lines.insert(insert_at, f"- {old_text}")
    return True

def normalize(lines):
    out = []
    prev_blank = False
    for line in lines:
        blank = (line.strip() == "")
        if blank and prev_blank:
            continue
        out.append(line)
        prev_blank = blank
    return out

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True)
    parser.add_argument("--outcome", required=True)
    parser.add_argument("--candidate-item", required=True)
    parser.add_argument("--existing-item")
    args = parser.parse_args()

    target = Path(args.target)
    candidate = load_json(args.candidate_item)
    existing = load_json(args.existing_item) if args.existing_item else None

    lines = target.read_text(encoding="utf-8").splitlines()

    candidate_text = candidate["text"]
    existing_text = existing["text"] if existing else None

    outcome = args.outcome

    if outcome == "IGNORE_DUPLICATE":
        print("NO_CHANGE duplicate")
        return

    if outcome == "APPEND_NEW":
        append_under_heading(lines, "Active", candidate_text)
        target.write_text("\n".join(normalize(lines)) + "\n", encoding="utf-8")
        print("PATCHED append_new")
        return

    if outcome == "MERGE_INTO_EXISTING":
        if not existing_text:
            print("ERROR existing item required for merge")
            return
        ok = replace_bullet(lines, existing_text, candidate_text)
        if not ok:
            print("ERROR existing bullet not found for merge")
            return
        target.write_text("\n".join(normalize(lines)) + "\n", encoding="utf-8")
        print("PATCHED merge_into_existing")
        return

    if outcome == "SUPERSEDE_EXISTING":
        if not existing_text:
            print("ERROR existing item required for supersede")
            return
        ok = remove_bullet(lines, existing_text)
        if not ok:
            print("ERROR existing bullet not found for supersede")
            return
        append_under_heading(lines, "Active", candidate_text)
        append_superseded(lines, existing_text)
        target.write_text("\n".join(normalize(lines)) + "\n", encoding="utf-8")
        print("PATCHED supersede_existing")
        return

    if outcome == "FLAG_CONFLICT":
        print("NO_CHANGE conflict")
        return

    print(f"ERROR unknown outcome: {outcome}")

if __name__ == "__main__":
    main()
