"""
Memory system utility: ingest workspace state memory.

Key functions: load_json, existing_text, append_block_if_new, normalize_summary_line
"""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from project_memory_paths import (
    get_project_current_file,
    get_project_daily_file,
)

def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))

def existing_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="ignore")

def append_block_if_new(path: Path, header: str, lines: list[str]):
    if not lines:
        return False

    text = existing_text(path)

    if all(line in text for line in lines):
        return False

    block = [header, ""]
    block.extend(lines)
    block.append("")

    if text and not text.endswith("\n"):
        text += "\n"

    text += "\n".join(block)
    path.write_text(text, encoding="utf-8")
    return True

def normalize_summary_line(item: dict):
    text = item.get("text", "").strip()
    category = item.get("category", "").strip()
    evidence = item.get("evidence", [])

    if not text:
        return None

    line = f"- [{category}] {text}"
    if evidence:
        preview = ", ".join(str(x) for x in evidence[:6])
        line += f" Evidence: {preview}"
    return line

def unique_preserve_order(lines):
    seen = set()
    out = []
    for line in lines:
        if not line or line in seen:
            continue
        seen.add(line)
        out.append(line)
    return out

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", required=True)
    parser.add_argument("--project-id", required=True)
    args = parser.parse_args()

    summary_path = Path(args.summary).expanduser().resolve()
    data = load_json(summary_path)

    captured_at = data.get("captured_at", "")
    summaries = data.get("summaries", [])

    project_lines = []
    daily_lines = []

    for item in summaries:
        scope = item.get("scope", "").strip()
        line = normalize_summary_line(item)
        if line is None:
            continue

        if scope == "project":
            project_lines.append(line)
        elif scope == "daily":
            daily_lines.append(line)

    project_lines = unique_preserve_order(project_lines)
    daily_lines = unique_preserve_order(daily_lines)

    project_file = get_project_current_file(args.project_id)
    daily_file = get_project_daily_file(args.project_id)

    project_written = append_block_if_new(
        project_file,
        f"## Workspace update {captured_at}",
        project_lines,
    )

    daily_written = append_block_if_new(
        daily_file,
        f"## Workspace update {captured_at}",
        daily_lines,
    )

    print(f"PROJECT_ID={args.project_id}")
    print(f"PROJECT_LINES={len(project_lines)}")
    print(f"DAILY_LINES={len(daily_lines)}")
    print(f"PROJECT_WRITTEN={int(project_written)}")
    print(f"DAILY_WRITTEN={int(daily_written)}")
    print(f"PROJECT_FILE={project_file}")
    print(f"DAILY_FILE={daily_file}")

if __name__ == "__main__":
    main()
