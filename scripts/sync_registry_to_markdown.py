import sys
from collections import defaultdict
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from memory_db import fetch_memory_items_by_status, close_pool

WORKSPACE = Path.home() / ".openclaw" / "workspace"
MEMORY_DIR = WORKSPACE / "memory"


def group_items(items):
    grouped = defaultdict(list)
    for item in items:
        target_file = item.get("target_file")
        if not target_file:
            continue
        grouped[target_file].append(item)
    return grouped


def render_file(active_items, superseded_items):
    lines = []
    lines.append("## Active")
    lines.append("")

    if active_items:
        for item in active_items:
            lines.append(f"- {item.get('text', '').strip()}")
    else:
        lines.append("(none)")

    lines.append("")
    lines.append("## Superseded")
    lines.append("")

    if superseded_items:
        for item in superseded_items:
            lines.append(f"- {item.get('text', '').strip()}")
    else:
        lines.append("(none)")

    lines.append("")
    return "\n".join(lines)


def main():
    active = fetch_memory_items_by_status("active")
    superseded = fetch_memory_items_by_status("superseded")

    active_grouped = group_items(active)
    superseded_grouped = group_items(superseded)

    all_files = sorted(set(active_grouped.keys()) | set(superseded_grouped.keys()))
    changed = 0

    for rel_name in all_files:
        out_path = MEMORY_DIR / rel_name
        out_path.parent.mkdir(parents=True, exist_ok=True)

        text = render_file(
            active_grouped.get(rel_name, []),
            superseded_grouped.get(rel_name, []),
        )

        existing = out_path.read_text(encoding="utf-8", errors="ignore") if out_path.exists() else ""
        if existing != text:
            out_path.write_text(text, encoding="utf-8")
            changed += 1

    print(f"SYNCED_FILES={changed}")
    close_pool()


if __name__ == "__main__":
    main()
