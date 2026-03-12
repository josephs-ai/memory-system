import os
from pathlib import Path

WORKSPACE = Path(os.path.expanduser("~/.openclaw/workspace"))
MEMORY = WORKSPACE / "memory"

stable_files = [
    MEMORY / "gateway.md",
    MEMORY / "worker.md",
    MEMORY / "browser.md",
    MEMORY / "preferences.md",
    MEMORY / "learned-fixes.md",
]

daily_dir = MEMORY / "daily"
project_dir = MEMORY / "projects"

def read_lines(path):
    if not path.exists():
        return []
    lines = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        s = line.strip()
        if s and not s.startswith("#"):
            lines.append(s)
    return lines

stable_lines = set()
for f in stable_files:
    for line in read_lines(f):
        stable_lines.add(line)

print("=== Duplicate lines already covered by stable memory ===\n")

for folder in [daily_dir, project_dir]:
    for path in sorted(folder.rglob("*.md")):
        dups = []
        for line in read_lines(path):
            if line in stable_lines:
                dups.append(line)

        if dups:
            print(f"{path}")
            for d in dups:
                print(f"  DUPLICATE: {d}")
            print()

print("=== Compaction scan complete ===")
