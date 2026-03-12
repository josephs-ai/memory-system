from pathlib import Path
from datetime import datetime, timedelta
import shutil
import re

WORKSPACE = Path.home() / ".openclaw" / "workspace"
MEMORY_DIR = WORKSPACE / "memory"
DAILY_DIR = MEMORY_DIR / "daily"
ARCHIVE_DAILY_DIR = MEMORY_DIR / "archive" / "daily"
ARCHIVE_DAILY_DIR.mkdir(parents=True, exist_ok=True)

KEEP_DAYS = 7

date_re = re.compile(r"^\d{4}-\d{2}-\d{2}\.md$")

def main():
    cutoff = datetime.now().date() - timedelta(days=KEEP_DAYS)
    moved = []

    for path in sorted(DAILY_DIR.glob("*.md")):
        if not date_re.match(path.name):
            continue
        try:
            d = datetime.strptime(path.stem, "%Y-%m-%d").date()
        except ValueError:
            continue

        if d < cutoff:
            target = ARCHIVE_DAILY_DIR / path.name
            shutil.move(str(path), str(target))
            moved.append((path, target))

    if not moved:
        print("No daily files archived.")
        return

    print("Archived daily files:")
    for src, dst in moved:
        print(f"- {src.name} -> {dst}")

if __name__ == "__main__":
    main()
