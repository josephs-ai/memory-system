import argparse
from datetime import datetime, timezone
from pathlib import Path

WORKSPACE = Path.home() / ".openclaw" / "workspace"
MEMORY_DIR = WORKSPACE / "memory"
GLOBAL_INSIGHTS_DIR = MEMORY_DIR / "global-insights"
LOG_FILE = WORKSPACE / ".memory-index" / "logs" / "global-insights.log"

GLOBAL_INSIGHTS_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)


def now_iso():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def append_log(line: str):
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def append_if_new(path: Path, line: str):
    existing = path.read_text(encoding="utf-8", errors="ignore") if path.exists() else ""
    if line in existing:
        return False
    if existing and not existing.endswith("\n"):
        existing += "\n"
    existing += line + "\n"
    path.write_text(existing, encoding="utf-8")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--topic", required=True, help="e.g. python, browser, memory-system")
    parser.add_argument("--text", required=True)
    parser.add_argument("--source-project")
    parser.add_argument("--promoted-by", default="human")
    args = parser.parse_args()

    topic = args.topic.strip().lower().replace(" ", "-")
    out_file = GLOBAL_INSIGHTS_DIR / f"{topic}.md"

    line = f"- {args.text}"
    if args.source_project:
        line += f" [source_project={args.source_project}]"

    written = append_if_new(out_file, line)

    append_log(
        f"{now_iso()} topic={topic} promoted_by={args.promoted_by} "
        f"written={int(written)} source_project={args.source_project or 'none'} text={args.text}"
    )

    print("PROMOTED" if written else "ALREADY_PRESENT")
    print(f"file={out_file}")


if __name__ == "__main__":
    main()
