import argparse
from collections import Counter
from pathlib import Path


def load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def collect_bullets(text: str):
    bullets = []
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("- "):
            bullets.append(s)
    return bullets


def normalize_topic(line: str) -> str:
    s = line.lower()
    if "[docs]" in s:
        return "docs"
    if "[untracked]" in s:
        return "untracked"
    if "[top_dirs]" in s:
        return "top_dirs"
    if "[memory_system]" in s:
        return "memory_system"
    if "[browser]" in s:
        return "browser"
    if "[worker]" in s:
        return "worker"
    if "[gateway]" in s:
        return "gateway"
    return "other"


def unique_preserve_order(lines):
    seen = set()
    out = []
    for line in lines:
        if line in seen:
            continue
        seen.add(line)
        out.append(line)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-dir", required=True)
    parser.add_argument("--month", required=True, help="YYYY-MM")
    args = parser.parse_args()

    project_dir = Path(args.project_dir).expanduser().resolve()
    daily_dir = project_dir / "daily"
    out_file = project_dir / f"milestone-{args.month}.md"

    files = sorted(daily_dir.glob(f"{args.month}-*.md"))
    if not files:
        print("NONE")
        return

    all_bullets = []
    for f in files:
        all_bullets.extend(collect_bullets(load_text(f)))

    if not all_bullets:
        print("NONE")
        return

    all_bullets = unique_preserve_order(all_bullets)

    topic_counts = Counter(normalize_topic(x) for x in all_bullets)

    lines = []
    lines.append(f"## Milestone summary for {args.month}")
    lines.append("")
    lines.append("### Topic counts")
    lines.append("")
    for topic, count in topic_counts.most_common():
        lines.append(f"- {topic}: {count}")
    lines.append("")
    lines.append("### Consolidated highlights")
    lines.append("")

    for bullet in all_bullets[:50]:
        lines.append(bullet)

    text = "\n".join(lines) + "\n"
    out_file.write_text(text, encoding="utf-8")

    print("CONSOLIDATED")
    print(f"project_dir={project_dir}")
    print(f"month={args.month}")
    print(f"daily_files={len(files)}")
    print(f"output={out_file}")


if __name__ == "__main__":
    main()
