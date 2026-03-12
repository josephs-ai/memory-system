import argparse
from pathlib import Path

WORKSPACE = Path.home() / ".openclaw" / "workspace"
GLOBAL_INSIGHTS_DIR = WORKSPACE / "memory" / "global-insights"


def score_text(query: str, text: str) -> int:
    q_terms = [x.lower() for x in query.split() if x.strip()]
    t = (text or "").lower()
    score = 0
    for term in q_terms:
        if term in t:
            score += 1
    return score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", required=True)
    args = parser.parse_args()

    if not GLOBAL_INSIGHTS_DIR.exists():
        print("NONE")
        return

    rows = []
    for path in sorted(GLOBAL_INSIGHTS_DIR.glob("*.md")):
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            s = line.strip()
            if not s.startswith("- "):
                continue
            score = score_text(args.query, s)
            if score > 0:
                rows.append((score, path, s))

    rows.sort(key=lambda x: (-x[0], str(x[1])))

    if not rows:
        print("NONE")
        return

    for score, path, line in rows[:10]:
        print(f"{score}\t{path}\t{line}")


if __name__ == "__main__":
    main()
